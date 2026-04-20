import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from shared.preprocessing         import load_xml_files, ID2LABEL, NUM_LABELS, TRAIN_DIR, TEST_DIR
from shared.evaluate              import evaluate
from track_b.drug_smiles          import get_all_drug_smiles
from track_b.chemberta_embeddings import build_drug_embeddings, EMB_DIM
from track_b.dataset              import (ChemGATDataset, build_chem_gat_examples,
                                           load_spacy, load_biobert,
                                           MAX_SDP_NODES, NUM_DEP_TYPES)
from track_b.train                import train_model


# ── R-GAT Layer ───────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Relational Multi-head Graph Attention Layer.

    Extends standard GAT with per-edge dependency-type embeddings (R-GAT):
      e_ij = LeakyReLU( a1(W·h_i) + a2(W·h_j) + edge_emb[dep_type_ij] )
      α_ij = softmax_j( e_ij )
      h'_i = ELU( LayerNorm( concat_heads( Σ_j α_ij · W·h_j ) ) )

    edge_emb is an Embedding table over dependency relation types so the model
    learns that nsubj→verb→dobj paths mean something different from prep→pobj.
    """

    def __init__(self, in_features, out_features, num_heads=8, dropout=0.5,
                 num_edge_types=NUM_DEP_TYPES):
        super().__init__()
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = out_features // num_heads
        self.dropout   = dropout

        self.W        = nn.Linear(in_features, out_features, bias=False)
        self.a1       = nn.Linear(self.head_dim, 1, bias=False)
        self.a2       = nn.Linear(self.head_dim, 1, bias=False)
        self.edge_emb = nn.Embedding(num_edge_types, num_heads, padding_idx=0)
        self.ln       = nn.LayerNorm(out_features)

    def forward(self, x, adj, edge_types):
        # x          : (B, N, in_features)
        # adj        : (B, N, N)
        # edge_types : (B, N, N)  long tensor of dependency relation IDs
        B, N, _ = x.shape

        Wh = self.W(x).view(B, N, self.num_heads, self.head_dim)   # (B, N, H, D)

        a1 = self.a1(Wh).squeeze(-1)   # (B, N, H)
        a2 = self.a2(Wh).squeeze(-1)   # (B, N, H)

        # Node attention: e[b,i,j,h] = a1[b,i,h] + a2[b,j,h]
        e = a1.unsqueeze(2) + a2.unsqueeze(1)              # (B, N, N, H)

        # Edge type bias: edge_emb[dep_type] → (B, N, N, H)
        e = e + self.edge_emb(edge_types)

        e = F.leaky_relu(e, negative_slope=0.2)

        # Mask non-edges
        e = e.masked_fill(adj.unsqueeze(-1) == 0, -1e9)    # (B, N, N, H)

        alpha = F.softmax(e, dim=2)                         # (B, N, N, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        h = torch.einsum("bijh,bjhd->bihd", alpha, Wh)     # (B, N, H, D)
        h = h.reshape(B, N, self.num_heads * self.head_dim) # (B, N, out_features)

        return F.elu(self.ln(h))


# ── ChemBERTa-R-GAT Model ─────────────────────────────────────────────────────

class ChemGAT(nn.Module):
    """
    Chemically-Aware Relational Graph Attention Network for DDI classification.

    Node features:
      Drug nodes  → 768-dim ChemBERTa embeddings (molecular structure)
      Other nodes → 768-dim frozen BioBERT token embeddings (biomedical context)

    Edge features:
      Each directed edge carries the spaCy dependency relation type (nsubj,
      dobj, prep, etc.) as a learned embedding inside R-GAT attention.

    Architecture:
      Input projection  : 768 → hidden_dim
      4 × GATLayer      : hidden_dim → hidden_dim  (8 heads, R-GAT)
      Extract Drug1 + Drug2 embeddings → concatenate (hidden_dim × 2)
      MLP classifier    : hidden_dim × 2 → hidden_dim → 5
    """

    def __init__(
        self,
        node_dim   = EMB_DIM,
        hidden_dim = 512,
        num_heads  = 8,
        num_layers = 4,
        dropout    = 0.5,
        num_labels = NUM_LABELS,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, node_features, adj, edge_types, e1_pos, e2_pos):
        x = self.input_proj(node_features)        # (B, N, hidden_dim)
        for layer in self.gat_layers:
            x = self.dropout(layer(x, adj, edge_types))
        batch_size = x.size(0)
        e1_emb = x[torch.arange(batch_size), e1_pos, :]
        e2_emb = x[torch.arange(batch_size), e2_pos, :]
        return self.classifier(torch.cat([e1_emb, e2_emb], dim=-1))


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "node_dim":    EMB_DIM,   # 768
    "hidden_dim":  512,
    "num_heads":   8,
    "num_layers":  4,
    "dropout":     0.5,
    "epochs":      45,
    "batch_size":  64,
    "lr":          5e-4,
    "weight_decay": 5e-4,
    "val_frac":    0.1,
    "demo_mode":        False,
    "demo_train_size":  1500,
    "demo_test_size":   300,
    "device": (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    ),
    "seed": 42,
}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(CONFIG["seed"])
    print(f"\nDevice : {CONFIG['device']}")
    print("Model  : ChemBERTa-R-GAT  (ChemBERTa drug nodes + frozen BioBERT non-drug nodes + R-GAT edge types)\n")

    if CONFIG["demo_mode"]:
        print("*** DEMO MODE — set demo_mode=False for full training ***\n")

    # ── 1. Drug SMILES ────────────────────────────────────────────────────────
    print("[1/7] Loading drug SMILES from PubChem ...")
    drug_smiles = get_all_drug_smiles()

    # ── 2. ChemBERTa embeddings ───────────────────────────────────────────────
    print("\n[2/7] Generating ChemBERTa drug embeddings ...")
    drug_embeddings = build_drug_embeddings(drug_smiles, device=CONFIG["device"])

    # ── 3. Frozen BioBERT ─────────────────────────────────────────────────────
    print("\n[3/7] Loading frozen BioBERT for non-drug node features ...")
    biobert_model, biobert_tokenizer = load_biobert(device=CONFIG["device"])

    # ── 4. SDP graphs ─────────────────────────────────────────────────────────
    print("\n[4/7] Loading spaCy + building R-GAT SDP graphs ...")
    nlp       = load_spacy()
    train_raw = load_xml_files(TRAIN_DIR)
    test_raw  = load_xml_files(TEST_DIR)

    if CONFIG["demo_mode"]:
        rng = random.Random(CONFIG["seed"])
        rng.shuffle(train_raw); rng.shuffle(test_raw)
        train_raw = train_raw[:CONFIG["demo_train_size"]]
        test_raw  = test_raw[:CONFIG["demo_test_size"]]

    train_examples = build_chem_gat_examples(
        train_raw, nlp, drug_embeddings,
        biobert_model, biobert_tokenizer, CONFIG["device"], cache_key="train",
    )
    test_examples = build_chem_gat_examples(
        test_raw, nlp, drug_embeddings,
        biobert_model, biobert_tokenizer, CONFIG["device"], cache_key="test",
    )

    # Free BioBERT from GPU/MPS memory — no longer needed after graph building
    del biobert_model
    if CONFIG["device"] == "mps":
        torch.mps.empty_cache()
    elif CONFIG["device"] == "cuda":
        torch.cuda.empty_cache()

    print(f"       Train: {len(train_examples)}  |  Test: {len(test_examples)}")
    dist = Counter(ex["label"] for ex in train_examples)
    for lid, cnt in sorted(dist.items()):
        print(f"         {ID2LABEL[lid]:12s}: {cnt:6d}  ({100 * cnt / len(train_examples):.1f}%)")

    # ── 5. Datasets ───────────────────────────────────────────────────────────
    print("\n[5/7] Building datasets ...")
    full_train = ChemGATDataset(train_examples)
    test_ds    = ChemGATDataset(test_examples)
    val_size   = int(len(full_train) * CONFIG["val_frac"])
    train_ds, val_ds = random_split(
        full_train, [len(full_train) - val_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    train_ds.examples = [full_train.examples[i] for i in train_ds.indices]
    val_ds.examples   = [full_train.examples[i] for i in val_ds.indices]
    print(f"       Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

    # ── 6. Model ──────────────────────────────────────────────────────────────
    print("\n[6/7] Building ChemBERTa-R-GAT model ...")
    model = ChemGAT(
        node_dim   = CONFIG["node_dim"],
        hidden_dim = CONFIG["hidden_dim"],
        num_heads  = CONFIG["num_heads"],
        num_layers = CONFIG["num_layers"],
        dropout    = CONFIG["dropout"],
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Total params    : {total:,}")
    print(f"       Trainable params: {trainable:,}")

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print(f"\n[7/7] Training for {CONFIG['epochs']} epochs ...")
    model = train_model(model, train_ds, val_ds, CONFIG)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[8/8] Evaluating on test set ...")
    results = evaluate(model, test_ds, CONFIG, model_type="gat")
    print(f"\nFinal macro-F1 (positive classes): {results['macro_f1']:.4f}")

    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "track_b_chemgat_best.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
