import pickle
import numpy as np
import torch
from collections import deque, defaultdict
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

from shared.preprocessing import load_xml_files, LABEL2ID, ID2LABEL, NUM_LABELS
from track_b.chemberta_embeddings import EMB_DIM

MAX_SDP_NODES = 20
NODE_DIM      = EMB_DIM   # 768
SPACY_DIM     = 300
CACHE_PATH    = Path(__file__).parent / "graph_cache.pkl"
BIOBERT_ID    = "dmis-lab/biobert-base-cased-v1.1"

# Dependency relation vocabulary for R-GAT edge embeddings
_DEP_VOCAB = [
    "<pad>", "<self>",
    "ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos",
    "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj",
    "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl",
    "intj", "mark", "meta", "neg", "nmod", "nn", "npmod", "nsubj",
    "nsubjpass", "nummod", "oprd", "parataxis", "pcomp", "pobj", "poss",
    "preconj", "predet", "prep", "prt", "punct", "quantmod", "relcl",
    "xcomp", "<unk>",
]
DEP2ID        = {d: i for i, d in enumerate(_DEP_VOCAB)}
NUM_DEP_TYPES = len(_DEP_VOCAB)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        spacy.cli.download("en_core_web_md")
        return spacy.load("en_core_web_md")


def load_biobert(device="cpu"):
    print(f"  Loading BioBERT ({BIOBERT_ID}) — frozen feature extractor ...")
    # AutoTokenizer fails on dmis-lab BioBERT due to a broken sentencepiece config.
    # BertTokenizer bypasses this — vocabulary is identical to bert-base-cased.
    tokenizer = BertTokenizer.from_pretrained(BIOBERT_ID)
    model     = BertModel.from_pretrained(BIOBERT_ID).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


# ── Utilities ─────────────────────────────────────────────────────────────────

def _char_to_token(doc, char_start):
    for token in doc:
        if token.idx >= char_start:
            return min(token.i, len(doc) - 1)
    return len(doc) - 1


def _biobert_token_emb(spacy_token, hidden_states, offset_mapping):
    """
    Mean-pool BioBERT subword embeddings that overlap with a spaCy token's
    character span. Falls back to zero vector if no overlap found.
    """
    start = spacy_token.idx
    end   = spacy_token.idx + len(spacy_token.text)
    idxs  = [
        i for i, (s, e) in enumerate(offset_mapping)
        if not (s == 0 and e == 0) and s < end and e > start
    ]
    if idxs:
        return hidden_states[idxs].mean(axis=0)
    return np.zeros(EMB_DIM, dtype=np.float32)


def _extract_sdp(doc, e1_idx, e2_idx):
    if e1_idx == e2_idx:
        return [e1_idx]

    n         = len(doc)
    neighbors = defaultdict(set)
    for token in doc:
        i, j = min(token.i, n - 1), min(token.head.i, n - 1)
        if i != j:
            neighbors[i].add(j)
            neighbors[j].add(i)

    visited = {e1_idx}
    queue   = deque([(e1_idx, [e1_idx])])
    while queue:
        node, path = queue.popleft()
        for nb in neighbors[node]:
            if nb not in visited:
                new_path = path + [nb]
                if nb == e2_idx:
                    return new_path[:MAX_SDP_NODES]
                visited.add(nb)
                queue.append((nb, new_path))

    return [e1_idx, e2_idx]


def _build_graph(doc, e1_idx, e2_idx, drug_embeddings, e1_text, e2_text,
                 biobert_hidden, offset_mapping):
    """
    Build the SDP subgraph with:
      drug nodes     → 768-dim ChemBERTa chemical embedding
      non-drug nodes → 768-dim frozen BioBERT token embedding (contextual, biomedical)
      edge_types     → dependency relation ID per edge (for R-GAT)

    Returns (padded_adj, padded_edge_types, padded_features, e1_sdp_pos, e2_sdp_pos)
    """
    sdp_nodes = _extract_sdp(doc, e1_idx, e2_idx)
    n         = len(sdp_nodes)
    idx_map   = {tok: i for i, tok in enumerate(sdp_nodes)}

    # Adjacency (bidirectional + self-loops, row-normalised)
    adj        = np.zeros((n, n), dtype=np.float32)
    edge_types = np.zeros((n, n), dtype=np.int64)

    for tok in sdp_nodes:
        i = idx_map[tok]
        adj[i, i]        = 1.0
        edge_types[i, i] = DEP2ID["<self>"]

        head = doc[tok].head.i
        if head in idx_map:
            j        = idx_map[head]
            dep_id   = DEP2ID.get(doc[tok].dep_.lower(), DEP2ID["<unk>"])
            adj[i, j]        = 1.0
            adj[j, i]        = 1.0
            edge_types[i, j] = dep_id
            edge_types[j, i] = dep_id

    deg = adj.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    adj /= deg

    # Node features: ChemBERTa for drug nodes, frozen BioBERT for everything else
    features = np.zeros((n, NODE_DIM), dtype=np.float32)
    for i, tok in enumerate(sdp_nodes):
        if tok == e1_idx:
            emb = drug_embeddings.get(e1_text.lower())
            features[i] = emb if emb is not None else np.zeros(NODE_DIM, dtype=np.float32)
        elif tok == e2_idx:
            emb = drug_embeddings.get(e2_text.lower())
            features[i] = emb if emb is not None else np.zeros(NODE_DIM, dtype=np.float32)
        else:
            features[i] = _biobert_token_emb(doc[tok], biobert_hidden, offset_mapping)

    e1_sdp = idx_map.get(e1_idx, 0)
    e2_sdp = idx_map.get(e2_idx, min(1, n - 1))

    # Pad to MAX_SDP_NODES
    padded_adj   = np.zeros((MAX_SDP_NODES, MAX_SDP_NODES), dtype=np.float32)
    padded_etypes = np.zeros((MAX_SDP_NODES, MAX_SDP_NODES), dtype=np.int64)
    padded_feat  = np.zeros((MAX_SDP_NODES, NODE_DIM),      dtype=np.float32)
    padded_adj[:n, :n]    = adj
    padded_etypes[:n, :n] = edge_types
    padded_feat[:n]       = features

    return padded_adj, padded_etypes, padded_feat, e1_sdp, e2_sdp


# ── Graph dataset builder ─────────────────────────────────────────────────────

def build_chem_gat_examples(raw_examples, nlp, drug_embeddings,
                             biobert_model, biobert_tokenizer, device, cache_key):
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache_key in cache and len(cache[cache_key]) > 0:
            if "edge_types" in cache[cache_key][0]:
                print(f"  Loaded graph cache  ({len(cache[cache_key])} examples)")
                return cache[cache_key]
            print("  Graph cache outdated (no edge_types) — rebuilding ...")
            del cache[cache_key]
    else:
        cache = {}

    examples = []
    for ex in tqdm(raw_examples, desc=f"  Building SDP graphs [{cache_key}]"):
        doc    = nlp(ex["text"])
        e1_idx = _char_to_token(doc, ex["e1_start"])
        e2_idx = _char_to_token(doc, ex["e2_start"])

        # Frozen BioBERT forward pass for this sentence
        enc = biobert_tokenizer(
            ex["text"],
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offset_mapping = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            out = biobert_model(**{k: v.to(device) for k, v in enc.items()})
        hidden = out.last_hidden_state[0].cpu().numpy()  # (seq_len, 768)

        adj, edge_types, features, e1_pos, e2_pos = _build_graph(
            doc, e1_idx, e2_idx,
            drug_embeddings, ex["e1_text"], ex["e2_text"],
            hidden, offset_mapping,
        )
        examples.append({
            "adj":           adj,
            "edge_types":    edge_types,
            "node_features": features,
            "e1_pos":        e1_pos,
            "e2_pos":        e2_pos,
            "label":         ex["label"],
            "e1_text":       ex["e1_text"],
            "e2_text":       ex["e2_text"],
        })

    cache[cache_key] = examples
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Cached {len(examples)} examples → {CACHE_PATH}")
    return examples


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ChemGATDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "adj":           torch.tensor(ex["adj"],           dtype=torch.float),
            "edge_types":    torch.tensor(ex["edge_types"],    dtype=torch.long),
            "node_features": torch.tensor(ex["node_features"], dtype=torch.float),
            "e1_pos":        torch.tensor(ex["e1_pos"],        dtype=torch.long),
            "e2_pos":        torch.tensor(ex["e2_pos"],        dtype=torch.long),
            "label":         torch.tensor(ex["label"],         dtype=torch.long),
        }
