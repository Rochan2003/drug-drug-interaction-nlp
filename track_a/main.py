import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from collections import Counter

import torch
from torch.utils.data import random_split

import torch.nn as nn
from transformers import AutoModel

from shared.preprocessing import load_xml_files, ID2LABEL, NUM_LABELS, TRAIN_DIR, TEST_DIR
from track_a.dataset      import DDIDataset, get_tokenizer
from track_a.train        import train_model
from shared.evaluate      import evaluate


class DDIClassifier(nn.Module):
    """
    BioBERT encoder with entity-span classification head.
    Extracts hidden states at [E1] and [E2] positions, concatenates → Linear(5).
    freeze_layers=10 keeps only top 2 BERT layers + head trainable (~14M params).
    """
    def __init__(self, model_name="dmis-lab/biobert-v1.1", num_labels=NUM_LABELS,
                 dropout=0.1, vocab_size=None, freeze_layers=10):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
        hidden_size     = self.bert.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, e1_pos, e2_pos):
        hidden     = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch_size = hidden.size(0)
        e1_hidden  = hidden[torch.arange(batch_size), e1_pos, :]
        e2_hidden  = hidden[torch.arange(batch_size), e2_pos, :]
        return self.classifier(self.dropout(torch.cat([e1_hidden, e2_hidden], dim=-1)))

CONFIG = {
    "model_name":      "dmis-lab/biobert-v1.1",
    "epochs":          3,
    "batch_size":      32,
    "bert_lr":         2e-5,
    "head_lr":         1e-4,
    "freeze_layers":   10,     # 0 = full fine-tune (GPU recommended)
    "max_length":      128,
    "val_frac":        0.1,
    "demo_mode":       False,  # True = ~25 min sanity check; False = full run
    "demo_train_size": 1500,
    "demo_test_size":  300,
    "device": (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    ),
    "seed": 42,
}


def main():
    torch.manual_seed(CONFIG["seed"])
    print(f"\nDevice : {CONFIG['device']}")
    print(f"Model  : {CONFIG['model_name']}\n")

    if CONFIG["demo_mode"]:
        print("*** DEMO MODE — set demo_mode=False for full training ***\n")

    # ── Load corpus ───────────────────────────────────────────────────────────
    print("[1/5] Loading corpus ...")
    train_examples = load_xml_files(TRAIN_DIR)
    test_examples  = load_xml_files(TEST_DIR)

    if CONFIG["demo_mode"]:
        rng = random.Random(CONFIG["seed"])
        rng.shuffle(train_examples)
        rng.shuffle(test_examples)
        train_examples = train_examples[:CONFIG["demo_train_size"]]
        test_examples  = test_examples[:CONFIG["demo_test_size"]]

    print(f"       Train: {len(train_examples)}  |  Test: {len(test_examples)}")
    dist = Counter(ex["label"] for ex in train_examples)
    for lid, count in sorted(dist.items()):
        print(f"         {ID2LABEL[lid]:12s}: {count:6d}  ({100 * count / len(train_examples):.1f}%)")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\n[2/5] Loading tokenizer ...")
    tokenizer = get_tokenizer(CONFIG["model_name"])

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("[3/5] Building datasets ...")
    full_train = DDIDataset(train_examples, tokenizer, CONFIG["max_length"])
    test_ds    = DDIDataset(test_examples,  tokenizer, CONFIG["max_length"])

    val_size   = int(len(full_train) * CONFIG["val_frac"])
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    train_ds.examples = [full_train.examples[i] for i in train_ds.indices]
    val_ds.examples   = [full_train.examples[i] for i in val_ds.indices]
    print(f"       Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[4/5] Building model ...")
    model = DDIClassifier(
        model_name    = CONFIG["model_name"],
        vocab_size    = len(tokenizer),
        freeze_layers = CONFIG["freeze_layers"],
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Total params    : {total:,}")
    print(f"       Trainable params: {trainable:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training for {CONFIG['epochs']} epochs ...")
    model = train_model(model, train_ds, val_ds, CONFIG)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[6/6] Evaluating on test set ...")
    results = evaluate(model, test_ds, CONFIG)
    print(f"\nFinal macro-F1 (positive classes): {results['macro_f1']:.4f}")

    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "track_a_biobert_best.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
