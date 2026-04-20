import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from shared.preprocessing import LABEL2ID, ID2LABEL, NUM_LABELS


def mark_entities(text, e1_start, e1_end, e2_start, e2_end):
    """
    Insert [E1]/[/E1] and [E2]/[/E2] markers around the two drug spans.
    Insertions are applied right-to-left so earlier offsets stay valid.

    Example:
      "Aspirin may interact with Warfarin."
      → "[E1] Aspirin [/E1] may interact with [E2] Warfarin [/E2] ."
    """
    if e1_start < e2_start:
        insertions = [
            (e2_end + 1, " [/E2]"),
            (e2_start,   "[E2] "),
            (e1_end + 1, " [/E1]"),
            (e1_start,   "[E1] "),
        ]
    else:
        insertions = [
            (e1_end + 1, " [/E1]"),
            (e1_start,   "[E1] "),
            (e2_end + 1, " [/E2]"),
            (e2_start,   "[E2] "),
        ]
    for pos, marker in insertions:
        text = text[:pos] + marker + text[pos:]
    return text


def get_tokenizer(model_name="dmis-lab/biobert-v1.1"):
    """Load BioBERT tokenizer and register the four entity marker tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    )
    return tokenizer


class DDIDataset(Dataset):
    """
    Converts raw preprocessing examples into BioBERT-ready tensors.
    Entity marking is applied once at construction time.
    """

    def __init__(self, raw_examples, tokenizer, max_length=128):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = []
        for ex in raw_examples:
            marked = mark_entities(
                ex["text"], ex["e1_start"], ex["e1_end"],
                ex["e2_start"], ex["e2_end"],
            )
            self.examples.append({
                "text":    marked,
                "label":   ex["label"],
                "e1_text": ex["e1_text"],
                "e2_text": ex["e2_text"],
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        e1_pos = (input_ids == e1_token_id).nonzero(as_tuple=True)[0]
        e2_pos = (input_ids == e2_token_id).nonzero(as_tuple=True)[0]

        # Fall back to [CLS] if a marker was truncated away
        e1_pos = e1_pos[0].item() if len(e1_pos) > 0 else 0
        e2_pos = e2_pos[0].item() if len(e2_pos) > 0 else 0

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "e1_pos":         torch.tensor(e1_pos,      dtype=torch.long),
            "e2_pos":         torch.tensor(e2_pos,      dtype=torch.long),
            "label":          torch.tensor(ex["label"], dtype=torch.long),
        }
