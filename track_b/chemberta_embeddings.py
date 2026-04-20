import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

CACHE_PATH   = Path(__file__).parent / "drug_embeddings_cache.pkl"
CHEMBERTA_ID = "seyonec/ChemBERTa-zinc-base-v1"
EMB_DIM      = 768


def build_drug_embeddings(drug_smiles: dict, device="cpu") -> dict:
    """
    Generates a 768-dim ChemBERTa embedding for every drug in drug_smiles.

    Drug nodes with a known SMILES string:
      SMILES → ChemBERTa tokenizer → ChemBERTa encoder → [CLS] token → 768-dim vector
    Drug nodes without a SMILES (not found in PubChem):
      Zero vector — the GAT learns to handle unknown drugs gracefully.

    Returns {drug_name_lower: np.array(768,)}
    Caches to drug_embeddings_cache.pkl so ChemBERTa only runs once.
    """
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            embeddings = pickle.load(f)
        print(f"  Loaded embedding cache  ({len(embeddings)} drugs)")
        return embeddings

    print(f"  Loading ChemBERTa ({CHEMBERTA_ID}) ...")
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_ID)
    model     = AutoModel.from_pretrained(CHEMBERTA_ID).to(device)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for drug_name, smiles in tqdm(drug_smiles.items(), desc="  ChemBERTa embeddings"):
            if not smiles:
                embeddings[drug_name] = np.zeros(EMB_DIM, dtype=np.float32)
                continue
            try:
                enc = tokenizer(
                    smiles,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding=True,
                ).to(device)
                out = model(**enc)
                emb = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                embeddings[drug_name] = emb.astype(np.float32)
            except Exception:
                embeddings[drug_name] = np.zeros(EMB_DIM, dtype=np.float32)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"  Saved → {CACHE_PATH}")
    return embeddings
