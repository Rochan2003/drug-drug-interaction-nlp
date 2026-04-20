import json
import re
import time
import requests
from pathlib import Path
from tqdm import tqdm

from shared.preprocessing import load_xml_files, TRAIN_DIR, TEST_DIR

CACHE_PATH  = Path(__file__).parent / "drug_smiles_cache.json"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"


def _fetch_smiles(drug_name, retries=3):
    """
    Query PubChem REST API for canonical SMILES with retry logic.
    Tries progressively cleaned variants of the drug name if the first fails.
    Returns None if all attempts fail.
    """
    # Build a list of name variants to try, from most to least specific
    variants = _name_variants(drug_name)

    for variant in variants:
        for attempt in range(retries):
            try:
                url  = PUBCHEM_URL.format(requests.utils.quote(variant))
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    props = resp.json()["PropertyTable"]["Properties"][0]
                    return (props.get("CanonicalSMILES") or
                            props.get("ConnectivitySMILES") or
                            props.get("IsomericSMILES"))
                if resp.status_code == 404:
                    break   # this variant genuinely not found, try next
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)   # exponential back-off on rate limit
            except requests.exceptions.Timeout:
                time.sleep(1)
            except Exception:
                break
    return None


def _name_variants(name):
    """
    Generate progressively simplified variants of a drug name to maximise
    PubChem hit rate. DDI corpus names include brand names, abbreviations,
    multi-word compounds, and parenthetical suffixes.

    Examples:
      "fluconazole (150 mg)" → ["fluconazole (150 mg)", "fluconazole"]
      "ace inhibitors"       → ["ace inhibitors", "ace inhibitor"]
      "St. John's Wort"      → ["St. John's Wort", "St Johns Wort", "St John Wort"]
    """
    variants = [name]

    # Strip parenthetical dosage/qualifier suffixes: "drug (150 mg)" → "drug"
    stripped = re.sub(r'\s*\(.*?\)', '', name).strip()
    if stripped and stripped != name:
        variants.append(stripped)

    # Strip trailing plural 's' for class names
    if name.endswith('s') and len(name) > 4:
        variants.append(name[:-1])

    # Remove punctuation (apostrophes, periods, hyphens)
    cleaned = re.sub(r"['\.\-]", ' ', variants[-1]).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if cleaned not in variants:
        variants.append(cleaned)

    # First word only (e.g. "warfarin sodium" → "warfarin")
    first_word = name.split()[0]
    if first_word not in variants and len(first_word) > 3:
        variants.append(first_word)

    return variants


def get_all_drug_smiles():
    """
    Returns {drug_name_lower: smiles_string_or_None} for every drug in the corpus.
    Fetches from PubChem on first call, loads from cache on subsequent calls.
    If cache exists but has 0 hits, deletes it and rebuilds automatically.
    """
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        found = sum(v is not None for v in cache.values())
        print(f"  Loaded SMILES cache  ({found}/{len(cache)} found)")
        if found > 0:
            return cache
        print("  Cache has 0 hits — deleting and rebuilding ...")
        CACHE_PATH.unlink()

    print("  Building SMILES cache from PubChem API ...")
    all_examples = load_xml_files(TRAIN_DIR) + load_xml_files(TEST_DIR)
    drug_names   = {ex["e1_text"].lower() for ex in all_examples} | \
                   {ex["e2_text"].lower() for ex in all_examples}
    print(f"  Unique drug names: {len(drug_names)}")

    cache = {}
    for name in tqdm(sorted(drug_names), desc="  PubChem lookup"):
        cache[name] = _fetch_smiles(name)
        time.sleep(0.25)   # stay within PubChem's 5 req/s rate limit

    found = sum(v is not None for v in cache.values())
    print(f"  SMILES found: {found}/{len(drug_names)}")

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  Saved → {CACHE_PATH}")
    return cache
