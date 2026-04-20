import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, f1_score,
    precision_score, recall_score, confusion_matrix,
)

from shared.preprocessing import ID2LABEL, NUM_LABELS

POSITIVE_IDS = [1, 2, 3, 4]   # mechanism, effect, advise, int — excludes negative


def evaluate(model, test_dataset, config):
    """
    Run the trained model on the test set and print the full report.
    Official metric: macro-F1 over positive classes only (SemEval-2013 convention).
    """
    device = config["device"]
    model  = model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["e1_pos"].to(device),
                batch["e2_pos"].to(device),
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    print("\n" + "=" * 65)
    print("FULL CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    macro_f1 = f1_score(     all_labels, all_preds, labels=POSITIVE_IDS, average="macro", zero_division=0)
    macro_p  = precision_score(all_labels, all_preds, labels=POSITIVE_IDS, average="macro", zero_division=0)
    macro_r  = recall_score(  all_labels, all_preds, labels=POSITIVE_IDS, average="macro", zero_division=0)

    print("=" * 65)
    print("OFFICIAL DDI METRIC  (macro over positive classes only)")
    print("=" * 65)
    print(f"  Macro Precision : {macro_p:.4f}")
    print(f"  Macro Recall    : {macro_r:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}  ← headline number")

    print("\n" + "=" * 65)
    print("CONFUSION MATRIX  (rows = true label, cols = predicted)")
    print("=" * 65)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_LABELS)))
    print("        " + "  ".join(f"{n[:6]:>6}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"{label_names[i][:8]:8s}" + "  ".join(f"{v:6d}" for v in row))

    print("\n" + "=" * 65)
    print("PER-CLASS BREAKDOWN  (positive classes)")
    print("=" * 65)
    pcf = f1_score(        all_labels, all_preds, average=None, zero_division=0)
    pcp = precision_score( all_labels, all_preds, average=None, zero_division=0)
    pcr = recall_score(    all_labels, all_preds, average=None, zero_division=0)
    print(f"  {'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-' * 50}")
    for i in POSITIVE_IDS:
        print(f"  {ID2LABEL[i]:<18} {pcp[i]:>10.4f} {pcr[i]:>10.4f} {pcf[i]:>10.4f}")

    return {"macro_f1": macro_f1, "precision": macro_p, "recall": macro_r}
