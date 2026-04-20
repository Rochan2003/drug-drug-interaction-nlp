import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

from shared.losses import FocalLoss, compute_class_weights


def train_model(model, train_dataset, val_dataset, config):
    device = config["device"]
    model  = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW([
        {"params": list(model.bert.parameters()),       "lr": config["bert_lr"]},
        {"params": list(model.classifier.parameters()) +
                   list(model.dropout.parameters()),    "lr": config["head_lr"]},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * config["epochs"]
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    criterion   = FocalLoss(alpha=compute_class_weights(train_dataset.examples).to(device), gamma=2.0)
    best_val_f1 = 0.0
    best_state  = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [train]"):
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["e1_pos"].to(device),
                batch["e2_pos"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        val_f1 = _val_f1(model, val_loader, device)
        print(f"Epoch {epoch:2d} | loss={total_loss/len(train_loader):.4f} | val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"         ✓ New best (val_f1={best_val_f1:.4f})")

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
    print(f"\nTraining complete. Best val macro-F1: {best_val_f1:.4f}")
    return model


def _val_f1(model, loader, device):
    model.eval()
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
    return f1_score(all_labels, all_preds, average="macro", zero_division=0)
