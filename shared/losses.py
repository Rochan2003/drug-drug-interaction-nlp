from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(examples, num_labels=5, smooth=False):
    """
    Inverse-frequency weights so rare classes get proportionally higher loss.
    smooth=True applies sqrt to soften extreme ratios (recommended for GAT
    where raw inverse-frequency causes over-prediction of rare classes).
    """
    counts  = Counter(ex["label"] for ex in examples)
    total   = sum(counts.values())
    weights = [total / (num_labels * counts.get(i, 1)) for i in range(num_labels)]
    if smooth:
        weights = [w ** 0.5 for w in weights]
    return torch.tensor(weights, dtype=torch.float)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy negatives, focuses training on hard positives.
    gamma=2 is the standard value from the original paper.
    """

    def __init__(self, alpha, gamma=2.0, reduction="mean"):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = torch.exp(log_probs)
        log_pt    = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt        = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t   = self.alpha[targets]
        loss      = -alpha_t * (1 - pt) ** self.gamma * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
