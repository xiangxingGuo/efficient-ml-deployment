from typing import Dict, Iterable, Optional
import torch
import torch.nn.functional as F

EPS = 1e-12

# ---------- helpers

def _topk_accuracy(logits, target, ks=(1,)):
    maxk = max(ks)
    _, pred = logits.topk(maxk, 1, True, True)  # [B,maxk]
    pred = pred.t()                             # [maxk,B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = {}
    for k in ks:
        acc[f"top{k}"] = correct[:k].reshape(-1).float().mean().item()
    return acc

def _binary_counts(preds_bool, target_bool):
    tp = ((preds_bool == 1) & (target_bool == 1)).sum().item()
    tn = ((preds_bool == 0) & (target_bool == 0)).sum().item()
    fp = ((preds_bool == 1) & (target_bool == 0)).sum().item()
    fn = ((preds_bool == 0) & (target_bool == 1)).sum().item()
    return tp, tn, fp, fn

def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    # Prob. a random positive has higher score than a random negative (Mann-Whitney U)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()  # ascending
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    sum_pos_ranks = ranks[pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg + EPS)
    return float(auc)

# ---------- multi-class classification

def multiclass_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int, want_topk: Iterable[int]=(1,)):
    out = {}
    out.update(_topk_accuracy(logits, target, ks=tuple(sorted(set(want_topk)))))
    preds = logits.argmax(1)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=logits.device)
    for t, p in zip(target, preds):
        conf[t, p] += 1
    tp = conf.diag().to(torch.float32)
    per_true = conf.sum(1).to(torch.float32).clamp_min(1)
    per_pred = conf.sum(0).to(torch.float32).clamp_min(1)
    prec = (tp / per_pred)
    rec  = (tp / per_true)
    f1   = (2 * prec * rec) / (prec + rec).clamp_min(EPS)
    weights = per_true / per_true.sum().clamp_min(1)
    out["accuracy"] = out.get("top1", (preds == target).float().mean().item())
    out["precision_macro"] = prec.mean().item()
    out["recall_macro"]    = rec.mean().item()
    out["f1_macro"]        = f1.mean().item()
    out["f1_weighted"]     = (f1 * weights).sum().item()
    out["per_class_acc_mean"] = (tp / per_true).mean().item()
    return out

# ---------- binary classification

def binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits.view(-1))
    preds = (probs >= threshold).to(torch.int64)
    target = target.view(-1).to(torch.int64)
    tp, tn, fp, fn = _binary_counts(preds, target)
    total = max(int(target.numel()), 1)
    acc = (tp + tn) / total
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    auroc = _auroc(probs.detach().cpu(), target.detach().cpu())
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auroc": auroc
    }

# ---------- regression

def regression_metrics(preds: torch.Tensor, target: torch.Tensor,
                       include=("mse","mae","rmse","r2","mape"),
                       mase_den: Optional[float]=None, mase_m: Optional[int]=None):
    preds = preds.view_as(target).to(torch.float32)
    target = target.to(torch.float32)
    diff = preds - target
    mse  = (diff.pow(2)).mean().item()
    mae  = (diff.abs()).mean().item()
    rmse = mse ** 0.5
    # R2 (coefficient of determination)
    var = (target - target.mean()).pow(2).sum().item()
    ssr = (diff.pow(2)).sum().item()
    r2  = 1.0 - (ssr / (var + EPS))
    # MAPE (robust with EPS)
    mape = (diff.abs() / (target.abs() + EPS)).mean().item()
    out = {}
    if "mse"  in include: out["mse"]  = mse
    if "mae"  in include: out["mae"]  = mae
    if "rmse" in include: out["rmse"] = rmse
    if "r2"   in include: out["r2"]   = r2
    if "mape" in include: out["mape"] = mape
    if "mase" in include:
        if mase_den is None or mase_den <= 0 or mase_m is None or mase_m <= 0:
            out["mase"] = float("nan")
        else:
            out["mase"] = mae / (mase_den + EPS)
    return out

def compute_mase_denominator(train_targets: torch.Tensor, m: int) -> float:
    # train_targets: 1D ordered series (no shuffle)
    if m <= 0 or train_targets.numel() <= m:
        return float("nan")
    diffs = (train_targets[m:] - train_targets[:-m]).abs()
    return float(diffs.mean().item())
