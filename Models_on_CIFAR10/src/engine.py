from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from metrics import multiclass_metrics, binary_metrics, regression_metrics

@torch.no_grad()
def evaluate(model, loader, device, criterion, cfg, mase_den: float = None):
    """
    Returns: (avg_loss, metrics_dict)
    Chooses metrics based on cfg['task'].
    """
    task = cfg["task"]["type"].lower()
    num_classes = int(cfg["task"].get("num_classes", 1))
    want = tuple(cfg["task"].get("metrics", []))

    model.eval()
    total, loss_sum = 0, 0.0
    all_logits = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=False):
            logits = model(x)
            loss = criterion(logits.view(y.shape[0], -1) if task in ("binary","regression") else logits, y)

        loss_sum += loss.item() * y.size(0)
        total += y.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    avg_loss = loss_sum / max(total, 1)
    logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0)
    targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, dtype=torch.long)

    if task == "classification":
        mets = multiclass_metrics(logits, targets, num_classes, want_topk=(1,))
    elif task == "binary":
        mets = binary_metrics(logits, targets, threshold=float(cfg["task"].get("threshold", 0.5)))
    elif task == "regression":
        preds = logits.view_as(targets.to(logits.dtype))
        mets = regression_metrics(preds, targets.to(torch.float32),
                                  include=tuple(want) if want else ("mse","mae","rmse","r2","mape"),
                                  mase_den=mase_den, mase_m=cfg["task"].get("mase_m"))
    else:
        raise ValueError(f"Unknown task {task}")

    return avg_loss, mets

def train_one_epoch(model, loader, device, criterion, optimizer, scaler, clip_norm, amp, log_interval, batch_scheduler=None):
    model.train()
    running, total = 0.0, 0
    for i,(x,y) in enumerate(loader, start=1):
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        if amp:
            scaler.scale(loss).backward()
            if clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()
        bs = y.size(0)
        running += loss.item() * bs
        total += bs
        if log_interval and (i % log_interval == 0):
            yield {"train_loss_step": running / total, "seen": total}
    yield {"train_loss_epoch": running / total}
