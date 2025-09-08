import argparse, os, yaml, json, time
import torch
import torch.nn as nn

from data import get_loaders
from models import build_model
from optim_sched import build_optimizer, build_scheduler
from engine import train_one_epoch, evaluate
from utils import set_seed, EarlyStopper, prepare_save_dir, now_str
from torch.cuda.amp import GradScaler
from metrics import compute_mase_denominator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/cifar10.yaml")
    p.add_argument("--override", type=str, default=None, help='JSON string to override config, e.g. \'{"optim.lr":1e-3}\'')
    p.add_argument("--search", action="store_true", help="Run grid search (see search space in code)")
    return p.parse_args()

def apply_overrides(cfg, override_str):
    if not override_str: return cfg
    overrides = json.loads(override_str)
    for k,v in overrides.items():
        keys = k.split(".")
        d = cfg
        for kk in keys[:-1]:
            d = d[kk]
        d[keys[-1]] = v
    return cfg

def maybe_tb(logger_cfg, save_dir):
    if logger_cfg.get("use_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
        except Exception as e:
            print(f"[warn] TensorBoard disabled: {e}")
    return None

def _build_criterion(cfg):
    task = cfg["task"]["type"].lower()
    if task == "classification":
        ls = float(cfg["train"].get("label_smoothing", 0.0))
        return nn.CrossEntropyLoss(label_smoothing=ls)
    elif task == "binary":
        # optional: pos_weight for imbalance -> cfg["task"].get("pos_weight")
        pw = cfg["task"].get("pos_weight", None)
        pos_weight = torch.tensor([float(pw)]) if pw is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task == "regression":
        # choose loss via cfg["task"].get("loss", "mse")
        loss_name = str(cfg["task"].get("loss", "mse")).lower()
        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name in ("mae","l1"):
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.SmoothL1Loss(beta=float(cfg["task"].get("huber_beta", 1.0)))
        else:
            raise ValueError(f"Unknown regression loss {loss_name}")
    else:
        raise ValueError(f"Unknown task.type {task}")

def train_once(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"], cfg["deterministic"])
    save_dir = prepare_save_dir(cfg["train"]["save_dir"])
    writer = maybe_tb(cfg["logging"], save_dir)

    # ----- data
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # ----- model + loss + optim
    model = build_model(cfg).to(device)
    criterion = _build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    steps_per_epoch = len(train_loader)
    scheduler, sched_step, batch_mode = build_scheduler(cfg, optimizer, steps_per_epoch)
    scaler = GradScaler(enabled=cfg["train"]["amp"])
    early = EarlyStopper(patience=cfg["train"]["early_stopping_patience"], mode="min")

    # ----- optional MASE denominator (time-series regression only)
    mase_den = None
    if cfg["task"]["type"].lower() == "regression" and "mase" in set(cfg["task"].get("metrics", [])):
        m = int(cfg["task"].get("mase_m", 0))
        if m > 0:
            # gather ordered targets from the training set
            # IMPORTANT: requires data.shuffle_train == false in YAML for time-series
            import itertools
            y_list = []
            for _, yb in itertools.islice(iter(train_loader), 9999999):  # don't exhaustively copy huge data in practice
                y_list.append(yb.view(-1).cpu())
            if y_list:
                y_series = torch.cat(y_list, dim=0)
                mase_den = compute_mase_denominator(y_series, m)
            else:
                mase_den = None

    best_val, best_path = float("inf"), os.path.join(save_dir, "best.pt")
    last_path = os.path.join(save_dir, "last.pt")

    for epoch in range(1, cfg["train"]["epochs"]+1):
        # ----- train epoch (same as before)
        last_log = None
        for log in train_one_epoch(
            model, train_loader, device, criterion, optimizer, scaler,
            clip_norm=cfg["train"]["grad_clip_norm"], amp=cfg["train"]["amp"],
            log_interval=cfg["train"]["log_interval"],
            batch_scheduler=scheduler if batch_mode else None
        ):
            last_log = log
            if writer and "train_loss_step" in log:
                step = (epoch-1)*steps_per_epoch + log["seen"]//train_loader.batch_size
                writer.add_scalar("train/loss_step", log["train_loss_step"], step)
        train_loss = last_log["train_loss_epoch"] if last_log else float("nan")
        if writer: writer.add_scalar("train/loss_epoch", train_loss, epoch)

        # ----- evaluate (task-aware)
        val_loss, val_metrics = evaluate(model, val_loader, device, criterion, cfg, mase_den=mase_den)
        if writer:
            writer.add_scalar("val/loss", val_loss, epoch)
            for k,v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        if scheduler and not batch_mode:
            sched_step(val_loss)

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, last_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)

        # pretty print: only show the metrics the user asked for
        show = " | ".join(f"{k} {val_metrics[k]:.4f}" for k in cfg["task"].get("metrics", []) if k in val_metrics)
        print(f"Epoch {epoch}/{cfg['train']['epochs']} | train {train_loss:.4f} | val {val_loss:.4f} | {show} | lr {optimizer.param_groups[0]['lr']:.5g}")

        if early.step(val_loss):
            pass
        if early.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # test with best weights
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state["model"])
    test_loss, test_metrics = evaluate(model, test_loader, device, criterion, cfg, mase_den=mase_den)
    print("[TEST] " + " | ".join([f"loss {test_loss:.4f}"] + [f"{k} {test_metrics[k]:.4f}" for k in cfg["task"].get("metrics", []) if k in test_metrics]))
    if writer:
        writer.close()
    return {"val_loss": best_val, "test_loss": test_loss, **test_metrics}


def grid_search(cfg):
    """
    Pure-Python grid search over a small space.
    Edit 'search_space' as needed.
    """
    from itertools import product
    search_space = {
        "optim.lr": [1e-3, 3e-3, 1e-2],
        "optim.weight_decay": [0.0, 0.01],
        "scheduler.name": ["cosine", "onecycle"],
        "model.hidden_dim": [128, 256, 512],
    }
    keys, values = zip(*search_space.items())
    results = []
    start = time.time()
    for combo in product(*values):
        local = yaml.safe_load(yaml.dump(cfg))  # deep copy
        for k,v in zip(keys, combo):
            # nested override
            d = local
            parts = k.split(".")
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = v
        tag = "_".join(f"{k.split('.')[-1]}={v}" for k,v in zip(keys, combo))
        local["train"]["save_dir"] = os.path.join(cfg["train"]["save_dir"], "search", tag)
        print(f"\n=== Running {tag} ===")
        res = train_once(local)
        results.append((tag, res))
    elapsed = time.time() - start
    print(f"\nGrid search done in {elapsed/60:.1f} min. Top results:")
    results.sort(key=lambda x: (x[1]["val_loss"]))
    for tag, res in results[:5]:
        print(tag, "->", res)
    return results

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg = apply_overrides(cfg, args.override)
    if args.search:
        grid_search(cfg)
    else:
        train_once(cfg)

if __name__ == "__main__":
    main()