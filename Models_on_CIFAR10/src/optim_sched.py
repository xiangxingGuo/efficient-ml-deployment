import math
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR

def build_optimizer(cfg, model):
    o = cfg["optim"]; name = o["name"].lower()
    if name == "sgd":
        opt = SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"], nesterov=True)
    elif name == "adam":
        opt = Adam(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    elif name == "adamw":
        opt = AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer {name}")
    return opt

def build_scheduler(cfg, optimizer, steps_per_epoch=None):
    s = cfg["scheduler"]; name = s["name"].lower()
    epochs = s["epochs"]
    warmup_epochs = s.get("warmup_epochs", 0)

    if name == "none":
        return None, lambda: None, False

    if name == "onecycle":
        assert steps_per_epoch is not None, "steps_per_epoch required for OneCycleLR"
        sched = OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=s.get("pct_start", 0.3),
            anneal_strategy="cos",
        )
        # OneCycle is stepped per-batch
        return sched, lambda metric=None: None, True

    # epoch-stepped schedulers (+ optional warmup)
    if name == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs if epochs>warmup_epochs else 1)
    elif name == "step":
        main = StepLR(optimizer, step_size=s["step_size"], gamma=s["gamma"])
    elif name == "reduce_on_plateau":
        main = ReduceLROnPlateau(optimizer, mode="min", factor=s["reduce_factor"], patience=s["reduce_patience"])
    else:
        raise ValueError(f"Unknown scheduler {name}")

    if warmup_epochs > 0:
        # Simple linear warmup wrapper
        class WarmupWrap:
            def __init__(self, opt, main, warmup_epochs):
                self.opt, self.main, self.we, self.epoch = opt, main, warmup_epochs, 0
                self.base_lrs = [g["lr"] for g in opt.param_groups]
            def step_warmup(self):
                self.epoch += 1
                ratio = min(self.epoch / self.we, 1.0)
                for i,g in enumerate(self.opt.param_groups):
                    g["lr"] = self.base_lrs[i] * ratio
            def step(self, metric=None):
                if self.epoch < self.we:
                    self.step_warmup()
                else:
                    if isinstance(self.main, ReduceLROnPlateau):
                        self.main.step(metric)
                    else:
                        self.main.step()
            def get_last_lr(self):
                return [g["lr"] for g in self.opt.param_groups]
        wrapper = WarmupWrap(optimizer, main, warmup_epochs)
        return wrapper, wrapper.step, False

    # No warmup, epoch step
    def step_fn(metric=None):
        if isinstance(main, ReduceLROnPlateau): main.step(metric)
        else: main.step()
    return main, step_fn, False
