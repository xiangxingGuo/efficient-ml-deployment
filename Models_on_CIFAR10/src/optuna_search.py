# pip install optuna
import os, yaml, optuna
from train import train_once, apply_overrides

def objective(trial, base_cfg):
    cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy
    # Suggest hyperparams
    cfg["optim"]["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cfg["optim"]["weight_decay"] = trial.suggest_float("wd", 0.0, 0.1, log=False)
    cfg["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
    cfg["scheduler"]["name"] = trial.suggest_categorical("sched", ["cosine", "onecycle"])
    cfg["train"]["save_dir"] = os.path.join(base_cfg["train"]["save_dir"], f"optuna/trial_{trial.number}")
    res = train_once(cfg)
    return res["val_loss"]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/example.yaml")
    p.add_argument("--trials", type=int, default=20)
    args = p.parse_args()
    base_cfg = yaml.safe_load(open(args.config))
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, base_cfg), n_trials=args.trials)
    print("Best:", study.best_trial.params)
