import os, random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EarlyStopper:
    def __init__(self, patience, mode="min"):
        self.patience = patience; self.mode = mode; self.best = None; self.num_bad = 0; self.should_stop=False
    def step(self, value):
        improve = (value < self.best) if self.best is not None else True
        if improve:
            self.best = value; self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience: self.should_stop = True
        return improve

def prepare_save_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
