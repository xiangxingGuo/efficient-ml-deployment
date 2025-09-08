import torch.nn as nn
from torchvision.models import resnet18

class SmallCNN(nn.Module):
    def __init__(self, out_dim=10, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): return self.classifier(self.features(x))

def _out_dim_from_task(task_cfg):
    t = task_cfg["type"].lower()
    if t == "classification":
        return int(task_cfg["num_classes"])
    elif t in ("binary", "regression"):
        return 1
    else:
        raise ValueError(f"Unknown task.type {t}")

def build_model(cfg):
    name = cfg["model"]["name"].lower()
    out_dim = _out_dim_from_task(cfg["task"])
    if name == "resnet18":
        m = resnet18(weights=None, num_classes=out_dim)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    elif name == "smallcnn":
        return SmallCNN(out_dim=out_dim, dropout=cfg["model"].get("dropout", 0.0))
    else:
        raise ValueError(f"Unknown model {name}")
