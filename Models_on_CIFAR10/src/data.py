import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random

# CIFAR-10 stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def _cutout(img, size):
    # img: Tensor CxHxW
    if size <= 0: return img
    c, h, w = img.shape
    y = random.randint(0, h - 1)
    x = random.randint(0, w - 1)
    y1 = max(0, y - size // 2); y2 = min(h, y + size // 2)
    x1 = max(0, x - size // 2); x2 = min(w, x + size // 2)
    img[:, y1:y2, x1:x2] = 0.0
    return img

class Cutout(object):
    def __init__(self, size): self.size = size
    def __call__(self, img): return _cutout(img, self.size)

def get_cifar10_transforms(aug: str, cutout_px: int):
    """Aug choices:
       - none : only normalize
       - basic: RandomCrop(32,4) + RandomHorizontalFlip
       - auto : torchvision AutoAugment(CIFAR10) + basic
    """
    train_trans = []
    if aug in ("basic", "auto"):
        train_trans += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    if aug == "auto":
        # AutoAugment for CIFAR10 (light but helpful)
        train_trans.insert(0, transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))

    train_trans += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    if cutout_px and cutout_px > 0:
        train_trans += [Cutout(cutout_px)]

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transforms.Compose(train_trans), test_trans

def get_loaders(cfg):
    name = cfg["data"]["name"].lower()
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    root = cfg["data"]["root"]

    if name == "cifar10":
        aug = cfg["data"].get("aug", "basic")
        cutout_px = cfg["data"].get("cutout", 0)
        train_tf, test_tf = get_cifar10_transforms(aug, cutout_px)
        train = datasets.CIFAR10(root, train=True,  transform=train_tf,  download=True)
        test  = datasets.CIFAR10(root, train=False, transform=test_tf,   download=True)
        # 45k / 5k split for val
        train_set, val_set = torch.utils.data.random_split(
            train, [45000, 5000], generator=torch.Generator().manual_seed(123)
        )
    elif name == "mnist":
        # (existing MNIST block from earlier, keep if you want both)
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(root, train=True, transform=tfm, download=True)
        test  = datasets.MNIST(root, train=False, transform=tfm, download=True)
        train_set, val_set = torch.utils.data.random_split(train, [55000, 5000], generator=torch.Generator().manual_seed(123))
    else:
        raise ValueError(f"Unknown dataset {name}")

    shuffle_train = bool(cfg["data"].get("shuffle_train", True))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test,      batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
