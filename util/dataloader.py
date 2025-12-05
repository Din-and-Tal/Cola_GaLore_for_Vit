import os
import zipfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from PIL import Image


# =========================
# CONSTANTS
# =========================
PROJECT_BASE = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_BASE / "datasets"
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_DIR = DATASET_DIR / "tiny-imagenet-200"


# =========================
# HELPERS
# =========================
def _ensure_tiny_imagenet() -> Path:
    """Ensure tiny-imagenet-200 exists under datasets/. Download if needed."""
    if TINY_IMAGENET_DIR.exists():
        return TINY_IMAGENET_DIR

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR / "tiny-imagenet-200.zip"

    print(f"[TinyImageNet] Downloading to {zip_path} ...")
    urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)

    print(f"[TinyImageNet] Unzipping ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATASET_DIR)

    zip_path.unlink(missing_ok=True)
    return TINY_IMAGENET_DIR


class TinyImageNetVal(Dataset):
    """Use tiny-imagenet-200/val as test set."""
    def __init__(self, root, class_to_idx, transform=None):
        val_root = os.path.join(root, "val")
        images_dir = os.path.join(val_root, "images")
        ann_path = os.path.join(val_root, "val_annotations.txt")

        self.samples = []
        self.transform = transform

        with open(ann_path, "r") as f:
            for line in f:
                fname, wnid, *_ = line.strip().split()
                if wnid not in class_to_idx:
                    continue
                label = class_to_idx[wnid]
                path = os.path.join(images_dir, fname)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


class TransformSubset(Dataset):
    """Subset wrapper that applies its own transform (so val never sees train_tf)."""
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, target = self.base_dataset[idx]  # base has transform=None -> PIL
        if self.transform:
            img = self.transform(img)
        return img, target


# =========================
# MAIN
# =========================
def get_data_loaders(cfg):
    """
    Tiny ImageNet only.
    Raises if cfg.dataset_name != 'tiny_imagenet'.
    Uses cfg.seed for reproducibility.
    """
    if getattr(cfg, "dataset_name", "").lower() != "tiny_imagenet":
        raise ValueError("get_data_loaders only supports dataset_name='tiny_imagenet'")

    root = _ensure_tiny_imagenet()

    image_size = getattr(cfg, "image_size", 64)
    seed = getattr(cfg, "seed", 42)
    pin_memory = getattr(cfg, "pin_memory", True)

    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2770, 0.2691, 0.2821)

    # ✅ Train: augmentation; Val/Test: deterministic only
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Base dataset (no transform yet)
    base_train = datasets.ImageFolder(os.path.join(root, "train"), transform=None)

    # 90/10 split using cfg.seed
    n_total = len(base_train)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_train, [n_train, n_val], generator=gen)

    # Wrap with transforms
    train_ds = TransformSubset(base_train, train_subset.indices, transform=train_tf)
    val_ds = TransformSubset(base_train, val_subset.indices, transform=eval_tf)

    # Test from val/ folder
    test_ds = TinyImageNetVal(root, class_to_idx=base_train.class_to_idx, transform=eval_tf)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=pin_memory)
    print(
            f"Samples -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}, Classes: {len(base_train.class_to_idx)}"
        )
    print(
            f"Batches -> Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
        )
    return train_loader, val_loader, test_loader
