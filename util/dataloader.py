import random
import numpy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset


# Mean and Std for normalization
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "imagefolder": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet standard
}


def seed_worker(worker_id):
    """Worker init function for reproducibility - must be at module level for pickling on Windows."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loaders(cfg):
    """
    Returns training, validation and test dataloaders based on config.
    Splits the combined dataset according to TRAIN_RATIO, VAL_RATIO, TEST_RATIO.
    If fullTrain is False, only a fraction of the data is used.
    """
    print(f"Loading dataset: {cfg.dataset_name}...")

    # Fraction of data to use when fullTrain is False

    dataset_name = cfg.dataset_name.lower()
    mean, std = STATS.get(dataset_name, STATS["imagefolder"])

    # Standard ViT transforms
    common_transforms = [
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    train_transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(), # Optional augmentation
            *common_transforms
        ]
    )

    val_transform = transforms.Compose([*common_transforms])

    # Helper to load datasets
    def load_datasets(transform, train_split=True):
        if dataset_name == "cifar10":
            return datasets.CIFAR10(
                root="./datasets", train=train_split, download=True, transform=transform
            )
        elif dataset_name == "cifar100":
            return datasets.CIFAR100(
                root="./datasets", train=train_split, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset name: {cfg.dataset_name}")

    # 1. Create two versions of the full dataset: one with Augmentation, one Clean
    # Load official train and test sets
    ds_train_aug = load_datasets(train_transform, train_split=True)
    ds_test_aug = load_datasets(train_transform, train_split=False)
    full_aug = ConcatDataset([ds_train_aug, ds_test_aug])

    ds_train_clean = load_datasets(val_transform, train_split=True)
    ds_test_clean = load_datasets(val_transform, train_split=False)
    full_clean = ConcatDataset([ds_train_clean, ds_test_clean])

    # 2. Calculate Split Sizes
    total_size = len(full_aug)
    train_size = int(cfg.train_ratio * total_size)
    val_size = int(cfg.val_ratio * total_size)

    # reproducibility on multi-threaded loader -----------------------------------------------------
    g = torch.Generator().manual_seed(cfg.seed)

    # 3. Generate Indices
    indices = torch.randperm(total_size, generator=g).tolist()

    # -----------------------------------------------------------------------------------------------

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    debug_labels_count = int(len(test_indices) * cfg.debug_data_scale)
    # Reduce data if fullTrain is False
    if not cfg.full_train:
        train_indices = train_indices[: max(1, debug_labels_count)]
        val_indices = val_indices[: max(1, debug_labels_count)]
        test_indices = test_indices[: max(1, debug_labels_count)]
        print(f"Using reduced data: {debug_labels_count} epoches")

    # 4. Create Subsets
    # Train gets Augmented dataset, Val/Test get Clean dataset
    train_dataset = Subset(full_aug, train_indices)
    val_dataset = Subset(full_clean, val_indices)
    test_dataset = Subset(full_clean, test_indices)

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,  # make so the memory wont spill into disk (non-paging)
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    print(
        f"Batches -> Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )
    return train_loader, val_loader, test_loader
