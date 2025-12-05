import os
import torch
import random
import numpy as np



import torch
import numpy as np


def rand_bbox(size, lam):
    """CutMix bounding box."""
    W = size[-1]
    H = size[-2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def apply_mixup_cutmix(inputs, targets, cfg, device):
    """
    Returns:
        mixed_inputs, targets_a, targets_b, lam, used
    """
    use_mix = cfg.use_mixup or cfg.use_cutmix
    if (not use_mix) or (torch.rand(1).item() > cfg.mix_prob):
        return inputs, targets, targets, 1.0, False

    batch_size = inputs.size(0)
    indices = torch.randperm(batch_size, device=device)

    targets_a = targets
    targets_b = targets[indices]

    mixed_inputs = inputs
    lam = 1.0

    # sample lambda from Beta
    if cfg.use_mixup and cfg.use_cutmix:
        # randomly choose one of them
        choose_mixup = torch.rand(1).item() < 0.5
    else:
        choose_mixup = cfg.use_mixup

    if choose_mixup and cfg.mixup_alpha > 0:
        lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
        mixed_inputs = lam * inputs + (1.0 - lam) * inputs[indices]
    elif cfg.use_cutmix and cfg.cutmix_alpha > 0:
        lam = np.random.beta(cfg.cutmix_alpha, cfg.cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, y1:y2, x1:x2] = inputs[indices, :, y1:y2, x1:x2]
        # adjust lam based on the exact area
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (inputs.size(-1) * inputs.size(-2)))

    return mixed_inputs, targets_a, targets_b, lam, True



def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  # will increase library footprint in GPU memory by approximately 24MiB
    )

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate_trainer_initialization(trainer):
    """
    Validate that all required Trainer attributes are initialized.
    Raises ValueError if anything is missing.
    """
    none_attrs = []

    # Attributes that must NOT be None
    for attr_name in [
        "cfg",
        "model",
        "optimizer",
        "loss_func",
        "scheduler",
    ]:
        if getattr(trainer, attr_name) is None:
            none_attrs.append(attr_name)

    # Check loaders
    if trainer.loaders.train is None:
        none_attrs.append("loaders.train")
    if trainer.loaders.val is None:
        none_attrs.append("loaders.val")
    if trainer.loaders.test is None:
        none_attrs.append("loaders.test")

    # Check wandb
    if trainer.cfg.use_wandb and getattr(trainer, "wandb") is None:
        none_attrs.append("wandb")

    # Throw if any missing
    if none_attrs:
        raise ValueError(
            "The following Trainer attributes were not initialized: "
            + ", ".join(none_attrs)
        )
