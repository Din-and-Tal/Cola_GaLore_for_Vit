import os
import torch
import random
import numpy as np


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
