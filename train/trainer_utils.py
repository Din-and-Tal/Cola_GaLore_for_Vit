
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
        "lossFunc",
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
    if trainer.cfg.USE_WANDB and getattr(trainer, "wandb") is None:
        none_attrs.append("wandb")

    # Throw if any missing
    if none_attrs:
        raise ValueError(
            "The following Trainer attributes were not initialized: "
            + ", ".join(none_attrs)
        )
