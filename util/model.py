import torch
import os

from transformers import ViTConfig, ViTForImageClassification

from model.v1_cola_model import ColaViTConfig, ColaViTForImageClassification
from model.v2_cola_model import convert_vit_to_cola_m

# TODO: refactor


def build_model(cfg):

    vit_config = ViTConfig(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_channels=cfg.num_channels,
        num_labels=cfg.num_classes,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
    )

    if cfg.model_name == "vit":
        vit = ViTForImageClassification(vit_config)
        return vit
    elif cfg.model_name == "v1_cola":
        cola = ColaViTForImageClassification(
            config=ColaViTConfig(
                rank=cfg.cola_rank,
                lr_act_type=cfg.cola_lr_act_type,
                **vit_config.to_dict(),  # Inherit all standard args
            )
        )
        return cola
    elif cfg.model_name == "v2_cola":
        v2_cola = ViTForImageClassification(vit_config)
        v2_cola = convert_vit_to_cola_m(v2_cola, rank_ratio=cfg.cola_rank_ratio)
        return v2_cola

    raise ValueError(f"bad model name, ")


# TODO: change save/load model
def save_model(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    val_acc,
    best_acc,
    cfg,
    optimizer_dict=None,
):
    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)

    state_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": train_loss,
        "best_acc": best_acc,
        "use_cola": cfg.use_cola,
        "optimizer_name": cfg.optimizer_name,
    }

    # Handle different optimizer types
    if optimizer_dict is not None:
        # Per-layer optimizer case
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
        state_dict["optimizer_dict_state"] = {
            id(p): opt.state_dict() for p, opt in optimizer_dict.items()
        }
        state_dict["optimizer_dict_keys"] = [id(p) for p in optimizer_dict.keys()]
    else:
        # Single optimizer case
        state_dict["optimizer_state_dict"] = optimizer.state_dict()

    if val_acc > best_acc:
        best_acc = val_acc
        save_path = os.path.join(cfg.output_dir, "best_model.pth")
        torch.save(state_dict, save_path)
        print(f"--> New best model saved! ({best_acc:.2f}%) at {save_path}")

    return best_acc


def load_model(model, optimizer, scheduler, model_path, device, optimizer_params=None):
    if not os.path.exists(model_path):
        print(f"!! Model not found at {model_path}, starting from scratch.")
        return 0

    print(f"Loading model from {model_path}...")
    loaded_model = torch.load(model_path, map_location=device, weights_only=False)

    # Check if it's a full model (dict with metadata) or just weights
    if isinstance(loaded_model, dict) and "model_state_dict" in loaded_model:
        model.load_state_dict(loaded_model["model_state_dict"])

        # Load main optimizer
        if optimizer is not None and "optimizer_state_dict" in loaded_model:
            optimizer.load_state_dict(loaded_model["optimizer_state_dict"])

        # Load per-layer optimizer dict if present
        if optimizer_params is not None and "optimizer_dict_state" in loaded_model:
            optimizer_dict_state = loaded_model["optimizer_dict_state"]
            optimizer_dict_keys = loaded_model["optimizer_dict_keys"]

            # Match parameters by their id
            param_id_map = {id(p): p for p in optimizer_params.keys()}
            for saved_id, saved_state in zip(
                optimizer_dict_keys, optimizer_dict_state.values()
            ):
                if saved_id in param_id_map:
                    optimizer_params[param_id_map[saved_id]].load_state_dict(
                        saved_state
                    )

        if scheduler is not None and "scheduler_state_dict" in loaded_model:
            scheduler.load_state_dict(loaded_model["scheduler_state_dict"])

        start_epoch = loaded_model.get("epoch", -1) + 1

        # Show what was loaded
        loaded_info = f"epoch {start_epoch - 1}"
        if "use_cola" in loaded_model:
            loaded_info += f", use_cola={loaded_model['use_cola']}"
        if "optimizer_name" in loaded_model:
            loaded_info += f", optimizer={loaded_model['optimizer_name']}"

        print(f"--> Loaded model from {loaded_info}.")
        return start_epoch
    else:
        # Assume it's just state_dict (e.g. old best_model.pth)
        model.load_state_dict(loaded_model)
        print(
            "--> Loaded model weights only. WARNING: Optimizer/Scheduler state not restored."
        )
        return 0
