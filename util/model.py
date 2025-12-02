import torch
import os

from transformers import ViTConfig, ViTForImageClassification

from model.v1_cola_model import ColaViTConfig, ColaViTForImageClassification
from model.v2_cola_model import convert_vit_to_cola_m

# TODO: refactor


def build_model(cfg):

    vit_config = ViTConfig(
        image_size=cfg.IMAGE_SIZE,
        patch_size=cfg.PATCH_SIZE,
        num_channels=cfg.NUM_CHANNELS,
        num_labels=cfg.NUM_CLASSES,
        hidden_size=cfg.HIDDEN_SIZE,
        num_hidden_layers=cfg.NUM_HIDDEN_LAYERS,
        num_attention_heads=cfg.NUM_ATTENTION_HEADS,
        intermediate_size=cfg.INTERMEDIATE_SIZE,
        hidden_dropout_prob=cfg.HIDDEN_DROPOUT_PROB,
    )

    if cfg.MODEL_NAME == "vit":
        vit = ViTForImageClassification(vit_config)
        return vit
    elif cfg.MODEL_NAME == "v1_cola":
        cola = ColaViTForImageClassification(
            config=ColaViTConfig(
                rank=cfg.COLA_RANK,
                lr_act_type=cfg.COLA_LR_ACT_TYPE,
                **vit_config.to_dict(),  # Inherit all standard args
            )
        )
        return cola
    elif cfg.MODEL_NAME == "v2_cola":
        v2_cola = ViTForImageClassification(vit_config)
        v2_cola = convert_vit_to_cola_m(v2_cola, rank_ratio=0.25)
        return v2_cola

    raise ValueError(f"bad model name, ")


# TODO: change save/load model
def save_model(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    train_loss,
    val_acc,
    best_acc,
    cfg,
    optimizer_dict=None,
):
    # Ensure output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    state_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": train_loss,
        "best_acc": best_acc,
        "use_cola": cfg.USE_COLA,
        "optimizer_name": cfg.OPTIMIZER_NAME,
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
        save_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
        torch.save(state_dict, save_path)
        print(f"--> New best model saved! ({best_acc:.2f}%) at {save_path}")

    return best_acc


def load_model(
    model, optimizer, scheduler, scaler, modelPath, device, optimizerParams=None
):
    if not os.path.exists(modelPath):
        print(f"!! Model not found at {modelPath}, starting from scratch.")
        return 0

    print(f"Loading model from {modelPath}...")
    loaded_model = torch.load(modelPath, map_location=device, weights_only=False)

    # Check if it's a full model (dict with metadata) or just weights
    if isinstance(loaded_model, dict) and "model_state_dict" in loaded_model:
        model.load_state_dict(loaded_model["model_state_dict"])

        # Load main optimizer
        if optimizer is not None and "optimizer_state_dict" in loaded_model:
            optimizer.load_state_dict(loaded_model["optimizer_state_dict"])

        # Load per-layer optimizer dict if present
        if optimizerParams is not None and "optimizer_dict_state" in loaded_model:
            optimizer_dict_state = loaded_model["optimizer_dict_state"]
            optimizer_dict_keys = loaded_model["optimizer_dict_keys"]

            # Match parameters by their id
            param_id_map = {id(p): p for p in optimizerParams.keys()}
            for saved_id, saved_state in zip(
                optimizer_dict_keys, optimizer_dict_state.values()
            ):
                if saved_id in param_id_map:
                    optimizerParams[param_id_map[saved_id]].load_state_dict(saved_state)

        if scheduler is not None and "scheduler_state_dict" in loaded_model:
            scheduler.load_state_dict(loaded_model["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in loaded_model:
            scaler.load_state_dict(loaded_model["scaler_state_dict"])

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
