import torch
import torch.optim as optim
from optimizer.adamw8bit import AdamW8bit as GaLoreAdamW8bit


def get_optimizer(model, cfg):
    # ============================
    # 3.1 SPLIT PARAMS FOR GaLore
    # ============================
    galore_params = []
    non_galore_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Apply GaLore to weight matrices (>=2D), keep biases/LayerNorm in normal group
        if p.ndim == 2:
            galore_params.append(p)
        else:
            non_galore_params.append(p)

    # ============================
    # 4. OPTIMIZER & SCHEDULER
    # ============================
    optimizer, optimizer_dict = None, None
    opt_name = cfg.OPTIMIZER_NAME
    print(f"Using optimizer: {opt_name}")

    if opt_name == "adamw":
        # 1) Plain AdamW on all parameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )

    elif opt_name == "galore8":
        # 2) GaLoreAdamW8bit (one optimizer with param groups)
        print("Using GaLoreAdamW8bit (param groups)")
        param_groups = [
            {
                "params": non_galore_params,
                "weight_decay": cfg.WEIGHT_DECAY,
            },
            {
                "params": galore_params,
                "rank": cfg.GLORA_RANK,
                "update_proj_gap": cfg.GLORA_UPDATE_PROJ_GAP,
                "scale": cfg.GLORA_SCALE,
                "proj_type": cfg.GLORA_PROJ_TYPE,
                "weight_decay": cfg.WEIGHT_DECAY,
            },
        ]
        optimizer = GaLoreAdamW8bit(
            param_groups,
            lr=cfg.LEARNING_RATE,
        )

    elif opt_name == "galore8_per_layer":
        # 3) GaLoreAdamW8bit per parameter with hooks
        print("Using GaLoreAdamW8bit *per layer*")

        # AdamW for non-GaLore params
        optimizer = optim.AdamW(
            non_galore_params,
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        # One GaLoreAdamW8bit per big weight param
        optimizer_dict = {}
        for p in galore_params:
            optimizer_dict[p] = GaLoreAdamW8bit(
                [
                    {
                        "params": galore_params,
                        "rank": cfg.GLORA_RANK,
                        "update_proj_gap": cfg.GLORA_UPDATE_PROJ_GAP,
                        "scale": cfg.GLORA_SCALE,
                        "proj_type": cfg.GLORA_PROJ_TYPE,
                        "weight_decay": cfg.WEIGHT_DECAY,
                    }
                ],
                lr=cfg.LEARNING_RATE,
            )

        # Hook to update that param after grad accumulation
        def optimizer_hook(param: torch.nn.Parameter):
            if param.grad is None:
                return
            opt = optimizer_dict[param]
            opt.step()
            opt.zero_grad()

        # Register hook for each GaLore param
        for p in galore_params:
            p.register_post_accumulate_grad_hook(optimizer_hook)

    else:
        raise ValueError(f"Unknown optimizer_name: {opt_name}")

    return optimizer, optimizer_dict
