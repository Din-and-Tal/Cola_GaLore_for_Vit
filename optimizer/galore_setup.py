import bitsandbytes as bnb
import torch.nn as nn
import torch
from model.cola_layer import ColaLinear
from optimizer.galore8bit import AdamW8bit as GaLoreAdamW8bit
from util.scheduler import CosineAnnealingWarmupRestarts


def get_optimizer_scheduler(model, cfg):
    optimizer, optimizer_dict = None,None

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if 'galore' in cfg.optimizer_name.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attention", "intermediate"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear) and not isinstance(module, ColaLinear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)

        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [
            {"params": regular_params},
            {
                "params": galore_params,
                "rank": cfg.galore_rank,
                "update_proj_gap": cfg.galore_update_proj_gap,
                "scale": cfg.galore_scale,
                "proj_type": cfg.galore_proj_type,
            },
        ]

    if cfg.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.scheduler_max_lr, weight_decay=cfg.weight_decay)

    elif cfg.optimizer_name == "adamw8":
        optimizer = bnb.optim.AdamW8bit(
            trainable_params, lr=cfg.scheduler_max_lr, weight_decay=cfg.weight_decay
        )
        
    elif cfg.optimizer_name == "galore":
        optimizer = GaLoreAdamW8bit(
            param_groups, lr=cfg.scheduler_max_lr, weight_decay=cfg.weight_decay
        )
        
    elif cfg.optimizer_name == "galore_layer":
            
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap

        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit(
                        [
                            {
                                "params": [p],
                                "rank": cfg.galore_rank,
                                "update_proj_gap": cfg.galore_update_proj_gap * 2,
                                "scale": cfg.galore_scale,
                                "proj_type": cfg.galore_proj_type,
                            }
                        ],
                        lr=cfg.scheduler_max_lr,
                        weight_decay=cfg.weight_decay,
                    )
                else:
                    optimizer_dict[p] = bnb.optim.AdamW8bit(
                        [p], lr=cfg.scheduler_max_lr, weight_decay=cfg.weight_decay
                    )

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                # galore scheduler use steps and adamw use epoches so conversion is needed
                scheduler_dict[p] = CosineAnnealingWarmupRestarts(
                    optimizer=optimizer_dict[p],
                    first_cycle_steps=cfg.total_steps * 2,
                    cycle_mult=cfg.scheduler_cycle_mult,
                    max_lr=cfg.scheduler_max_lr,
                    min_lr=cfg.scheduler_min_lr,
                    warmup_steps=int(
                        cfg.scheduler_warmup_pct
                        * cfg.total_steps
                        * 2  # TODO: move adamw to steps (instead of epochs)
                    ),
                    gamma=cfg.scheduler_gamma,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        print("Using optimizer: galore_layer")
        return None, optimizer, optimizer_dict
    
    else:
        raise ValueError('optimizer name error')
            
    scheduler = CosineAnnealingWarmupRestarts(
                optimizer=optimizer,
                first_cycle_steps=cfg.num_epochs,
                cycle_mult=cfg.scheduler_cycle_mult,
                max_lr=cfg.scheduler_max_lr,
                min_lr=cfg.scheduler_min_lr,
                warmup_steps=cfg.scheduler_warmup_steps,
                gamma=cfg.scheduler_gamma,
            )
    print(f'using optimizer: {cfg.optimizer_name}')
    return scheduler, optimizer, None    
