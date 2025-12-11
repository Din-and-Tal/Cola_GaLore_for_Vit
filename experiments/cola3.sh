#!/bin/bash

config_names=(
    # "vit_adamw"
    # "vit_glora"
    # "vit_glora_layer"

    "v2_cola_adamw"
    # "v2_cola_glora"
    # "v2_cola_glora_layer"

)

cola_rank_ratios=(0.1 0.25 0.4 0.5 0.75 0.9 1)
# tal: 9.33h
# din: 12.5h
for config in "${config_names[@]}"; do
    for ratio in "${cola_rank_ratios[@]}"; do
        echo "Running with config: $config, cola_rank_ratio: $ratio"
        python main/main.py --config-name $config cola_rank_ratio=$ratio wandb_project_name=cola_rank_exp scheduler_max_lr=0.00020545522334013431 weight_decay=0.037343324638226455 label_smoothing=0.09740441812870372 hidden_dropout_prob=0.0910218284106628 mixup_alpha=0.27067234802367746 cutmix_alpha=0.06735967157072321 mix_prob=0.5440847747941563
        python main/main.py --config-name $config cola_rank_ratio=$ratio intermediate_rank_scale=true wandb_project_name=cola_rank_exp scheduler_max_lr=0.00020545522334013431 weight_decay=0.037343324638226455 label_smoothing=0.09740441812870372 hidden_dropout_prob=0.0910218284106628 mixup_alpha=0.27067234802367746 cutmix_alpha=0.06735967157072321 mix_prob=0.5440847747941563
    done
done
