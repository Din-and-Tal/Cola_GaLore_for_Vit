#!/bin/bash

config_names=(
    # "vit_adamw"
    # "vit_glora"
    # "vit_glora_layer"

    "v2_cola_adamw"
    "v2_cola_glora"
    "v2_cola_glora_layer"

)

for config in "${config_names[@]}"; do
    echo "Running with config: $config"
    python main/main.py --config-name $config wandb_project_name=cola_tuning patch_size=4
    python main/main.py --config-name $config wandb_project_name=cola_tuning hidden_dropout_prob=0.0
    python main/main.py --config-name $config wandb_project_name=cola_tuning mixup_alpha=0.8 cutmix_alpha=1.0 mix_prob=0.8
    python main/main.py --config-name $config wandb_project_name=cola_tuning aug_crop_min_scale=0.2 aug_rand_magnitude=7 aug_erase_prob=0.25
    python main/main.py --config-name $config wandb_project_name=cola_tuning patch_size=4 hidden_dropout_prob=0.0 mixup_alpha=0.8 cutmix_alpha=1.0 mix_prob=0.8 aug_crop_min_scale=0.2 aug_rand_magnitude=7 aug_erase_prob=0.25
done
