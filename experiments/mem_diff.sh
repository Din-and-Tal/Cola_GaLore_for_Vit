#!/bin/bash

config_names=(
    "vit_adamw"
    "vit_galore_layer"
    "cola_adamw"
    "cola_galore_layer"
)


for config in "${config_names[@]}"; do
    echo "Running with config: $config"
    python main/main.py --config-name $config full_train=false use_checkpointing=false wandb_project_name=cola_galore_mem_diff
    python main/main.py --config-name $config full_train=false use_checkpointing=true wandb_project_name=cola_galore_mem_diff
    python main/main.py --config-name $config full_train=false size=huge batch_size=32 use_checkpointing=false wandb_project_name=cola_galore_mem_diff
    python main/main.py --config-name $config full_train=false size=huge batch_size=32 use_checkpointing=true wandb_project_name=cola_galore_mem_diff
done
