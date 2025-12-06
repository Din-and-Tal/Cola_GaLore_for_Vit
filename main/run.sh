#!/bin/bash

config_names=(
    "vit_adamw"

)

set -e # terminates consecutive calls when error occurs

for config in "${config_names[@]}"; do
    echo "Running with config: $config"
    python main/main.py --config-name $config use_wandb=true  wandb_project_name="optuna" num_epochs=4 full_train=false
    echo "-----------------------------------"
done

echo "All benchmarks completed!"
