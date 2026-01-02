#!/bin/bash

config_names=(
    "vit_adamw"
    "vit_galore_layer"
    "cola_adamw"
    "cola_galore_layer"
)
set -e

for config in "${config_names[@]}"; do
    echo "Running with config: $config"
    python main/main.py --config-name $config full_train=false
done
