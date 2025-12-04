#!/bin/bash

config_names=(
    "vit_adamw"
    "vit_glora"
    "vit_glora_layer"
    "v1_cola_adamw"
    "v1_cola_glora"
    "v1_cola_glora_layer"
    "v2_cola_adamw"
    "v2_cola_glora"
    "v2_cola_glora_layer"
)

for config in "${config_names[@]}"; do
    echo "Running with config: $config"
    python main/main.py --config-name $config full_train=false size=tiny use_wandb=false
    echo "-----------------------------------"
    sleep 5
done

echo "All benchmarks completed!"
