#!/bin/bash

config_names=(
    "vit_adamw"
    # "vit_glora"
    # "vit_glora_layer"
    # "cola_adamw"
    # "cola_glora"
    # "cola_glora_layer"

)

set -e # terminates consecutive calls when error occurs

for config in "${config_names[@]}"; do
    echo "Running with config: $config"
    python main/main.py --config-name $config full_train=false
done
