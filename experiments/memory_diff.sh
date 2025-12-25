#!/bin/bash

config_names=(
    # "vit_adamw"
    "vit_galore_layer"
#    "cola_adamw"
    "cola_galore_layer"
)

set -e # terminates consecutive calls when error occurs

for config in "${config_names[@]}"; do
    echo "Running with config: $config"

    python main/main.py --config-name $config full_train=false size=huge batch_size=64 use_profiler=true use_activation_checkpointing=true
    python main/main.py --config-name $config full_train=false size=huge batch_size=64 use_profiler=true

done
