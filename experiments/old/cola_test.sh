#!/bin/bash

config_names=(
    "vit_adamw"
    "vit_galore_layer"
    "cola_adamw"
    # "cola_galore_layer"
)

for config in "${config_names[@]}"; do

    echo "Running with config: $config"
    python main/main.py --config-name $config size=huge batch_size=64 use_activation_checkpointing=true early_stopping_patience=20
    python main/main.py --config-name $config size=huge batch_size=64 early_stopping_patience=20
    
done
