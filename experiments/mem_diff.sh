#!/bin/bash

config_names=(
    "vit_adamw"
    # "vit_galore_layer"
    #"cola_adamw"
    "cola_galore_layer"
)


for config in "${config_names[@]}"; do
    echo "Running with config: $config"
    python main/main.py --config-name $config optimizer_name=adamw8 full_train=false size=huge batch_size=64
    python main/main.py --config-name $config optimizer_name=galore full_train=false galore_rank=128 galore_scale=0.25 size=huge batch_size=64

done
