#!/bin/bash

config_names=(
    "vit_adamw"
    # "vit_glora"
    # "vit_glora_layer"

    # "v2_cola_adamw"
    # "v2_cola_glora"
    # "v2_cola_glora_layer"
    # "v1_cola_adamw"
    # "v1_cola_glora"
    # "v1_cola_glora_layer"
)

for config in "${config_names[@]}"; do
    echo "Running with config: $config"
    python main/main.py --config-name $config wandb_project_name=basic_vit_search 

    # python main/main.py --config-name $config wandb_project_name 16,32,64,128,256,512
    # if cola works on 768: 16,32,64,128,256,512
    # reduce relo
done
