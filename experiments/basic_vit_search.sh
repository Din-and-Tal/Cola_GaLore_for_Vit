#!/bin/bash

config_names=(
    "vit_adamw"
    # "vit_galore"
    # "vit_galore_layer"

    # "cola_adamw"
    # "cola_galore"
    # "cola_galore_layer"

)

for config in "${config_names[@]}"; do
    echo "Running with config: $config"
    python main/main.py --config-name $config wandb_project_name=basic_vit_search 

    # python main/main.py --config-name $config wandb_project_name 16,32,64,128,256,512
    # if cola works on 768: 16,32,64,128,256,512
    # reduce relo
done
