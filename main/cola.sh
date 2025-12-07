#!/bin/bash

# config_names=(
    # "vit_adamw"
    # "vit_glora"
    # "vit_glora_layer"
    # "v1_cola_adamw"
    # "v1_cola_glora"
    # "v1_cola_glora_layer"
    # "v2_cola_adamw"
    # "v2_cola_glora"
    # "v2_cola_glora_layer"
# )

python main/main.py --config-name v2_cola_adamw wandb_project_name=cola_v2_rank_search
