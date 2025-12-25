#!/bin/bash

config_names=(
    #  "vit_adamw"
#     "vit_galore"
   "cola_adamw"
#    "cola_galore"
)

set -e # terminates consecutive calls when error occurs

#python main/main.py --config-name vit_adamw full_train=false size=huge batch_size=64 big_checkpointing=false cola_use_checkpointing=false
for config in "${config_names[@]}"; do
    echo "Running with config: $config"

    python main/main.py --config-name $config full_train=false size=huge batch_size=64 big_checkpointing=false cola_use_checkpointing=false
#    python main/main.py --config-name $config full_train=false size=huge batch_size=64 big_checkpointing=true cola_use_checkpointing=false
#    python main/main.py --config-name $config full_train=false size=huge batch_size=64 big_checkpointing=false cola_use_checkpointing=true
#    python main/main.py --config-name $config full_train=false size=huge batch_size=64 big_checkpointing=true cola_use_checkpointing=true

done
