
## galore_scale: 1
## cola_use_intermediate_rank_scale: true
## cola_rank_ratio: 0.25
## wandb_project_name: cola_galore_long_run


python main/main.py \
    --config-name cola_galore_layer \
    wandb_project_name=cola_galore_long_run \
    galore_update_proj_gap=200 \
    galore_scale=1 \
    cola_use_intermediate_rank_scale=true \
    cola_rank_ratio=0.25 \
    label_smoothing=0.1 \
    aug_rand_num_ops=2 \
    aug_rand_magnitude=20 \
    weight_decay=0.005 \
    num_epochs=600
