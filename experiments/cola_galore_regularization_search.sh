
## galore_scale: 1
## cola_use_intermediate_rank_scale: true
## cola_rank_ratio: 0.25
## wandb_project_name: cola_galore_search

# cola_galroe_layer sweep for cola+galore layer configuration
## run config cola_galroe_layer with the loop order:
## galore_update_proj_gap: 200,50
# label_smoothing: 0.1,0.05
# aug_rand_num_ops: 2, 3
# aug_rand_magnitude: 14, 20, 25
# weight_decay: 0.1, 0.05, 0.01, 0.005, 0.001

for galore_update_proj_gap in 200 50; do
    for label_smoothing in 0.1 0.05; do
        for aug_rand_num_ops in 2 3; do
            for aug_rand_magnitude in 14 20 25; do
                for weight_decay in 0.1 0.05 0.01 0.005 0.001; do
                    python main/main.py \
                        --config-name cola_galore_layer \
                        wandb_project_name=cola_galore_regularization_search \
                        galore_update_proj_gap=$galore_update_proj_gap \
                        galore_scale=1 \
                        cola_use_intermediate_rank_scale=true \
                        cola_rank_ratio=0.25 \
                        label_smoothing=$label_smoothing \
                        aug_rand_num_ops=$aug_rand_num_ops \
                        aug_rand_magnitude=$aug_rand_magnitude \
                        weight_decay=$weight_decay \
                        early_stopping_patience=10 \
                        use_profiler=false 
                done
            done
        done
    done
done
