#
## run config cola_galroe_layer with the loop order:
## use_bf16: false, true (most outer loop)
## galore_update_proj_gap : 100, 50
## galore_scale: 1, 0.5
## cola_use_intermediate_rank_scale: true, false
## cola_rank_ratio: 0.25 ,0.5 ,0.75
## Experiment: sweep for cola_galore_layer.

 python main/main.py \
            --config-name cola_adamw \
            cola_use_intermediate_rank_scale=true \
            cola_rank_ratio=0.1 \
            wandb_project_name=cola_and_glora_combined_WD \
            use_profiler=false \
            weight_decay=0 \

#
#for use_bf16 in false true; do
#  for cola_use_intermediate_rank_scale in true false; do
#    for cola_rank_ratio in 0.5 0.75 0.25 1; do
#      for galore_scale in 0.5 0.25 1; do
#        for galore_update_proj_gap in 10 25 50 100 200; do
#          python main/main.py \
#            --config-name cola_galore_layer \
#            use_bf16=${use_bf16} \
#            galore_update_proj_gap=${galore_update_proj_gap} \
#            galore_scale=${galore_scale} \
#            cola_use_intermediate_rank_scale=${cola_use_intermediate_rank_scale} \
#            cola_rank_ratio=${cola_rank_ratio} \
#            wandb_project_name=cola_and_glora_combined_WD \
#            use_profiler=false \
#            weight_decay=1e-3 \
#            early_stopping_patience=4
#        done
#      done
#    done
#  done
#done
