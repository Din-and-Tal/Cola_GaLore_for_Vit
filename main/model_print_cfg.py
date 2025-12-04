import torch
import os
import sys
import gc
import hydra
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# TODO: fix memory allocation between configs
def main():
    config_names = [
        "vit_adamw",
        "vit_glora",
        "vit_glora_layer",
        "v1_cola_adamw",
        "v1_cola_glora",
        "v1_cola_glora_layer",
        "v2_cola_adamw",
        "v2_cola_glora",
        "v2_cola_glora_layer",
    ]

    for config_name in config_names:
        print(f"\n{'='*50}")
        print(f"Running config: {config_name}")
        print(f"{'='*50}")

        # Clear any previous Hydra state
        GlobalHydra.instance().clear()

        with hydra.initialize(version_base=None, config_path="../conf"):
            current_cfg = hydra.compose(config_name=config_name)
            current_cfg.size = "large"
            print(
                f"HIDDEN_SIZE: {current_cfg.HIDDEN_SIZE}, NUM_HIDDEN_LAYERS: {current_cfg.NUM_HIDDEN_LAYERS}, NUM_ATTENTION_HEADS: {current_cfg.NUM_ATTENTION_HEADS}, INTERMEDIATE_SIZE: {current_cfg.INTERMEDIATE_SIZE}, TOTAL_PARAMS: {current_cfg.TOTAL_PARAMS}"
            )

        # Force garbage collection and clear CUDA cache after each model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
