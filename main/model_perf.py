import os
import sys
import gc
import hydra
from hydra.core.global_hydra import GlobalHydra
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from Trainer import Trainer


# TODO: fix memory allocation between configs
def main():
    # config_names = [
    #     "vit_adamw",
    #     "vit_glora",
    #     "vit_glora_layer",
    #     "v1_cola_adamw",
    #     "v1_cola_glora",
    #     "v1_cola_glora_layer",
    #     "v2_cola_adamw",
    #     "v2_cola_glora",
    #     "v2_cola_glora_layer",
    # ]
    config_names = ["v1_cola_adamw"]

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "model_perf")
    os.makedirs(output_dir, exist_ok=True)

    for config_name in config_names:
        output_file = os.path.join(output_dir, f"{config_name}.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print(f"\n{'='*50}")
                print(f"Running config: {config_name}")
                print(f"{'='*50}")

                # Clear any previous Hydra state
                GlobalHydra.instance().clear()

                with hydra.initialize(version_base=None, config_path="../conf"):
                    current_cfg = hydra.compose(config_name=config_name)
                    trainer = Trainer(current_cfg, debug=True, fullTrain=False)
                    trainer.train()

                    # Explicitly delete trainer to free memory
                    del trainer
                    del current_cfg

        # Force garbage collection and clear CUDA cache after each model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
