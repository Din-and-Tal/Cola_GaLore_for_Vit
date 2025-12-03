import os
import sys
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Trainer import Trainer
from util.profiler import zzzbenchmark_training_memory
from torchinfo import summary


# @hydra.main(version_base=None, config_path="../conf", config_name="cola_adamw")
def main():
    config_names = ["vit_adamw","vit_glora","vit_glora_layer","cola_adamw","cola_glora","cola_glora_layer"]

    for config_name in config_names:
        print(f"\n{'='*50}")
        print(f"Running config: {config_name}")
        print(f"{'='*50}")

        with hydra.initialize(version_base=None, config_path="../conf"):
            current_cfg = hydra.compose(config_name=config_name)
            current_cfg.USE_WANDB = False
            trainer = Trainer(current_cfg)
            data, target = next(iter(trainer.loaders.train))
            # zzzbenchmark_training_memory(model=trainer.model,lossFunc=trainer.lossFunc,optimizer=trainer.optimizer,input_data=data,target=target)
            if trainer.model is not None:
                summary(trainer.model, input_size=data.shape)


if __name__ == "__main__":
    main()
