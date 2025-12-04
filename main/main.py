import os
import sys
import hydra

from train.trainer_setup import Trainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# TODO: iterate over configs with proceses create
# TODO: override with cmdline cfg parameter and iterate in code
# TODO: make base cfg

@hydra.main(version_base=None, config_path="../conf", config_name="vit_general")

def main(cfg):
    cfg.size = "huge"
    # cfg.MODEL_NAME = "v1_cola"
    cfg.OPTIMIZER_NAME = "galore8"

    cfg.PROFILE_MEMORY = True
    cfg.COLA_RANK = 256
    cfg.GLORA_RANK = 256
    cfg.limit_train_steps = True
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
