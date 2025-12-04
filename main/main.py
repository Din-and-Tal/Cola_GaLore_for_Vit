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
    cfg.size = "tiny"
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
