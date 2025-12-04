import os
import sys
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Trainer import Trainer

# TODO: iterate over configs with proceses create
# TODO: override with cmdline cfg parameter and iterate in code
# TODO: make base cfg

@hydra.main(version_base=None, config_path="../conf", config_name="vit_glora")
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
