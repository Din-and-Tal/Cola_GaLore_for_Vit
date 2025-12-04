import os
import sys
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.trainer_setup import Trainer


@hydra.main(version_base=None, config_path="../conf", config_name="vit_glora")
def main(cfg):
    cfg.size = "tiny"
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
