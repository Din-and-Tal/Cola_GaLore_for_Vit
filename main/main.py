import os
import sys
import hydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Trainer import Trainer

@hydra.main(version_base=None, config_path="../conf", config_name="cola_glora")
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
