import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from Trainer import Trainer

@hydra.main(version_base=None, config_path="../conf", config_name="cola_adamw")
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
