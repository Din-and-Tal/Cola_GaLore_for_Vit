import os
import sys

import hydra
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.trainer_setup import Trainer
from util.optuna_utils import run_optuna


@hydra.main(version_base=None, config_path="../conf", config_name="vit_galore_layer")
def main(cfg):
    torch.set_float32_matmul_precision("high")

    if cfg.use_optuna:
        run_optuna(cfg)

    else:
        trainer = Trainer(cfg)
        trainer.train()


if __name__ == "__main__":
    main()
