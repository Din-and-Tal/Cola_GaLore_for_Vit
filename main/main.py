import os
import sys
import hydra
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.trainer_setup import Trainer
from util.optuna_utils import create_optuna_study, suggest_hyperparams



@hydra.main(version_base=None, config_path="../conf", config_name="")
def main(cfg):
    torch.set_float32_matmul_precision('high')

    if cfg.get('use_optuna', False):
        study = create_optuna_study(cfg)

        def objective(trial):
            suggested_params = suggest_hyperparams(trial, cfg)
            print(f"\n[Trial {trial.number}] Testing hyperparameters: {suggested_params}")

            cfg_copy = cfg.copy()
            for key, value in suggested_params.items():
                cfg_copy[key] = value

            trainer = Trainer(cfg_copy)
            best_accuracy = trainer.train(trial=trial)
            print(f"[Trial {trial.number}] Result: {best_accuracy:.2f}%")

            return best_accuracy

        study.optimize(objective, n_trials=cfg.optuna.n_trials)
        print(f"Best params: {study.best_params}")

    else:
        trainer = Trainer(cfg)
        trainer.train()


if __name__ == "__main__":
    main()
