import os
import sys
import hydra
import torch
import optuna

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.trainer_setup import Trainer
from util.optuna_utils import create_optuna_study, suggest_hyperparams


@hydra.main(version_base=None, config_path="../conf", config_name="")
def main(cfg):
    torch.set_float32_matmul_precision("high")

    if cfg.get("use_optuna", False):
        study = create_optuna_study(cfg)

        def objective(trial):
            suggested_params = suggest_hyperparams(trial, cfg)
            print(
                f"\n[Trial {trial.number}] Testing hyperparameters: {suggested_params}"
            )

            cfg_copy = cfg.copy()
            for key, value in suggested_params.items():
                cfg_copy[key] = value

            trainer = Trainer(cfg_copy)

            try:
                best_loss = trainer.train(trial=trial)
                print(f"[Trial {trial.number}] Result (Val Loss): {best_loss:.4f}")
                return best_loss
            except optuna.exceptions.TrialPruned:
                print(f"[Trial {trial.number}] Pruned.")
                raise

        study.optimize(objective, n_trials=cfg.optuna.n_trials)
        print(f"Best params: {study.best_params}")

    else:
        trainer = Trainer(cfg)
        trainer.train()


if __name__ == "__main__":
    main()
