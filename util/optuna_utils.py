import optuna
from optuna.pruners import SuccessiveHalvingPruner, PatientPruner
from optuna.samplers import TPESampler


def create_optuna_study(config) -> optuna.Study:
    sampler = TPESampler(seed=config.seed)

    pruner = SuccessiveHalvingPruner(min_resource=config.optuna.n_warmup_steps, reduction_factor=3)
    if "early_stopping_patience" in config.optuna:
        pruner = PatientPruner(pruner, patience=config.optuna.early_stopping_patience)

    study = optuna.create_study(
        study_name=config.optuna.study_name,
        direction=config.optuna.direction,
        pruner=pruner,
        sampler=sampler,
    )
    return study


def suggest_hyperparams(trial, cfg):
    search_space = cfg.optuna.search_space
    suggested_params = {}

    # Learning rate
    if "scheduler_max_lr" in search_space:
        lr_min, lr_max = search_space["scheduler_max_lr"]
        suggested_params["scheduler_max_lr"] = trial.suggest_float(
            "scheduler_max_lr", lr_min, lr_max, log=True
        )

    # Weight decay
    if "weight_decay" in search_space:
        wd_min, wd_max = search_space["weight_decay"]
        suggested_params["weight_decay"] = trial.suggest_float(
            "weight_decay", wd_min, wd_max, log=True
        )

    # Label smoothing
    if "label_smoothing" in search_space:
        ls_min, ls_max = search_space["label_smoothing"]
        suggested_params["label_smoothing"] = trial.suggest_float(
            "label_smoothing", ls_min, ls_max
        )

    if "hidden_dropout_prob" in search_space:
        drop_min, drop_max = search_space["hidden_dropout_prob"]
        suggested_params["hidden_dropout_prob"] = trial.suggest_float(
            "hidden_dropout_prob", drop_min, drop_max
        )

    # Mixup alpha (ADD THIS)
    if "mixup_alpha" in search_space:
        mixup_min, mixup_max = search_space["mixup_alpha"]
        suggested_params["mixup_alpha"] = trial.suggest_float(
            "mixup_alpha", mixup_min, mixup_max
        )

    # Cutmix alpha (ADD THIS)
    if "cutmix_alpha" in search_space:
        cutmix_min, cutmix_max = search_space["cutmix_alpha"]
        suggested_params["cutmix_alpha"] = trial.suggest_float(
            "cutmix_alpha", cutmix_min, cutmix_max
        )

    # Mix probability (ADD THIS)
    if "mix_prob" in search_space:
        mix_prob_min, mix_prob_max = search_space["mix_prob"]
        suggested_params["mix_prob"] = trial.suggest_float(
            "mix_prob", mix_prob_min, mix_prob_max
        )

    if "cola_rank_ratio" in search_space:
        cola_rank_ratio_min, cola_rank_ratio_max = search_space["cola_rank_ratio"]
        suggested_params["cola_rank_ratio"] = trial.suggest_float(
            "cola_rank_ratio", cola_rank_ratio_min, cola_rank_ratio_max
        )

    return suggested_params
