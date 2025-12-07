from datetime import datetime
from types import SimpleNamespace

import torch

import wandb
from optimizer.optimizer import get_optimizer
from train.trainer_loop import train_loop
from train.trainer_utils import set_seed, validate_trainer_initialization
from util.dataloader import get_data_loaders
from util.memory_record import profile_memory
from util.model import build_model
from util.scheduler import CosineAnnealingWarmupRestarts


class Trainer:
    """
    Trainer class to manage model training, validation, and testing.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.wandb = None  # optional
        self.model = None
        self.optimizer = None
        self.optimizer_params = {}  # for galore8PerLayer
        self.loss_func = None
        self.scheduler = None
        self.loaders = SimpleNamespace(train=None, val=None, test=None)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.GradScaler(
            enabled=getattr(cfg, "use_amp", False) and self.device == "cuda"
        )
        set_seed(cfg.seed)

        # 2. Data
        self.loaders.train, self.loaders.val, self.loaders.test = get_data_loaders(cfg)

        # 3. Model & Measure Memory
        # TODO: check how to make custom model go to gpu (ColaViTForImageClassification)
        self.model = build_model(cfg).to(self.device)

        # 4. Optimizer & Scheduler
        self.optimizer, self.optimizer_params = get_optimizer(self.model, cfg)
        self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        # Cosine Annealing Scheduler
        self.scheduler = CosineAnnealingWarmupRestarts(
            optimizer=self.optimizer,
            first_cycle_steps=cfg.scheduler_first_cycle_steps,
            cycle_mult=cfg.scheduler_cycle_mult,
            max_lr=cfg.scheduler_max_lr,
            min_lr=cfg.scheduler_min_lr,
            warmup_steps=cfg.scheduler_warmup_steps,
            gamma=cfg.scheduler_gamma,
        )

        if getattr(cfg, "compile_model", False):
            self.model = torch.compile(self.model)

        # 5. Wandb Initialization

        if cfg.use_wandb:
            today = datetime.now().strftime("%d_%m_%Y")
            project_name = (
                "test_runs"
                if not cfg.full_train
                else f"{cfg.wandb_project_name}_{today}"
            )
            self.wandb = wandb.init(
                project=project_name,
                name=f"{cfg.cola_rank_ratio}_{cfg.size}_{cfg.model_name}_{cfg.optimizer_name}_{cfg.dataset_name}",
                entity=cfg.wandb_team_name,
                config={"cfg": cfg},
            )

        # 6. run one profiling pass to get HTML memory timeline
        if getattr(cfg, "profile_memory", False):
            run_name = f"{self.cfg.model_name}_{self.cfg.optimizer_name}_{self.cfg.dataset_name}"
            profile_memory(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.loss_func,
                loader=self.loaders.train,
                num_iters=5,
                device=self.device,
                run_name=run_name,
                wandb_run=self.wandb if self.wandb is not None else None,
            )

        # Check that nothing is None
        validate_trainer_initialization(self)

    def train(self, trial=None):
        return train_loop(self, trial)
