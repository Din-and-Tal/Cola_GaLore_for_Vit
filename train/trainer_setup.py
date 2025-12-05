import os
import torch
from torch.cuda.amp import GradScaler
from types import SimpleNamespace

import wandb

from optimizer.optimizer import get_optimizer
from train.trainer_loop import train_loop
from train.trainer_utils import validate_trainer_initialization
from util.dataloader import get_data_loaders
from train.trainer_utils import set_seed
from util.memory_record import profile_memory
from util.model import build_model, load_model
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
        self.scaler = GradScaler(enabled=getattr(cfg, "use_amp", False) and self.device == "cuda")

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

        if cfg.load_model:
            model_path = os.path.join(cfg.output_dir, "model.pth")
            load_model(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                model_path=model_path,
                device=self.device,
                optimizer_params=self.optimizer_params,
            )

        if getattr(cfg, "compile_model", False):
            self.model = torch.compile(self.model)

        # 5. Wandb Initialization
        # TODO: capture important data
        if cfg.use_wandb:
            self.wandb = wandb.init(
                project=cfg.wandb_project_name,
                name=f"{cfg.size}_{cfg.model_name}_{cfg.optimizer_name}_{cfg.dataset_name}",
                entity=cfg.wandb_team_name,
                config={
                    "model_name": cfg.model_name,
                    "dataset": cfg.dataset_name,
                    "epochs": cfg.num_epochs,
                    "batch_size": cfg.batch_size,
                    "learning_rate": cfg.learning_rate,
                    "weight_decay": cfg.weight_decay,
                    "image_size": cfg.image_size,
                    "patch_size": cfg.patch_size,
                    "cola_rank": cfg.cola_rank,
                },
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

    def train(self):
        train_loop(self)
