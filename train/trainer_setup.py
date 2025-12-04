import os
import torch
from types import SimpleNamespace

import wandb

from optimizer.optimizer import get_optimizer
from train.trainer_utils import validate_trainer_initialization
from util.dataloader import get_data_loaders
from util.general import set_seed
from util.memory_record import profile_memory
from util.model import build_model, load_model
from util.scheduler import CosineAnnealingWarmupRestarts
from train.trainer_loop import train_loop


class Trainer:
    """
    Trainer class to manage model training, validation, and testing.
    """
    def __init__(self, cfg, debug=True,fullTrain=True):
        self.debug = debug
        self.cfg = cfg
        self.wandb = None  # optional
        self.model = None
        self.optimizer = None
        self.optimizerParams = {}  # for galore8PerLayer
        self.lossFunc = None
        self.scheduler = None
        self.loaders = SimpleNamespace(train=None, val=None, test=None)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        set_seed(cfg.SEED)

        # 2. Data
        self.loaders.train, self.loaders.val, self.loaders.test = get_data_loaders(
            cfg, debug,fullTrain
        )

        # 3. Model & Measure Memory
        # TODO: check how to make custom model go to gpu (ColaViTForImageClassification)
        self.model = build_model(cfg).to(self.device)

        # 4. Optimizer & Scheduler
        self.optimizer, self.optimizerParams = get_optimizer(self.model, cfg)
        self.lossFunc = torch.nn.CrossEntropyLoss()

        # Cosine Annealing Scheduler
        self.scheduler = CosineAnnealingWarmupRestarts(
            optimizer=self.optimizer,
            first_cycle_steps=cfg.SCHEDULER_FIRST_CYCLE_STEPS,
            cycle_mult=cfg.SCHEDULER_CYCLE_MULT,
            max_lr=cfg.SCHEDULER_MAX_LR,
            min_lr=cfg.SCHEDULER_MIN_LR,
            warmup_steps=cfg.SCHEDULER_WARMUP_STEPS,
            gamma=cfg.SCHEDULER_GAMMA,
        )





        if cfg.LOAD_MODEL:
            modelPath = os.path.join(cfg.OUTPUT_DIR, "model.pth")
            load_model(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                modelPath=modelPath,
                device=self.device,
                optimizerParams=self.optimizerParams,
            )

        # 5. Wandb Initialization
        # TODO: capture important data
        if cfg.USE_WANDB:
            self.wandb = wandb.init(
                project=cfg.WANDB_PROJECT_NAME,
                name=f"{cfg.MODEL_NAME}_{cfg.OPTIMIZER_NAME}_{cfg.DATASET_NAME}",
                entity=cfg.WANDB_TEAM_NAME,
                config={
                    "model_name": cfg.MODEL_NAME,
                    "dataset": cfg.DATASET_NAME,
                    "epochs": cfg.NUM_EPOCHS,
                    "batch_size": cfg.BATCH_SIZE,
                    "learning_rate": cfg.LEARNING_RATE,
                    "weight_decay": cfg.WEIGHT_DECAY,
                    "image_size": cfg.IMAGE_SIZE,
                    "patch_size": cfg.PATCH_SIZE,
                    "cola_rank": cfg.COLA_RANK,
                },
            )

        # 6. run one profiling pass to get HTML memory timeline
        if getattr(cfg, "PROFILE_MEMORY", False):
            run_name = f"{self.cfg.MODEL_NAME}_{self.cfg.OPTIMIZER_NAME}_{self.cfg.DATASET_NAME}"
            profile_memory(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.lossFunc,
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

