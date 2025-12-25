from types import SimpleNamespace

import torch

import wandb
from optimizer.galore_setup import get_optimizer_scheduler
from train.trainer_loop import train_loop
from train.trainer_utils import set_seed
from util.dataloader import get_data_loaders
from util.memory_record import profile_memory
from util.model import build_model

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
        self.scaler = torch.GradScaler(enabled=cfg.use_amp and self.device == "cuda")
        set_seed(cfg.seed)

        # 2. Data
        self.loaders.train, self.loaders.val, self.loaders.test = get_data_loaders(cfg)
        cfg.total_steps = len(self.loaders.train) * cfg.num_epochs

        # 3. Model & Measure Memory
        # TODO: check how to make custom model go to gpu (ColaViTForImageClassification)
        self.model = build_model(cfg).to(self.device)
        
        # if cfg.use_activation_checkpointing:
        #     self.model.gradient_checkpointing_enable()
        
        # 4. Optimizer & Scheduler
        self.scheduler, self.optimizer, self.optimizer_dict = get_optimizer_scheduler(
                model=self.model, cfg=cfg
            )
        
        self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        # 5. Wandb Initialization
        if cfg.use_wandb:
            project_name = "test_runs" if not cfg.full_train else cfg.wandb_project_name
            # Convert config to dict to save all parameters individually
            config_dict = {}
            for key in dir(cfg):
                if not key.startswith("_"):
                    try:
                        value = getattr(cfg, key)
                        # Skip methods and non-serializable objects
                        if not callable(value):
                            config_dict[key] = value
                    except:
                        pass

            extra_name = ""
            if cfg.use_cola:
                extra_name += f"_cr={cfg.cola_rank_ratio}_inter={int(cfg.cola_use_intermediate_rank_scale)}"
            elif "galore" in cfg.optimizer_name:
                extra_name += f"_gr={cfg.galore_rank}_T={cfg.galore_update_proj_gap}_s={cfg.galore_scale}"
                
            test_and_bug = f""
            model_name = 'cola' if cfg.use_cola else 'vit'
            run_name= f"{cfg.size}_{model_name}_{cfg.optimizer_name}{extra_name}{test_and_bug}"
            self.wandb = wandb.init(
                project=project_name,
                name=run_name,
                entity=cfg.wandb_team_name,
                config=config_dict,
            )

            if cfg.use_profiler:
                profile_memory(
                    model=self.model,
                    cfg= self.cfg,
                    optimizer=self.optimizer,
                    loss_fn=self.loss_func,
                    loader=self.loaders.train,
                    num_iters=5,
                    device=self.device,
                    run_name=run_name,
                    wandb_run=self.wandb if self.wandb is not None else None,
                )

    def train(self, trial=None):
        return train_loop(self, trial)
