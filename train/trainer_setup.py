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
        if cfg.use_bf16:
            self.model = build_model(cfg).to(self.device, dtype=torch.bfloat16)
        else:
            self.model = build_model(cfg).to(self.device)

        # Print setup information
        self._print_setup_information()

        # 4. Optimizer & Scheduler
        self.scheduler, self.optimizer, self.optimizer_dict = get_optimizer_scheduler(
            model=self.model, cfg=self.cfg
        )

        self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        # 5. Wandb Initialization
        if cfg.use_wandb:
            project_name = "test_runs" if not cfg.full_train and cfg.wandb_project_name=="no_project_name" else cfg.wandb_project_name
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
            if "galore" in cfg.optimizer_name:
                extra_name += f"_gr={cfg.galore_rank}_T={cfg.galore_update_proj_gap}_s={cfg.galore_scale}"

            test_and_bug = f"_cp={int(cfg.use_checkpointing)}"
            if cfg.use_bf16:
                test_and_bug = "_bf16"

            model_name = "cola" if cfg.use_cola else "vit"
            run_name = f"{cfg.size}_{model_name}_{cfg.optimizer_name}{extra_name}{test_and_bug}"
            self.wandb = wandb.init(
                project=project_name,
                name=run_name,
                entity=cfg.wandb_team_name,
                config=config_dict,
                dir="./outputs/wandb",
            )

            if cfg.use_profiler:
                profile_memory(
                    model=self.model,
                    cfg=self.cfg,
                    optimizer=self.optimizer,
                    loss_fn=self.loss_func,
                    loader=self.loaders.train,
                    num_iters=5,
                    device=self.device,
                    run_name=run_name,
                    wandb_run=self.wandb if self.wandb is not None else None,
                )

    def _print_setup_information(self):
        """Print model setup information including dataset sizes and training configuration."""
        cfg = self.cfg
        model_name = "cola" if cfg.use_cola else "vit"
        dataset_name = getattr(
            cfg, "dataset_name", "CIFAR10"
        )  # fallback for old dataloader
        batches_per_epoch = len(self.loaders.train)
        debug_mode = not cfg.full_train

        # Get number of examples in each dataset
        try:
            num_train_examples = len(self.loaders.train.dataset)
        except (TypeError, AttributeError):
            num_train_examples = "N/A"
        try:
            num_val_examples = (
                len(self.loaders.val.dataset) if self.loaders.val is not None else 0
            )
        except (TypeError, AttributeError):
            num_val_examples = "N/A"
        try:
            num_test_examples = (
                len(self.loaders.test.dataset) if self.loaders.test is not None else 0
            )
        except (TypeError, AttributeError):
            num_test_examples = "N/A"

        print("\n" + "=" * 60)
        print("MODEL SETUP INFORMATION")
        print("=" * 60)
        print(f"Model Name: {model_name.upper()}")
        print(f"Optimizer: {cfg.optimizer_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Number of Epochs: {cfg.num_epochs}")
        print(f"Batches per Epoch: {batches_per_epoch}")
        print(f"Total Batches: {cfg.num_epochs * batches_per_epoch}")
        train_examples_str = (
            f"{num_train_examples:,}"
            if isinstance(num_train_examples, int)
            else str(num_train_examples)
        )
        val_examples_str = (
            f"{num_val_examples:,}"
            if isinstance(num_val_examples, int)
            else str(num_val_examples)
        )
        test_examples_str = (
            f"{num_test_examples:,}"
            if isinstance(num_test_examples, int)
            else str(num_test_examples)
        )
        print(f"Train Examples: {train_examples_str}")
        print(f"Validation Examples: {val_examples_str}")
        print(f"Test Examples: {test_examples_str}")
        print(
            f"Debug Mode: {'YES' if debug_mode else 'NO'} (full_train={cfg.full_train})"
        )
        print("=" * 60 + "\n")

    def train(self, trial=None):
        return train_loop(self, trial)
