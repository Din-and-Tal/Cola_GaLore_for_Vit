import os
import time
import torch
from contextlib import nullcontext
from types import SimpleNamespace

from tqdm import tqdm
import wandb

from optimizer.optimizer import get_optimizer
from util.dataloader import get_data_loaders
from util.general import set_seed
from util.model import build_model, load_model, save_model
from util.scheduler import CosineAnnealingWarmupRestarts


class Trainer:

    def __init__(self, cfg, debug=True):
        self.debug = debug
        self.cfg = cfg
        self.wandb = None  # optional
        self.model = None
        self.optimizer = None
        self.optimizerParams = {}  # for galore8PerLayer
        self.lossFunc = None
        self.scheduler = None
        self.scaler = None
        self.loaders = SimpleNamespace(train=None, val=None, test=None)

        if debug:
            set_seed(cfg.SEED)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2. Data
        self.loaders.train, self.loaders.val, self.loaders.test = get_data_loaders(
            cfg, debug
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

        # Mixed Precision Scaler
        self.scaler = torch.GradScaler(device=self.device, enabled=cfg.USE_AMP)

        if cfg.LOAD_MODEL:
            modelPath = os.path.join(cfg.OUTPUT_DIR, "model.pth")
            load_model(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
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

        # Check that nothing is None
        self._validate_initialization()

    def train(self):
        best_acc = 0.0
        print("\nStarting training...")
        start_time = time.time()
        for epoch in range(self.cfg.NUM_EPOCHS):

            # Train
            train_loss, train_acc = self._epoch_step(
                loader=self.loaders.train, is_training=True, epoch=epoch
            )

            # Validate
            val_loss, val_acc = self._epoch_step(
                loader=self.loaders.val, is_training=False, epoch=epoch
            )
            # Scheduler Step
            self.scheduler.step()  # type: ignore
            current_lr = self.optimizer.param_groups[0]["lr"]  # type: ignore

            print(
                f"Summary Ep {epoch+1}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}"
            )
            # TODO: check if running wandb offline increases speed (not sync)
            if self.wandb:
                self.wandb.log(  # type: ignore
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "lr": current_lr,
                    }
                )

            # saving model
            if self.cfg.SAVE_MODEL:
                best_acc = save_model(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch,
                    train_loss,
                    val_acc,
                    best_acc,
                    self.cfg,
                    optimizer_dict=self.optimizerParams,  # Pass optimizer_dict
                )
            # TODO: check how torch.cuda.empty_cache() effects training

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.2f} minutes.")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")

        test_acc = self._epoch_step(
            loader=self.loaders.test, is_training=False, epoch=epoch
        )

        print(f"Test Accuracy: {best_acc:.2f}%")

        if self.wandb:
            self.wandb.log({"test_acc": test_acc})
            self.wandb.finish()

    def _epoch_step(self, loader, is_training, epoch):
        total_loss, correct, total = 0, 0, 0

        # TODO: check if ctx is right
        if is_training:
            ctx = torch.autocast("cuda") if self.cfg.USE_AMP else nullcontext()
        else:
            ctx = torch.no_grad()

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{self.cfg.NUM_EPOCHS} [{'Train' if is_training else 'Val'}]",
        )

        self.model.train() if is_training else self.model.eval()  # type: ignore
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward
            with ctx:
                outputs = self.model(pixel_values=inputs).logits  # type: ignore
                loss = self.lossFunc(outputs, targets)  # type: ignore

            # Backward
            if is_training:
                self.optimizer.zero_grad()  # type: ignore

                if self.cfg.USE_AMP:
                    self.scaler.scale(loss).backward()  # type: ignore
                    self.scaler.step(self.optimizer)  # type: ignore
                    self.scaler.update()  # type: ignore
                else:
                    loss.backward()
                    self.optimizer.step()  # type: ignore

            # Stats
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            # log pbar
            acc = 100.0 * correct / total
            lr = self.optimizer.param_groups[0]["lr"]  # type: ignore
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{acc:.2f}%",
                    "LR": f"{lr:.2e}",
                }
            )

        return total_loss / len(loader), 100.0 * correct / total

    def _validate_initialization(self):
        none_attrs = []
        for attr_name in [
            "cfg",
            "model",
            "optimizer",
            "lossFunc",
            "scheduler",
            "scaler",
        ]:
            if getattr(self, attr_name) is None:
                none_attrs.append(attr_name)

        if self.loaders.train is None:
            none_attrs.append("loaders.train")
        if self.loaders.val is None:
            none_attrs.append("loaders.val")
        if self.loaders.test is None:
            none_attrs.append("loaders.test")

        if getattr(self, "wandb") is None and self.cfg.USE_WANDB:
            none_attrs.append("wandb")

        # no need to validate optimizerParams (condition might change in future)

        if none_attrs:
            raise ValueError(
                f"The following attributes are None: {', '.join(none_attrs)}"
            )
