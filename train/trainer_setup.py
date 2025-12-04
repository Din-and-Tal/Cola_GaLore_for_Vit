import os
import time
import torch
from contextlib import nullcontext
from types import SimpleNamespace

import wandb

from optimizer.optimizer import get_optimizer
from train.trainer_utils import validate_trainer_initialization
from util.dataloader import get_data_loaders
from train.trainer_utils import set_seed
from util.memory_record import profile_memory
from util.model import build_model, load_model, save_model
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
        self.optimizerParams = {}  # for galore8PerLayer
        self.lossFunc = None
        self.scheduler = None
        self.loaders = SimpleNamespace(train=None, val=None, test=None)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        set_seed(cfg.SEED)

        # 2. Data
        self.loaders.train, self.loaders.val, self.loaders.test = get_data_loaders(
            cfg
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

        # Optional: run one profiling pass to get HTML memory timeline
        if getattr(cfg, "PROFILE_MEMORY", True):
            run_name = f"{self.cfg.MODEL_NAME}_{self.cfg.OPTIMIZER_NAME}_{self.cfg.DATASET_NAME}"
            profile_memory(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.lossFunc,
                loader=self.loaders.train,
                num_iters=5,
                device=self.device,
                run_name=run_name,
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

        # Check that nothing is None
        validate_trainer_initialization(self)
        
    def train(self):
        train_loop(self)

def train_loop(trainer):
    """Run full training using the Trainer object."""
    cfg = trainer.cfg
    best_acc = 0.0

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(cfg.NUM_EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc = epoch_step(trainer, trainer.loaders.train, True)
        val_loss, val_acc = epoch_step(trainer, trainer.loaders.val, False)
        best_acc = val_acc if val_acc > best_acc else best_acc

        trainer.scheduler.step()
        lr = trainer.optimizer.param_groups[0]["lr"]

        print(
            f"Summary Ep {epoch}: "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"LR: {lr:.2e} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Epoch time: {time.time() - epoch_start:.2f}s"
        )

        # Wandb logging
        if trainer.wandb:
            trainer.wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": lr,
                }
            )

        # Save best model
        if cfg.SAVE_MODEL:
            best_acc = save_model(
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                epoch,
                train_loss,
                val_acc,
                best_acc,
                cfg,
                optimizer_dict=trainer.optimizerParams,
            )

    # Final test
    _, test_acc = epoch_step(trainer, trainer.loaders.test, False)
    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")
    print(f"Best Val Acc: {best_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    if trainer.wandb:
        trainer.wandb.log({"test_acc": test_acc})
        trainer.wandb.finish()


def epoch_step(trainer, loader, is_training):
    model = trainer.model
    optimizer = trainer.optimizer
    loss_fn = trainer.lossFunc
    device = trainer.device

    total_loss, correct, total = 0.0, 0, 0

    ctx = nullcontext() if is_training else torch.no_grad()
    model.train() if is_training else model.eval()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with ctx:
            outputs = model(pixel_values=inputs).logits
            loss = loss_fn(outputs, targets)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc
