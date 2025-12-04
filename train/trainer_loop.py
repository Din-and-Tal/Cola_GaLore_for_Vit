import torch
import time
from contextlib import nullcontext

from util.model import save_model


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
            trainer.wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr,
            })

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

    for batch_idx, (inputs, targets) in enumerate(loader):

        # ---- LIMIT TRAINING STEPS HERE ----
        if getattr(trainer.cfg, "limit_train_steps", False) and batch_idx >= 3:
            break
        # -----------------------------------

        inputs, targets = inputs.to(device), targets.to(device)

        if is_training:
            outputs = model(pixel_values=inputs).logits
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(pixel_values=inputs).logits
                loss = loss_fn(outputs, targets)

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / max(1, (batch_idx + 1))
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc
