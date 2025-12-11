import time

import optuna
import torch

from train.trainer_utils import apply_mixup_cutmix


def train_loop(trainer, trial):
    """Run full training using the Trainer object."""
    cfg = trainer.cfg
    best_acc = 0.0
    best_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = getattr(cfg, "early_stopping_patience", -1)
    early_stopping_patience = -1 if cfg.use_optuna else early_stopping_patience

    # Limit training when full_train is False
    num_epochs = cfg.num_epochs if cfg.full_train else 1
    run_eval = cfg.full_train

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, train_acc = epoch_step(
            trainer, trainer.loaders.train, True, cfg.full_train
        )

        if run_eval:
            val_loss, val_acc = epoch_step(
                trainer, trainer.loaders.val, False, cfg.full_train
            )
            best_acc = val_acc if val_acc > best_acc else best_acc

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # ----- Optuna Pruning -----
            if trial is not None:
                trial.report(val_loss, epoch)  # Report loss instead of acc
                if trial.should_prune():
                    print(
                        f"[Epoch {epoch}] Trial pruned by Optuna (val_acc={val_acc:.2f})"
                    )
                    raise optuna.exceptions.TrialPruned()

            if (
                early_stopping_patience > 0
                and patience_counter >= early_stopping_patience
            ):
                print(
                    f"\n[Early Stopping] Triggered after {patience_counter} epochs without improvement."
                )
                break

        else:
            val_loss, val_acc = 0.0, 0.0

        trainer.scheduler.step()
        lr = trainer.optimizer.param_groups[0]["lr"]
        if cfg.verbose:
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

    # Final test
    if run_eval:
        _, test_acc = epoch_step(trainer, trainer.loaders.test, False, cfg.full_train)
    else:
        test_acc = 0.0
    print(f"Training finished in {(time.time() - start_time) / 60:.2f} minutes.")
    print(f"Best Val Acc: {best_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    if trainer.wandb:
        trainer.wandb.log({"test_acc": test_acc})
        trainer.wandb.finish()

    return best_loss  # Return loss for minimization


def epoch_step(trainer, loader, is_training, full_train=True):
    model = trainer.model
    optimizer = trainer.optimizer
    loss_fn = trainer.loss_func
    device = trainer.device
    scaler = trainer.scaler
    cfg = trainer.cfg

    total_loss, correct, total = 0.0, 0, 0
    batch_idx = 0
    max_batches = None if full_train else 1

    model.train() if is_training else model.eval()

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            # ----- Mixup / CutMix -----
            inputs_mixed, targets_a, targets_b, lam, used_mix = apply_mixup_cutmix(
                inputs, targets, trainer.cfg, device
            )

            optimizer.zero_grad()

            # ----- AMP forward -----
            with torch.autocast(device_type=device, enabled=scaler.is_enabled()):
                outputs = model(pixel_values=inputs_mixed).logits
                if used_mix:
                    loss = lam * loss_fn(outputs, targets_a) + (1.0 - lam) * loss_fn(
                        outputs, targets_b
                    )
                else:
                    loss = loss_fn(outputs, targets)

            # ----- AMP backward + step -----
            if scaler.is_enabled():
                scaler.scale(loss).backward()

                # optional: gradient clipping
                if getattr(cfg, "max_grad_norm", 0.0) and cfg.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if getattr(cfg, "max_grad_norm", 0.0) and cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
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
