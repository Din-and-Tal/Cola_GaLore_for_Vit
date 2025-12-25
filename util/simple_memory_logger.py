# util/simple_memory_logger.py
import torch

from util.model import build_model


def _mem_gb(device: torch.device) -> float:
    return torch.cuda.memory_allocated(device) / 1024**3


def _peak_mem_gb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024**3


def measure_memory_breakdown(model,cfg, optimizer, loss_fn, loader, device="cuda"):
    """
    Run a single normal training step and report an approximate breakdown:

      - parameter_memory_gb        (from param sizes)
      - optimizer_state_memory_gb  (allocated - params)
      - activation_memory_gb       (extra mem after forward)
      - gradient_memory_gb         (sum of all .grad tensors)
      - total_peak_memory_gb       (CUDA peak during step)

    Works with AdamW / GaLore as long as a normal step works.
    """
    if not torch.cuda.is_available():
        print("[MemLogger] CUDA not available, skipping measurement.")
        return None

    device = torch.device(device)
    if cfg.use_bf16:
        model = build_model(cfg).to(device, dtype=torch.bfloat16)
    else:
        model = build_model(cfg).to(device)

    model.train()

    # One batch
    try:
        inputs, targets = next(iter(loader))
    except StopIteration:
        print("[MemLogger] Loader is empty, cannot measure.")
        return None

    inputs, targets = inputs.to(device), targets.to(device)

    # ---- 0. Clear + reset stats ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # ---- 1. Parameter + optimizer memory (static) ----
    mem_after_model_opt = _mem_gb(device)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    parameter_memory_gb = param_bytes / 1024**3

    optimizer_state_memory_gb = max(mem_after_model_opt - parameter_memory_gb, 0.0)

    # ---- 2. Forward (activations) ----
    mem_baseline_gb = _mem_gb(device)

    outputs = model(pixel_values=inputs).logits
    loss = loss_fn(outputs, targets)

    torch.cuda.synchronize()
    mem_after_fwd_gb = _mem_gb(device)

    activation_memory_gb = max(mem_after_fwd_gb - mem_baseline_gb, 0.0)

    # ---- 3. Backward (gradients) ----
    if 'layer' not in cfg.optimizer_name:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.cuda.synchronize()

    # Sum actual gradient tensors
    grad_bytes = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_bytes += p.grad.numel() * p.grad.element_size()
    gradient_memory_gb = grad_bytes / 1024**3

    # Peak memory during the whole step
    total_peak_memory_gb = _peak_mem_gb(device)

    # ---- 4. Optimizer step (for realism) ----
    if 'layer' not in cfg.optimizer_name:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    breakdown = {
        "parameter_memory_gb": parameter_memory_gb,
        "optimizer_state_memory_gb": optimizer_state_memory_gb,
        "activation_memory_gb": activation_memory_gb,
        "gradient_memory_gb": gradient_memory_gb,
        "total_peak_memory_gb": total_peak_memory_gb,
    }

    print("\n[MemLogger] Memory breakdown (approx):")
    for k, v in breakdown.items():
        print(f"  {k}: {v:.3f} GiB")

    return breakdown
