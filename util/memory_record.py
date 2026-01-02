import os
import tempfile
import torch
from torch.profiler import profile, ProfilerActivity, record_function
import wandb
from util.model import build_model


def profile_memory(
    model,
    optimizer,
    loss_fn,
    loader,
    num_iters,
    device,
    run_name,
    cfg,
    wandb_run=None,
):
    """
    Same API as before.
    - Logs HTML timeline to W&B (no local files kept)
    - Logs numeric breakdown to W&B
    - Logs forward/backward/iter timing to W&B
    """

    # 1) Profiler HTML + timing
    profile_memory_once(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        loader=loader,
        cfg=cfg,
        num_iters=num_iters,
        device=device,
        run_name=run_name,
        wandb_run=wandb_run,   # <-- NEW: pass through
    )

    # 2) Simple numeric breakdown (params / optimizer / activations / grads)
    mem_stats = measure_memory_breakdown(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        loader=loader,
        device=device,
        cfg=cfg
    )

    # 3) Optional: log numeric stats to W&B
    if wandb_run is not None and mem_stats is not None:
        wandb_run.log({f"memory/{k}": v for k, v in mem_stats.items()})


def _trace_handler_html_only(prof: torch.profiler.profile, wandb_run, wandb_key: str):
    # Export ONLY HTML memory timeline into a temp file, then delete.
    with tempfile.TemporaryDirectory() as td:
        html_path = os.path.join(td, "memory_timeline.html")
        prof.export_memory_timeline(html_path, device="cuda:0")

        try:
            with open(html_path, "r") as f:
                wandb_run.log({wandb_key: wandb.Html(f.read())})
        except Exception as e:
            print(f"[Profiler] WARNING: failed to log HTML: {e}")


def profile_memory_once(
    model,
    optimizer,
    loss_fn,
    loader,
    cfg,
    num_iters: int = 5,
    device: str = "cuda",
    run_name: str = "default_run",
    base_path: str | None = None,   # kept for API compatibility, unused now
    wandb_run=None,                 # <-- NEW optional (doesn't break callers)
):
    """
    Runs torch.profiler schedule and exports ONLY HTML memory timeline to W&B.
    Also logs forward/backward/iter GPU times to W&B.
    Nothing is kept locally. No .json.gz is created.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling.")
        return None

    model.train()
    device = torch.device(device)

    # Take one batch
    try:
        inputs, targets = next(iter(loader))
    except StopIteration:
        print("Loader empty, cannot profile.")
        return None

    inputs, targets = inputs.to(device), targets.to(device)

    # CUDA timing events (GPU-accurate)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end   = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end   = torch.cuda.Event(enable_timing=True)
    it_start  = torch.cuda.Event(enable_timing=True)
    it_end    = torch.cuda.Event(enable_timing=True)

    # schedule: (wait=1, warmup=1, active=num_iters) repeated twice
    total_iters = (1 + 1 + num_iters) * 2

    on_trace_ready = None
    if wandb_run is not None:
        on_trace_ready = lambda prof: _trace_handler_html_only(
            prof,
            wandb_run,
            wandb_key=f"memory/profiler_html"
        )

    print(f"\n[Profiler] Running memory profiling for '{run_name}' (HTML only, no gz)...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=num_iters, repeat=2),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=on_trace_ready,
    ) as prof:
        for _ in range(total_iters):
            prof.step()

            it_start.record()

            with record_function("forward"):
                fwd_start.record()
                outputs = model(pixel_values=inputs).logits
                fwd_end.record()

            with record_function("backward"):
                bwd_start.record()
                loss = loss_fn(outputs, targets)
                loss.backward()
                bwd_end.record()

            with record_function("optimizer"):
                if "layer" not in cfg.optimizer_name:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            it_end.record()
            torch.cuda.synchronize()

            fwd_ms = fwd_start.elapsed_time(fwd_end)
            bwd_ms = bwd_start.elapsed_time(bwd_end)
            it_ms  = it_start.elapsed_time(it_end)

            wandb_run.log({
                "time/forward_ms": fwd_ms,
                "time/backward_ms": bwd_ms,
                "time/iter_ms": it_ms,
            }, step=0)

    print(f"[Profiler] Finished '{run_name}'. Logged to W&B (if enabled).")
    return None



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
