import os
import torch
import wandb
from datetime import datetime
from pathlib import Path
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from util.simple_memory_logger import measure_memory_breakdown

TIME_FMT = "%b_%d_%H_%M"

def _setup_dir(run_name, base_path):
    """Helper to create output directories."""
    path = Path(base_path) / "memory_profiles" if base_path else Path.cwd().parent / "memory_profiles"
    base_dir = path / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir), datetime.now().strftime(TIME_FMT)

def trace_handler(prof, output_dir, file_prefix):
    """
    Exports traces and PRINTS A BOTTLENECK REPORT.
    Now generates both CPU and CUDA sorted tables.
    """
    # 1. Export Standard Traces
    path_base = os.path.join(output_dir, file_prefix)
    prof.export_chrome_trace(f"{path_base}.json.gz")
    prof.export_memory_timeline(f"{path_base}.html", device="cuda:0")

    # 2. Generate Bottleneck Report
    report_path = f"{path_base}_bottlenecks.txt"
    
    # Get stats
    avgs = prof.key_averages()
    cpu_stats = avgs.table(sort_by="cpu_time_total", row_limit=20)
    
    # Prepare output string
    output_str = "=== Top Functions by CPU Time (Data Loading & Overhead) ===\n"
    output_str += cpu_stats + "\n\n"
    
    if torch.cuda.is_available():
        cuda_stats = avgs.table(sort_by="cuda_time_total", row_limit=20)
        output_str += "=== Top Functions by CUDA Time (GPU Compute) ===\n"
        output_str += cuda_stats + "\n"

    # Save to file
    with open(report_path, "w") as f:
        f.write(output_str)

    print(f"\n[Profiler] Report saved to {report_path}")
    print(output_str)

def profile_memory(model, optimizer, loss_fn, loader, num_iters, device, run_name, wandb_run=None):
    """
    Wrapper for HTML profiling + Simple Memory Logging + WandB logging.
    """
    # 1. Run Profiler (Generates HTML & Bottleneck Report)
    html_path = profile_memory_once(model, optimizer, loss_fn, loader, num_iters, device, run_name)

    # 2. Run Simple Breakdown
    mem_stats = measure_memory_breakdown(model, optimizer, loss_fn, loader, device)

    # 3. Log to W&B
    if wandb_run:
        if mem_stats:
            wandb_run.log({f"memory/{k}": v for k, v in mem_stats.items()})
        
        if html_path and os.path.exists(html_path):
            try:
                wandb_run.log({"memory/profiler_html": wandb.Html(open(html_path).read())})
            except Exception as e:
                print(f"[Profiler] WandB Log Error: {e}")

    exit(0)

def profile_memory_once(model, optimizer, loss_fn, loader, num_iters=5, device="cuda", run_name="default_run", base_path=None):
    """
    Runs torch.profiler with a schedule to generate timelines and bottleneck stats.
    Includes data loading time in the profile.
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA missing. Skipping profile_memory_once.")
        return None

    out_dir, prefix = _setup_dir(run_name, base_path)
    model.train().to(device)
    
    print(f"\n[Profiler] Profiling '{run_name}' (Warmup+Active cycles)...")

    # Iterator wrapper to handle recycling if loader is short
    loader_iter = iter(loader)

    # Schedule: Wait 1 -> Warmup 1 -> Active num_iters (Repeat 2x)
    sched = schedule(wait=1, warmup=1, active=num_iters, repeat=2)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=lambda p: trace_handler(p, out_dir, prefix)
    ) as prof:
        # Run enough steps to cover the schedule
        for _ in range((1 + 1 + num_iters) * 2):
            prof.step()
            
            # Profile Data Loading + Host-to-Device Transfer
            with record_function("data_loading"):
                try:
                    inputs, targets = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    inputs, targets = next(loader_iter)
                inputs, targets = inputs.to(device), targets.to(device)

            with record_function("forward"):
                outputs = model(pixel_values=inputs).logits
            with record_function("backward"):
                loss = loss_fn(outputs, targets)
                loss.backward()
            with record_function("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    html_path = os.path.join(out_dir, f"{prefix}.html")
    return html_path if os.path.exists(html_path) else None

def memory_snapshot_once(model, optimizer, loss_fn, loader, num_iters=5, device="cuda", run_name="default_run", base_path=None):
    """
    (Unused) Method A: Records low-level memory history snapshot.
    """
    if not torch.cuda.is_available(): return

    out_dir, prefix = _setup_dir(run_name, base_path)
    save_path = os.path.join(out_dir, f"{prefix}_snapshot.pickle")
    
    model.train().to(device)
    try:
        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)
    except: return

    print(f"[Snapshot] Recording {num_iters} iters...")
    try:
        torch.cuda.memory._record_memory_history(enabled=True)
        for _ in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(pixel_values=inputs).logits, targets).backward()
            optimizer.step()
        torch.cuda.memory._dump_snapshot(save_path)
        print(f"[Snapshot] Saved to {save_path}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=False)