import os
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from datetime import datetime
import socket

TIME_FORMAT_STR = "%b_%d_%H_%M_%S"

def profile_memory(model,optimizer,loss_fn,loader,num_iters,device,run_name):
    """
    Wrapper to run both memory profiling methods (A and B).
    """
    memory_snapshot_once(
        model,
        optimizer,
        loss_fn,
        loader,
        num_iters=num_iters,
        device=device,
        run_name=run_name,
    )
    profile_memory_once(
        model,
        optimizer,
        loss_fn,
        loader,
        num_iters=num_iters,
        device=device,
        run_name=run_name,
    )

def _make_run_dir(run_name: str, base_path: str | None = None) -> tuple[str, str]:
    """
    Create directory for a run and return (base_dir, file_prefix).
    base_dir:  <base_path>/memory_profiles/<run_name>
    file_prefix: <hostname>_<timestamp>
    """
    if base_path is None:
        # default: one level above current cwd, in "memory_profiles"
        base_path = os.path.join(os.path.dirname(os.getcwd()), "memory_profiles")
    else:
        base_path = os.path.join(base_path, "memory_profiles")

    base_dir = os.path.join(base_path, run_name)
    os.makedirs(base_dir, exist_ok=True)


    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{timestamp}"

    return base_dir, file_prefix


# =========================
# Method B: Profiler HTML
# =========================

def trace_handler(prof: torch.profiler.profile, output_dir: str, file_prefix: str):
    """Handles trace export (JSON + HTML memory timeline)."""
    json_path = os.path.join(output_dir, f"{file_prefix}.json.gz")
    html_path = os.path.join(output_dir, f"{file_prefix}.html")

    prof.export_chrome_trace(json_path)
    prof.export_memory_timeline(html_path, device="cuda:0")

    print(f"[Profiler] Trace saved:\n  {json_path}\n  {html_path}\n")


def profile_memory_once(
    model,
    optimizer,
    loss_fn,
    loader,
    num_iters: int = 5,
    device: str = "cuda",
    run_name: str = "default_run",
    base_path: str | None = None,
):
    """
    Method B: run a few dummy iterations with torch.profiler to generate
    Chrome trace + HTML memory timeline for the current run.

    Saves under: <base_path>/memory_profiles/<run_name>/<prefix>.html
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling (Method B).")
        return

    base_dir, file_prefix = _make_run_dir(run_name, base_path)

    model.train()
    device = torch.device(device)

    # Take one batch
    try:
        inputs, targets = next(iter(loader))
    except StopIteration:
        print("Loader empty, cannot profile (Method B).")
        return

    inputs, targets = inputs.to(device), targets.to(device)

    print(f"\n[Profiler] Running memory profiling (Method B) for '{run_name}'...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=num_iters, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=lambda prof: trace_handler(prof, base_dir, file_prefix),
    ) as prof:
        for _ in range(num_iters):
            prof.step()
            with record_function("forward"):
                outputs = model(pixel_values=inputs).logits
            with record_function("backward"):
                loss = loss_fn(outputs, targets)
                loss.backward()
            with record_function("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    print(f"[Profiler] Finished Method B. Files saved under: {base_dir}\n")


# =========================
# Method A: Snapshot pickle
# =========================

def memory_snapshot_once(
    model,
    optimizer,
    loss_fn,
    loader,
    num_iters: int = 5,
    device: str = "cuda",
    run_name: str = "default_run",
    base_path: str | None = None,
):
    """
    Method A: use torch.cuda.memory._record_memory_history to record
    a low-level memory snapshot and dump it as a .pickle file.

    Saves under: <base_path>/memory_profiles/<run_name>/<prefix>_snapshot.pickle
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory snapshot (Method A).")
        return

    base_dir, file_prefix = _make_run_dir(run_name, base_path)
    snapshot_path = os.path.join(base_dir, f"{file_prefix}_snapshot.pickle")

    model.train()
    device = torch.device(device)

    # Take one batch
    try:
        inputs, targets = next(iter(loader))
    except StopIteration:
        print("Loader empty, cannot profile (Method A).")
        return

    inputs, targets = inputs.to(device), targets.to(device)

    print(f"\n[Snapshot] Recording GPU memory history (Method A) for '{run_name}'...")
    try:
        # Start recording low-level memory events
        torch.cuda.memory._record_memory_history(
            enabled=True
        )

        for _ in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=inputs).logits
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        # Dump snapshot to pickle
        torch.cuda.memory._dump_snapshot(snapshot_path)
        print(f"[Snapshot] Dumped memory snapshot to: {snapshot_path}")

    finally:
        # Disable history recording
        torch.cuda.memory._record_memory_history(enabled=False)

    print(f"[Snapshot] Finished Method A. Files saved under: {base_dir}\n")
