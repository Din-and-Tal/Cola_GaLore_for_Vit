import torch
import re
from torch.profiler import profile, ProfilerActivity
import wandb


def benchmark_training_memory(model, loss_func, optimizer, input_data, target):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    input_data = input_data.to(device)
    target = target.to(device)

    # Reset peak stats to track only this specific run
    torch.cuda.reset_peak_memory_stats()

    # 1. Forward
    output = model(pixel_values=input_data).logits
    mem_forward = torch.cuda.max_memory_allocated()
    mem_forward_res = torch.cuda.max_memory_reserved()

    # 2. Backward
    loss = loss_func(output, target)
    loss.backward()
    mem_backward = torch.cuda.max_memory_allocated()
    mem_backward_res = torch.cuda.max_memory_reserved()

    # 3. Optimizer Step
    optimizer.step()
    optimizer.zero_grad()
    mem_step = torch.cuda.max_memory_allocated()
    mem_step_res = torch.cuda.max_memory_reserved()

    print(
        f"Peak Memory (Forward):  {mem_forward / 1024**2:.2f} MB | with Reserved: {(mem_forward_res+mem_forward) / 1024**2:.2f} MB"
    )
    print(
        f"Peak Memory (Backward): {mem_backward / 1024**2:.2f} MB | with Reserved: {(mem_backward_res+mem_backward) / 1024**2:.2f} MB"
    )
    print(
        f"Peak Memory (Step):     {mem_step / 1024**2:.2f} MB | with Reserved: {(mem_step_res+mem_step) / 1024**2:.2f} MB"
    )


def run_and_log_profiler(
    model,
    images,
    labels,
    loss_func,
    device,
    tag: str = "profiler",
    step: int | None = None,
    top_k: int = 10,
):
    """
    Runs the PyTorch profiler on a single forward/backward step AND logs:
      1. Full profiler table as a wandb.Table  -> {tag}/full_table
      2. Top-K memory-heavy ops as a wandb.Table -> {tag}/top10_mem_table
    """

    images = images.to(device)
    labels = labels.to(device)

    print(f"\n🔍 Running profiler ({tag})...\n")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        outputs = model(images)
        loss = loss_func(outputs.logits, labels)
        loss.backward()

    avg = prof.key_averages()

    # Try to sort by CUDA mem; else fall back to CPU mem
    try:
        table_obj = avg.table(sort_by="self_cuda_memory_usage", row_limit=300)
    except Exception:
        table_obj = avg.table(sort_by="self_cpu_memory_usage", row_limit=300)

    table_str = str(table_obj)

    # ---- print full table ----
    print("\n================ PyTorch Profiler Summary ================\n")
    print(table_str)
    print("\n==========================================================\n")

    # ---- 1) log full table to W&B ----
    full_table = wandb.Table(columns=["profile"], data=[[table_str]])
    wandb.log({f"{tag}/full_table": full_table}, step=step)

    # ---- 2) log top-K memory ops table ----
    top_rows = _extract_topk_from_table_str(table_str, top_k=top_k)

    if top_rows:
        data = []
        for name, mem_bytes, mem_str, calls_str in top_rows:
            data.append([name, mem_str, mem_bytes / (1024**2), int(calls_str)])

        top10_table = wandb.Table(
            columns=["op", "self_cuda_mem_str", "self_cuda_mem_MB", "calls"],
            data=data,
        )
        wandb.log({f"{tag}/top10_mem_table": top10_table}, step=step)

    print(f"✅ Profiler logged under `{tag}` (step={step})")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _parse_mem_str_to_bytes(s: str) -> int:
    s = s.strip()
    if not s or s == "0" or s == "0 B":
        return 0
    try:
        num_str, unit = s.split()
        val = float(num_str)
    except ValueError:
        return 0

    unit = unit.upper()
    if unit == "B":
        mult = 1
    elif unit == "KB":
        mult = 1024
    elif unit == "MB":
        mult = 1024**2
    elif unit == "GB":
        mult = 1024**3
    else:
        mult = 1
    return int(val * mult)


def _extract_topk_from_table_str(table_str: str, top_k: int = 10):
    """
    Parse the pretty-printed profiler table string and extract top-k rows
    by 'Self CUDA Mem' (second from last column).
    """
    lines = [ln for ln in table_str.splitlines() if ln.strip()]

    rows = []
    in_data = False

    for ln in lines:
        stripped = ln.strip()

        # Start after header line that begins with "Name"
        if stripped.startswith("Name"):
            in_data = True
            continue

        # Skip separators and stop at summary
        if not in_data:
            continue
        if stripped.startswith("-"):
            continue
        if stripped.startswith("Self CPU time total"):
            break

        # Split on 2+ spaces to get columns
        cols = re.split(r"\s{2,}", stripped)
        if len(cols) < 11:
            continue

        # Columns (based on PyTorch table format)
        # [0]=Name, ..., [-3]=CUDA Mem, [-2]=Self CUDA Mem, [-1]=# of Calls
        name = cols[0]
        self_cuda_mem_str = cols[-2]
        calls_str = cols[-1]

        mem_bytes = _parse_mem_str_to_bytes(self_cuda_mem_str)
        rows.append((name, mem_bytes, self_cuda_mem_str, calls_str))

    # Already sorted by self_cuda_memory_usage if sort_by worked,
    # but sort again just in case.
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]
