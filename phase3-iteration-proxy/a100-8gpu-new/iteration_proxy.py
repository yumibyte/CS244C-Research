"""
Phase 3 training-step proxy: compute phase → all-reduce phase.
Measures end-to-end iteration time (not just communication bandwidth).
Run with: torchrun --nproc_per_node=8 iteration_proxy.py [--iters N] [--size S]
Output: iteration times (ms) to stdout and to a file (rank 0 only).
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 iteration proxy")
    p.add_argument("--iters", type=int, default=50, help="Number of timed iterations")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    p.add_argument("--size", type=int, default=2**20, help="All-reduce tensor size (elements, float32)")
    p.add_argument("--compute-mul", type=int, default=4096, help="Compute matmul size (NxN)")
    p.add_argument("--out", type=str, default="", help="Output file for iteration times (rank 0)")
    return p.parse_args()


def compute_phase(device: torch.device, n: int, dtype=torch.float32):
    """Simulate per-iteration compute: matmul on GPU."""
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    return c


def allreduce_phase(tensor: torch.Tensor):
    """All-reduce the tensor across all ranks."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(device)

    # Per-iteration buffer for all-reduce (same size on all ranks)
    elem = args.size
    grad = torch.randn(elem, device=device, dtype=torch.float32) / world_size

    # Warmup
    for _ in range(args.warmup):
        compute_phase(device, args.compute_mul)
        allreduce_phase(grad.clone())

    # Timed iterations
    times_ms = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compute_phase(device, args.compute_mul)
        allreduce_phase(grad.clone())
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    if rank == 0:
        out_lines = [f"{t:.3f}" for t in times_ms]
        summary = (
            f"config={os.environ.get('NCCL_PROTO', 'AUTO')} "
            f"mean_ms={sum(times_ms)/len(times_ms):.2f} "
            f"p95_ms={sorted(times_ms)[int(len(times_ms)*0.95)]:.2f} "
            f"iters={args.iters}"
        )
        print(summary, flush=True)
        for line in out_lines:
            print(line, flush=True)
        if args.out:
            with open(args.out, "w") as f:
                f.write("\n".join(out_lines) + "\n")
            print(f"Wrote {args.out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
