# Phase 3: Iteration-Level Performance (A100 8-GPU)

## Goal

Show that **bandwidth-optimal NCCL tuning does not necessarily minimize iteration time** by running a lightweight training-step proxy and measuring **end-to-end iteration time** under different NCCL configurations (from the [Phase 3 README](../README.md)).

## Training-Step Proxy

- **Compute phase**: Each GPU runs a short compute kernel (PyTorch matmul / backward) to simulate per-iteration work.
- **All-reduce phase**: NCCL AllReduce on a gradient-sized tensor across 8 GPUs (PyTorch `dist.all_reduce`).
- **Iteration time**: Wall-clock for one iteration = compute + all-reduce (sequential; optional overlap via streams can be added later).
- We run many iterations per config and record mean/p95 iteration time.

## NCCL Configurations

We compare:

| Config   | Env / notes              |
|----------|--------------------------|
| **AUTO** | Default (NCCL chooses).  |
| **Simple** | `NCCL_PROTO=Simple`   |
| **LL128**  | `NCCL_PROTO=LL128`    |

Goal: identify regimes where **higher communication bandwidth** (e.g. AUTO) gives **worse iteration time** than a less aggressive protocol (e.g. Simple), due to interference with compute or other effects.

## Directory Structure

```
a100-8gpu-new/
├── README.md           # This file
├── run_modal.py        # Modal app: runs proxy under each NCCL config (job: browser-networking-test)
├── iteration_proxy.py  # PyTorch distributed proxy: compute → allreduce, reports iteration times
├── results/            # iteration_times_<config>.txt and summary
└── requirements-modal.txt
```

## Run on Modal

From repo root:

```bash
cd CS244C-Research/phase3-iteration-proxy/a100-8gpu-new
modal run run_modal.py
```

- App: **browser-networking-tests**
- Job: **browser-networking-test**
- Writes `results/iteration_times_auto.txt`, `iteration_times_simple.txt`, `iteration_times_ll128.txt` and a short summary.

## Output Format

Each `iteration_times_<config>.txt` contains one iteration time (ms) per line. Use for histograms or comparison (e.g. mean/p95) to show bandwidth vs iteration-time trade-offs.
