# Phase 2: L40S 2-GPU Contention (FarmShare)

## Overview

GPU contention experiments on 2x L40S (FarmShare): run GPU stress at low/medium/high while measuring NCCL AllReduce.

## Contents

| Path | Description |
|------|-------------|
| `run_nccl_with_contention.sh` | Run stress + NCCL; usage: `bash run_nccl_with_contention.sh [low\|medium\|high]` |
| `check_gpu_utilization.sh` | Run stress and log GPU utilization for 30s to CSV |
| `plot_gpu_utilization.py` | Plot utilization from CSV (edit `csv_file` or pass path) |
| `gpu_stress_benchmark.cu` | CUDA/cuBLAS stress benchmark source |
| `contention_results/` | NCCL output: `results_2gpu_allreduce_contended_<level>.txt` |
| `gpu_utilization_logs/` | Utilization CSVs and optional utilization plots |
| `bandwidth_graphs/` | Bandwidth plots from contended runs |
| `latency_graphs/` | Latency plots from contended runs |
| `analysis-2.md` | Analysis notes |
| `contention_analysis.prompt.md` | Prompt/notes for contention analysis |

## Quick run

```bash
# From this directory (l40s-2gpu)
bash run_nccl_with_contention.sh high
# Results in contention_results/results_2gpu_allreduce_contended_high.txt
```

NCCL binary: `../../nccl-tests/build/all_reduce_perf` (build nccl-tests from repo root if needed).
