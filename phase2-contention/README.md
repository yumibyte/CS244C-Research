# Phase 2: GPU Contention and NCCL Performance

## Overview

This phase investigates how concurrent GPU compute (contention) affects NCCL AllReduce performance:
- Run a CUDA-based GPU stress benchmark at different contention levels (low, medium, high).
- Simultaneously run NCCL AllReduce tests to measure communication performance under contention.
- Monitor and plot GPU utilization during these experiments.

## Directory Structure

```
phase2-contention/
├── README.md           # This file
├── l40s-2gpu/          # L40S 2-GPU (FarmShare): scripts, results, utilization logs, plots
│   ├── run_nccl_with_contention.sh
│   ├── check_gpu_utilization.sh
│   ├── gpu_stress_benchmark.cu
│   ├── plot_gpu_utilization.py
│   ├── contention_results/     # results_2gpu_allreduce_contended_<level>.txt
│   ├── gpu_utilization_logs/   # CSV logs and utilization plots
│   ├── bandwidth_graphs/
│   ├── latency_graphs/
│   ├── analysis-2.md
│   └── contention_analysis.prompt.md
└── a100-8gpu-new/      # A100 8-GPU (Modal): same flow via Modal, results + bandwidth/latency plots
    ├── run_modal.py
    ├── gpu_stress_benchmark.cu
    ├── results/        # results_8gpu_allreduce_contended_<level>.txt + plots
    └── ...
```

## L40S 2-GPU (FarmShare)

From `l40s-2gpu/`:

- **Run contention:** `bash run_nccl_with_contention.sh [low|medium|high]`  
  Saves to `contention_results/results_2gpu_allreduce_contended_<level>.txt`.
- **Monitor utilization:** `bash check_gpu_utilization.sh [low|medium|high]`  
  Logs to `gpu_utilization_logs/gpu_utilization_log_<level>.csv`.
- **Plot utilization:** `python plot_gpu_utilization.py`  
  (Edit `csv_file` in the script or pass path; saves PNG in same dir.)

NCCL tests use `../../nccl-tests/build/all_reduce_perf` (repo nccl-tests).

## A100 8-GPU (Modal)

From `a100-8gpu-new/`:

- **Run all levels on Modal:** `modal run run_modal.py`  
  Writes `results/results_8gpu_allreduce_contended_{low,medium,high}.txt` and you can generate bandwidth/latency plots with the phase1 plotting scripts.

See `a100-8gpu-new/README.md` for details.

## Notes

- Run and plot low/medium/high for both systems to compare baseline vs contended performance.
- Analysis and prompts live under `l40s-2gpu/` for the L40S run.
