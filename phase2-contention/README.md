# Phase 2: GPU Contention and NCCL Performance

## Overview

This phase investigates how concurrent GPU compute (contention) affects NCCL AllReduce performance:
- Run a CUDA-based GPU stress benchmark at different contention levels (low, medium, high).
- Simultaneously run NCCL AllReduce tests to measure communication performance under contention.
- Monitor and plot GPU utilization during these experiments.



## L40S Multi-GPU (FarmShare)

Ensure you are in the `phase2-contention/scripts/` directory:

- **Run contention:**
  ```bash
  bash scripts/run_nccl_with_contention.sh <low|medium|high> <num_gpus>
  ```
  - `<low|medium|high>`: Contention level for the stress benchmark
  - `<num_gpus>`: Number of GPUs to use (e.g., 2, 4)
  - Output is saved to `contention_results/results_<gpu_tag>-<num_gpus}gpu_allreduce_contended_<level>.txt` (where `<gpu_tag>` is the name of the GPU, e.g., `l40s`)

- **Monitor utilization:**
  ```bash
  bash scripts/check_gpu_utilization.sh <low|medium|high>
  ```
  - Logs to `gpu_utilization_logs/gpu_utilization_log_<level>.csv`

- **Plot utilization:**
  ```bash
  python scripts/plot_gpu_utilization.py <csv_file>
  ```
  - Saves PNG in the same directory as the CSV file.

- **Plot bandwidth and latency:**
  - To plot bandwidth and latency, run the output `.txt` file from the contention experiment through the plotting scripts in `phase1-baseline/scripts/`

## A100 8-GPU (Modal)

From `a100-8gpu-new/`:

- **Run all levels on Modal:** `modal run run_modal.py`  
  Writes `results/results_8gpu_allreduce_contended_{low,medium,high}.txt` and you can generate bandwidth/latency plots with the phase1 plotting scripts.

See `a100-8gpu-new/README.md` for details.

## Notes

- Run and plot low/medium/high for both systems to compare baseline vs contended performance.
- Analysis and prompts live under `l40s-2gpu/` for the L40S run.
