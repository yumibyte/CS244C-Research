# Phase 2: GPU Contention (A100 8-GPU)

## Overview

Same as the L40S Phase 2 contention setup, but for **8x A100** on Modal:
- Run a CUDA GPU stress benchmark (low / medium / high) on all 8 GPUs.
- Simultaneously run NCCL AllReduce and save results.
- Produce one result file per contention level for plotting and comparison.

## Directory Structure

```
a100-8gpu-new/
├── README.md              # This file
├── run_modal.py            # Modal app: stress + NCCL for low/medium/high (job: browser-networking-test)
├── gpu_stress_benchmark.cu # Same CUDA stress benchmark as L40S (cuBLAS SGEMM)
├── results/                # results_8gpu_allreduce_contended_{low,medium,high}.txt (+ optional plots)
├── plot_gpu_utilization.py # Plot GPU utilization from CSV (if you capture logs)
└── requirements-modal.txt  # modal (optional)
```

## How It Works (mirroring L40S)

1. **Stress benchmark** – `gpu_stress_benchmark.cu` runs matrix multiply (cuBLAS) at:
   - **low**: 1024×1024, sleep 10 ms between iterations  
   - **medium**: 4096×4096, sleep 2 ms  
   - **high**: 8192×8192, no sleep (max utilization)

2. **Contention run** – For each level we:
   - Start one stress process per GPU (8 processes, each bound to one GPU).
   - After 2 s, run `all_reduce_perf -b 8 -e 128M -f 2 -g 8`.
   - Save stdout to `results_8gpu_allreduce_contended_<level>.txt`.
   - Stop the stress processes.

3. **Output files** – Same format as L40S, so you can use the same bandwidth/latency plotting scripts (e.g. from phase1-baseline/a100-8gpu-new/scripts) on each file.

## Run on Modal

From repo root (with `nccl-tests` submodule checked out):

```bash
cd CS244C-Research/phase2-contention/a100-8gpu-new
modal run run_modal.py
```

- App name: **browser-networking-tests**  
- Job name: **browser-networking-test**  
- Writes `results_8gpu_allreduce_contended_low.txt`, `_medium.txt`, `_high.txt` to the Modal volume and to local `results/` via the entrypoint.

## Plotting

- **Bandwidth/latency**: Use the same scripts as phase1 A100, e.g.  
  `python plot_nccl_bw.py results/results_8gpu_allreduce_contended_high.txt`  
  and similarly for low/medium. You can compare baseline (no contention) vs contended.
- **GPU utilization**: If you capture utilization logs (e.g. via a separate run with `nvidia-smi` logging), use `plot_gpu_utilization.py` (edit `csv_file` or pass the path).

## Comparison to L40S

| Item        | L40S 2-GPU              | A100 8-GPU (this folder)     |
|------------|--------------------------|------------------------------|
| Script     | `run_nccl_with_contention.sh [low\|medium\|high]` | `modal run run_modal.py` (runs all three) |
| Stress     | 1 process (default GPU)  | 8 processes, 1 per GPU       |
| NCCL       | `all_reduce_perf ... -g 2` | `all_reduce_perf ... -g 8`   |
| Output     | `contention_results/results_2gpu_allreduce_contended_<level>.txt` | `results/results_8gpu_allreduce_contended_<level>.txt` |
