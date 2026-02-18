# A100 8-GPU Baseline (Modal)

## Overview

Phase 1 baseline NCCL all-reduce on **8x A100** via **Modal**, mirroring the L40S 2-GPU flow: run `nccl-tests` all-reduce, save a results `.txt` file, then use the bandwidth plotting script.

## System (Modal)

- **GPUs**: 8x NVIDIA A100 (Modal cloud)
- **Goal**: Same experiment as phase1-baseline (8-GPU all-reduce, 8B–128MB, factor 2), producing a txt file suitable for `plot_nccl_bw.py`.

## Directory Structure

```
a100-8gpu-new/
├── README.md           # This file
├── results/             # Raw nccl-tests output (results_8gpu_allreduce.txt)
├── scripts/             # plot_nccl_bw.py, plot_nccl_latency.py (from L40S, title updated)
├── run_modal.py         # Modal app: build nccl-tests, run benchmark, job name "browser-prediction-test"
└── requirements-modal.txt  # modal (optional, for local run)
```

## Usage

### 1. Run the benchmark on Modal

From the repo root (so `nccl-tests` submodule is available):

```bash
cd CS244C-Research/phase1-baseline/a100-8gpu-new
modal run run_modal.py
```

- Job name in Modal dashboard: **browser-prediction-test**.
- The run builds nccl-tests, runs `all_reduce_perf -b 8 -e 128M -f 2 -g 8`, and writes the output to the Modal Volume and prints it. Copy the printed output into `results/results_8gpu_allreduce.txt` if you want it locally.

### 2. Create the bandwidth graph

Using the same format as L40S (see `phase1-baseline/l40s-2gpu/run-baseline-tutorial.md`):

```bash
cd scripts
python plot_nccl_bw.py ../results/results_8gpu_allreduce.txt
```

Plots are written to `scripts/bandwidth_graphs/`.

## Comparison to L40S

| Step              | L40S 2-GPU (FarmShare)     | A100 8-GPU (Modal)        |
|-------------------|----------------------------|----------------------------|
| Run               | `./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2` | Same flags with `-g 8` |
| Output file       | `results_2gpu_allreduce.txt` | `results_8gpu_allreduce.txt` |
| Plot script       | `plot_nccl_bw.py`          | Same script, title updated for 8 A100 |
