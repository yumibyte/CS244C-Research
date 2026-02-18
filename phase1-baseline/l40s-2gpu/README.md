# L40S 2-GPU Baseline (Stanford FarmShare)

## Overview

Early baseline characterization of NCCL all-reduce on a 2-GPU system at Stanford FarmShare.

## System

- **GPUs**: 2x NVIDIA L40S (46 GB each)
- **Interconnect**: PCIe (NODE topology — no NVLink)
- **Host**: oat-05 (FarmShare)
- **NCCL**: 2.29.02
- **CUDA**: 13.0 (Driver 580.126.09)

## Contents

| Path | Description |
|------|-------------|
| `results/results_2gpu_allreduce.txt` | Raw nccl-tests all-reduce output (8B–128MB, factor 2) |
| `analysis/analysis-1.md` | Bandwidth and latency analysis |
| `scripts/plot_nccl_bw.py` | Bandwidth plotting script |
| `scripts/plot_nccl_latency.py` | Latency plotting script |
| `notes-for-farmshare.md` | Environment setup notes (modules, micromamba, build commands) |

## Key Results

- **Peak bandwidth**: ~15.8 GB/s (PCIe-limited, as expected)
- **Average bus bandwidth**: ~6.03 GB/s across all message sizes
- Smooth bandwidth scaling, no anomalies
- All correctness checks passed

## Comparison to A100 System

| Metric | L40S 2-GPU (FarmShare) | A100 8-GPU (skampere1) |
|--------|----------------------|----------------------|
| Peak BW | 15.8 GB/s | 850 GB/s |
| Interconnect | PCIe (NODE) | NVLink (NV12 mesh) |
| Speedup | 1x | ~54x |

This data is useful as a PCIe baseline contrast to the NVLink results but is not the primary system under study.
