# A100 8-GPU Baseline (skampere1)

## Overview

Baseline characterization of NCCL all-reduce on an 8x A100 NVLink system. This is the primary system under study.

## System

- **GPUs**: 8x NVIDIA A100-SXM4-80GB (80 GB HBM2e each, 640 GB total)
- **Interconnect**: NVLink 3.0 via NVSwitch — full mesh (NV12), 12 links per GPU, 600 GB/s bidirectional per GPU
- **NUMA**: 2 nodes (GPUs 0,1,4,5 on NUMA 1; GPUs 2,3,6,7 on NUMA 0)
- **NCCL**: 2.24.3
- **CUDA Driver**: 12.8

## Results

Raw benchmark outputs from `nccl-tests` all-reduce sweeps:

| File | Description |
|------|-------------|
| `results/baseline_auto.out` | Default NCCL (AUTO mode) — NCCL chooses algorithm and protocol |
| `results/ll128_forced.out` | LL128 protocol forced via `NCCL_PROTO=LL128` |
| `results/simple_forced.out` | Simple protocol forced via `NCCL_PROTO=Simple` |
| `results/ring_forced.out` | Ring algorithm forced via `NCCL_ALGO=Ring` |
| `results/tree_forced.out` | Tree algorithm forced via `NCCL_ALGO=Tree` |
| `results/allreduce_observe.out` | Dense sweep (1M–256M, factor 1.15) with NCCL INFO logging |
| `results/clean_3gpu_test.out` | 3-GPU test on verified-idle GPUs (4,5,6) to assess contention impact |

## Analysis Reports

| File | Description |
|------|-------------|
| `analysis/benchmark_results_summary.md` | Overview of all collective benchmarks (all-reduce, all-gather, reduce-scatter, broadcast, send/recv) |
| `analysis/nccl_analysis_report.md` | Algorithm and protocol transition analysis — identifies LL128→Simple transition zone at 4–16 MB |
| `analysis/protocol_comparison_report.md` | Head-to-head comparison of AUTO vs LL128 vs Simple protocols |
| `analysis/ring_vs_tree_comparison.md` | Ring vs Tree algorithm comparison — Tree dominates on NVLink mesh |
| `analysis/gpu_contention_analysis.md` | Impact of concurrent GPU workloads on NCCL performance (<3% effect for small/medium messages) |

## Scripts

| File | Description |
|------|-------------|
| `scripts/analyze_transitions.py` | Parses benchmark output, identifies significant bandwidth transitions (>20% change) |
| `scripts/compare_protocols.py` | Compares AUTO vs LL128 vs Simple from forced-protocol benchmark outputs |

## Topology

| File | Description |
|------|-------------|
| `topology/cluster_topology_report.md` | Full system topology: NVLink, PCIe, NUMA, NCCL channels and communication graphs |
| `topology/nccl_topo.xml` | NCCL hardware topology dump (NVLink connections, PCIe hierarchy) |
| `topology/nccl_graph.xml` | NCCL communication graph (32 channels, Pattern 3 and 4, 40 GB/s per channel) |

## Key Results

- **Peak all-reduce bandwidth**: ~850 GB/s (66% of 1,280 GB/s theoretical)
- **Protocol transition zone**: 4–16 MB (LL128 → Simple)
- **Tree dominates Ring**: 3–5x faster across all message sizes
- **AUTO mode**: Near-optimal in isolation
- **Per-GPU efficiency**: 106 GB/s (8 GPU) vs 277 GB/s (3 GPU) — expected non-linear scaling
