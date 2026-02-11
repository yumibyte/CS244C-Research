# Phase 1: Baseline Characterization of NCCL Behavior

## Overview

This phase establishes a baseline understanding of NCCL's algorithm and protocol choices on an 8x NVIDIA A100-SXM4-80GB system with full NVLink mesh (NV12 topology). All experiments were run on `skampere1` using NCCL 2.24.3.

## System

- **GPUs**: 8x NVIDIA A100-SXM4-80GB (80 GB HBM2e each, 640 GB total)
- **Interconnect**: NVLink 3.0 via NVSwitch — full mesh, 12 links per GPU, 600 GB/s bidirectional per GPU
- **NUMA**: 2 nodes (GPUs 0,1,4,5 on NUMA 1; GPUs 2,3,6,7 on NUMA 0)
- **NCCL**: 2.24.3
- **CUDA Driver**: 12.8

## Directory Structure

```
phase1-baseline/
├── results/          # Raw nccl-tests benchmark outputs (.out files)
├── analysis/         # Written analysis reports (.md files)
├── scripts/          # Python scripts for parsing and comparing results
├── topology/         # NCCL topology/graph XML files and initialization logs
```

## Results

Raw benchmark outputs from `nccl-tests` all-reduce sweeps:

| File | Description |
|------|-------------|
| `baseline_auto.out` | Default NCCL (AUTO mode) — NCCL chooses algorithm and protocol |
| `ll128_forced.out` | LL128 protocol forced via `NCCL_PROTO=LL128` |
| `simple_forced.out` | Simple protocol forced via `NCCL_PROTO=Simple` |
| `ring_forced.out` | Ring algorithm forced via `NCCL_ALGO=Ring` |
| `tree_forced.out` | Tree algorithm forced via `NCCL_ALGO=Tree` |
| `allreduce_observe.out` | Dense sweep (1M–256M, factor 1.15) with NCCL INFO logging |
| `allreduce_trace.out` | Dense sweep with NCCL TRACE logging |
| `clean_3gpu_test.out` | 3-GPU test on verified-idle GPUs (4,5,6) to assess contention impact |

## Analysis Reports

| File | Description |
|------|-------------|
| `benchmark_results_summary.md` | Overview of all collective benchmarks (all-reduce, all-gather, reduce-scatter, broadcast, send/recv) |
| `nccl_analysis_report.md` | Algorithm and protocol transition analysis — identifies LL128→Simple transition zone at 4–16 MB |
| `protocol_comparison_report.md` | Head-to-head comparison of AUTO vs LL128 vs Simple protocols |
| `ring_vs_tree_comparison.md` | Ring vs Tree algorithm comparison — Tree dominates on NVLink mesh |
| `gpu_contention_analysis.md` | Impact of concurrent GPU workloads on NCCL performance (<3% effect for small/medium messages) |
| `cluster_topology_report.md` | Full system topology: NVLink, PCIe, NUMA, NCCL channels and communication graphs |

## Scripts

| File | Description |
|------|-------------|
| `analyze_transitions.py` | Parses benchmark output, identifies significant bandwidth transitions (>20% change) |
| `compare_protocols.py` | Compares AUTO vs LL128 vs Simple from forced-protocol benchmark outputs |

## Topology

| File | Description |
|------|-------------|
| `nccl_topo.xml` | NCCL hardware topology dump (NVLink connections, PCIe hierarchy) |
| `nccl_graph.xml` | NCCL communication graph (32 channels, Pattern 3 and 4, 40 GB/s per channel) |
| `nccl_observe_skampere1_rank%r.log` | NCCL initialization log (INFO level) |
| `nccl_trace_skampere1_rank%r.log` | NCCL trace log |

## Key Findings

1. **Peak all-reduce bandwidth**: ~850 GB/s (66% of 1,280 GB/s theoretical) on 8 GPUs
2. **Protocol transition zone**: 4–16 MB (LL128 → Simple)
3. **Tree dominates Ring**: 3–5x faster across all message sizes on NVLink mesh
4. **AUTO mode is effective**: NCCL's default tuning selects near-optimal configurations in isolation
5. **GPU contention has <3% impact** on communication bandwidth for small/medium messages

## Gaps (To Be Addressed)

- No 2-GPU baseline on this A100 system (only have 2-GPU L40S data from FarmShare)
- No sweep below 1 MB (proposal calls for KB-range messages)
- No explicit latency-focused analysis (only bandwidth)
