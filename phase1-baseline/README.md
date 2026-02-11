# Phase 1: Baseline Characterization of NCCL Behavior

## Overview

This phase establishes a baseline understanding of NCCL's algorithm and protocol choices across two systems. The primary system is an 8x A100 NVLink node; a 2x L40S PCIe system provides a contrasting baseline.

## Directory Structure

```
phase1-baseline/
├── a100-8gpu/        # Primary system: 8x A100 with NVLink mesh (skampere1)
│   ├── results/      # Raw nccl-tests benchmark outputs (.out files)
│   ├── analysis/     # Written analysis reports (.md files)
│   ├── scripts/      # Python scripts for parsing and comparing results
│   └── topology/     # NCCL topology/graph XML files and logs
├── l40s-2gpu/        # Secondary system: 2x L40S with PCIe (FarmShare oat-05)
│   ├── results/      # Raw nccl-tests output
│   ├── analysis/     # Bandwidth/latency analysis
│   ├── scripts/      # Plotting scripts
│   └── notes-for-farmshare.md  # Environment setup notes
```

## Systems

### A100 8-GPU (Primary — skampere1)

- **GPUs**: 8x NVIDIA A100-SXM4-80GB (80 GB HBM2e each, 640 GB total)
- **Interconnect**: NVLink 3.0 via NVSwitch — full mesh (NV12), 12 links per GPU, 600 GB/s bidirectional per GPU
- **NUMA**: 2 nodes (GPUs 0,1,4,5 on NUMA 1; GPUs 2,3,6,7 on NUMA 0)
- **NCCL**: 2.24.3, **CUDA Driver**: 12.8

### L40S 2-GPU (Secondary — FarmShare oat-05)

- **GPUs**: 2x NVIDIA L40S (46 GB each)
- **Interconnect**: PCIe (NODE topology — no NVLink)
- **NCCL**: 2.29.02, **CUDA**: 13.0

See each subdirectory's README for detailed file listings and results.

## Key Findings

1. **Peak all-reduce bandwidth**: ~850 GB/s on 8 A100s (66% of theoretical max); ~15.8 GB/s on 2 L40S (PCIe-limited)
2. **Protocol transition zone**: LL128 → Simple at 4–16 MB on A100 system
3. **Tree dominates Ring**: 3–5x faster across all message sizes on NVLink mesh
4. **AUTO mode is effective in isolation**: NCCL picks near-optimal configs when no compute is running
5. **GPU contention has <3% impact** on communication bandwidth for small/medium messages (uncontrolled; systematic overlap experiments in Phase 2)
6. **NVLink vs PCIe**: ~54x bandwidth difference highlights the importance of interconnect topology for NCCL tuning behavior
