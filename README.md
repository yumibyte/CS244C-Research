# Workload-Aware Tuning of NCCL Collectives on Multi-GPU Systems

**CS244C Research Project**

## Summary

We study how NCCL's collective communication tuning interacts with concurrent GPU compute, and build a workload-aware tuner that improves iteration time — not just communication bandwidth.

NCCL optimizes communication in isolation. We show that under compute overlap, this objective breaks down, and that a workload-aware tuner can reduce iteration time even if raw bandwidth decreases.

## System Under Study

- **8x NVIDIA A100-SXM4-80GB** with full NVLink mesh (NV12, NVSwitch)
- **NCCL 2.24.3**, CUDA 12.8
- Single-node, 640 GB total GPU memory

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Baseline characterization of NCCL behavior (algorithms, protocols, transitions) | In progress |
| **Phase 2** | Controlled compute–communication overlap experiments | Not started |
| **Phase 3** | Iteration-level performance evaluation (training-step proxy) | Not started |
| **Phase 4** | Workload-aware NCCL tuner plugin (policy-level, no kernel changes) | Not started |
| **Phase 5** | Evaluation: default vs oracle vs workload-aware tuner | Not started |

## Repository Structure

```
CS244C-Research/
├── README.md                  # This file
├── background.md              # Background notes on NCCL tuners, algorithms, research questions
├── preliminary-research/      # Early exploratory work (FarmShare, L40S GPUs, 2-GPU baseline)
│   ├── analysis-1.md          # Initial bandwidth/latency analysis (L40S)
│   ├── results_2gpu_allreduce.txt  # Raw 2-GPU all-reduce results (L40S, FarmShare)
│   ├── plot_nccl_bw.py        # Bandwidth plotting script
│   ├── plot_nccl_latency.py   # Latency plotting script
│   ├── notes-for-farmshare.md # Setup notes for Stanford FarmShare
│   └── aws_nccl_baseline_plan.prompt.md  # AWS benchmarking plan (not executed)
├── phase1-baseline/           # Phase 1: NCCL baseline on 8x A100 NVLink system
│   ├── README.md              # Detailed description of all Phase 1 work
│   ├── results/               # Raw nccl-tests benchmark outputs
│   ├── analysis/              # Written analysis reports
│   ├── scripts/               # Python analysis/comparison scripts
│   └── topology/              # NCCL topology XML, communication graphs, logs
├── phase2-overlap/            # Phase 2: Compute–communication overlap (TODO)
├── phase3-iteration-proxy/    # Phase 3: Training-step proxy (TODO)
├── phase4-tuner/              # Phase 4: Workload-aware NCCL tuner plugin (TODO)
└── phase5-evaluation/         # Phase 5: Evaluation and comparison (TODO)
```

## Key Findings So Far (Phase 1)

1. **Peak all-reduce bandwidth**: ~850 GB/s on 8 A100s (66% of theoretical max)
2. **Protocol transition zone**: LL128 → Simple at 4–16 MB message sizes
3. **Tree algorithm dominates Ring**: 3–5x faster across all sizes on NVLink mesh
4. **AUTO mode is effective in isolation**: NCCL picks near-optimal configs when no compute is running
5. **GPU contention has <3% impact on comm bandwidth** — but this is uncontrolled; systematic overlap experiments (Phase 2) are needed

## Core Hypothesis

NCCL's tuning decisions optimize per-collective communication metrics without accounting for concurrent GPU compute. Under compute–communication overlap, the configuration that maximizes raw bandwidth may not minimize end-to-end iteration time. A workload-aware tuner can improve iteration-level performance by adapting algorithm and protocol selection to workload context.
