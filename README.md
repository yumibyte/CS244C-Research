# Workload-Aware Tuning of NCCL Collectives on Multi-GPU Systems

**CS244C Research Project**

## Summary

We study how NCCL's collective communication tuning interacts with concurrent GPU compute, and build a workload-aware tuner that improves iteration time — not just communication bandwidth.

NCCL optimizes communication in isolation. We show that under compute overlap, this objective breaks down, and that a workload-aware tuner can reduce iteration time even when raw bandwidth is not maximized.

## System Under Study

- **8× NVIDIA A100-SXM4-80GB** with full NVLink mesh (NV12, NVSwitch)
- **NCCL** (via PyTorch / nccl-tests), **CUDA 12.x**
- Single-node, 640 GB total GPU memory
- A100 8-GPU experiments run on **Modal** (cloud); L40S 2-GPU work on Stanford FarmShare

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Baseline characterization of NCCL (algorithms, protocols, transitions) on 8× A100 and 2× L40S | Done |
| **Phase 2** | GPU contention: NCCL AllReduce under low/medium/high compute stress (8× A100 Modal, 2× L40S) | Done |
| **Phase 3** | Iteration-level proxy (compute → all-reduce); measure iteration time under AUTO / Simple / LL128 | Done |
| **Phase 4** | Workload-aware NCCL tuner: CSV policy + online RL bandit plugin | Implemented |
| **Phase 5** | Evaluation: default vs oracle vs workload-aware tuner | Not started |

## Repository Structure

```
CS244C-Research/
├── README.md                  # This file
├── background.md              # NCCL tuners, algorithms, research questions
├── nccl-tests/                # Submodule: NCCL benchmark suite (phase1, phase2)
├── preliminary-research/      # Early work (FarmShare, L40S, 2-GPU baseline)
│   ├── analysis-1.md
│   ├── results_2gpu_allreduce.txt
│   ├── plot_nccl_bw.py, plot_nccl_latency.py
│   ├── notes-for-farmshare.md
│   └── aws_nccl_baseline_plan.prompt.md
├── phase1-baseline/           # Phase 1: NCCL baseline
│   ├── README.md
│   ├── a100-8gpu/            # 8× A100 (skampere1): results, scripts, topology
│   ├── a100-8gpu-new/        # 8× A100 (Modal): run_modal.py, results/, scripts/
│   └── l40s-2gpu/            # 2× L40S (FarmShare): results, scripts
├── phase2-contention/         # Phase 2: NCCL under GPU contention
│   ├── README.md
│   ├── a100-8gpu-new/        # Modal: run_modal.py, gpu_stress_benchmark.cu, results/
│   └── l40s-2gpu/            # FarmShare: run_nccl_with_contention.sh, plots, analysis
├── phase3-iteration-proxy/    # Phase 3: Training-step proxy
│   ├── README.md
│   └── a100-8gpu-new/        # iteration_proxy.py, run_modal.py, results/, analysis/plots
├── phase4-tuner/              # Phase 4: Workload-aware tuner
│   ├── README.md
│   ├── workload_aware_8gpu.conf   # CSV policy (tree+simple for 8× A100)
│   ├── rl_bandit_tuner_plugin.c   # Online bandit plugin (epsilon-greedy)
│   ├── tuner.h, common.h, err.h   # Tuner API headers for plugin build
│   └── a100-8gpu-new/        # run_modal.py (RL tuner), get_nccl_tuner_info.py
└── phase5-evaluation/         # Phase 5: Comparison (default / oracle / workload-aware)
    └── README.md
```

## Key Findings

### Phase 1 (Baseline)

1. **Peak all-reduce bandwidth**: ~850 GB/s on 8 A100s (66% of theoretical max); ~15.8 GB/s on 2 L40S (PCIe-limited).
2. **Protocol transition**: LL128 → Simple at 4–16 MB message sizes on A100.
3. **Tree dominates Ring**: 3–5× faster across sizes on NVLink mesh.
4. **AUTO mode** is effective when no compute is running.
5. **GPU contention** had &lt;3% impact on comm bandwidth in early tests; Phase 2 systematized this.

### Phase 2 (Contention)

- AllReduce bandwidth and latency were measured under **low / medium / high** GPU stress (8 processes, one per GPU). Bandwidth and latency plots per level are in `phase2-contention/a100-8gpu-new/results/`.

### Phase 3 (Iteration proxy)

- **Simple protocol gave the best mean iteration time** for the proxy (compute + ~4 MB all-reduce). AUTO (bandwidth-optimal) did not minimize iteration time — supports workload-aware tuning in Phase 4.

### Phase 4 (Tuner)

- **Static policy**: `workload_aware_8gpu.conf` encodes tree+simple for 8× A100 single-node allreduce (usable with NVIDIA’s example tuner plugin).
- **Online RL bandit**: `rl_bandit_tuner_plugin.c` implements epsilon-greedy selection over (algo, proto) per (collType, size_band, nNodes, nRanks), learning from application-reported latencies. Requires NCCL tuner headers aligned with the runtime (use `get_nccl_tuner_info.py` to dump headers from the Modal environment).

## Core Hypothesis

NCCL's tuning optimizes per-collective communication and does not account for concurrent GPU compute. Under compute–communication overlap, the configuration that maximizes raw bandwidth may not minimize end-to-end iteration time. A **workload-aware tuner** can improve iteration-level performance by adapting algorithm and protocol selection to workload context (message size, topology, and optionally overlap mode).
