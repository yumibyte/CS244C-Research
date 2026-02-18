# Phase 3: Iteration-Level Performance Evaluation

## Goal

Show that bandwidth-optimal NCCL tuning does not necessarily minimize iteration time by building a lightweight training-step proxy and measuring end-to-end iteration time under different NCCL configurations.

## Planned Experiments

- Build a training-step proxy: compute phase → all-reduce phase → optional overlap via separate CUDA streams
- Measure total iteration time (not just communication bandwidth)
- Identify regimes where higher communication bandwidth leads to worse iteration time
- Show that less aggressive protocols can reduce interference and improve end-to-end performance

## Status

- **A100 8-GPU (Modal)**: Implemented in `a100-8gpu-new/` — training-step proxy (compute → all-reduce), run under NCCL configs AUTO / Simple / LL128; outputs iteration times to `results/`. Run with `modal run run_modal.py` from `a100-8gpu-new/`.
