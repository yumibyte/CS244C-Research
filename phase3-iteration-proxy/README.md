# Phase 3: Iteration-Level Performance Evaluation

## Goal

Show that bandwidth-optimal NCCL tuning does not necessarily minimize iteration time by building a lightweight training-step proxy and measuring end-to-end iteration time under different NCCL configurations.

## Planned Experiments

- Build a training-step proxy: compute phase → all-reduce phase → optional overlap via separate CUDA streams
- Measure total iteration time (not just communication bandwidth)
- Identify regimes where higher communication bandwidth leads to worse iteration time
- Show that less aggressive protocols can reduce interference and improve end-to-end performance

## Status

Not started.
