# Phase 2: Controlled Compute–Communication Overlap

## Goal

Demonstrate that NCCL's performance and tuning behavior are sensitive to concurrent GPU compute by running NCCL all-reduce concurrently with controlled CUDA compute kernels.

## Planned Experiments

- Execute NCCL all-reduce while a CUDA compute kernel runs on the same GPUs
- Vary overlap intensity: low, medium, high contention
- Measure how communication bandwidth and latency change under increasing contention
- Compare protocol behavior (LL128 vs Simple) under each contention level

## Status

Not started.
