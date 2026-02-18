# Phase 4: Workload-Aware NCCL Tuner Plugin

## Goal

Implement a workload-aware tuning policy using NCCL's tuner plugin interface that adapts algorithm and protocol selection based on workload context, shifting the optimization objective from maximizing communication bandwidth to minimizing iteration time.

## Approach

- Use NCCL's external tuner plugin API (no kernel changes, no NCCL fork)
- Adapt algorithm (Ring vs Tree) and protocol (LL / LL128 / Simple) based on:
  - Message size
  - Presence or absence of compute overlap
- Policy-level tuning only

## Reference

- NCCL tuner plugin example: https://github.com/NVIDIA/nccl/tree/master/ext-tuner/example

## Status

Not started.
