# Phase 5: Evaluation and Comparison

## Goal

Evaluate the workload-aware tuner against baselines and demonstrate that it improves iteration time without modifying NCCL's communication mechanisms.

## Baselines

1. **Default NCCL** — AUTO mode (no env vars)
2. **Oracle** — best forced algorithm/protocol per message size regime (determined from Phase 1–3 data)
3. **Workload-aware tuner** — our Phase 4 plugin

## Metrics

- Communication bandwidth
- Latency
- **Iteration time** (primary metric)
- Compute slowdown due to communication interference (optional)

## Status

Not started.
