# Phase 4: Workload-Aware NCCL Tuner Plugin

## Goal

Implement a workload-aware tuning policy using NCCL's tuner plugin interface that adapts algorithm and protocol selection based on workload context, shifting the optimization objective from maximizing communication bandwidth to minimizing iteration time.

## Approach

- Use NCCL's external tuner plugin API (no kernel changes, no NCCL fork)
- Adapt algorithm (Ring vs Tree) and protocol (LL / LL128 / Simple) based on:
  - Message size
  - Presence or absence of compute overlap
- Policy-level tuning only

## How the tuner works

NCCL loads an external shared library that implements:

- **pluginInit**: called once per communicator (receives nRanks, nNodes, topology). Can load a config file (e.g. CSV).
- **pluginGetCollInfo**: called per collective with `collType`, `nBytes`, `numPipeOps`, `regBuff`. The plugin writes a **cost table** (per algorithm × protocol); NCCL picks the (algo, proto) with lowest cost. Setting one entry to `0.0` and leaving others higher makes that choice preferred.
- **pluginFinalize**: cleanup.

The API does **not** pass "compute overlap" or "iteration context" directly. Workload-awareness is achieved by:

1. **Size- and topology-based policy**: Use Phase 1 (bandwidth/latency) and Phase 3 (iteration time) results to choose algo/protocol by message size and rank count. For our 8-GPU iteration proxy, Phase 3 showed **Simple** gave the best mean iteration time; we encode that in a CSV.
2. **Optional overlap mode**: An env var (e.g. `NCCL_WORKLOAD_OVERLAP=1`) can be read in the plugin to switch to a different CSV or in-memory policy tuned for overlap.

## Reference

- NCCL tuner plugin example: https://github.com/NVIDIA/nccl/tree/master/ext-tuner/example
- Example README (config format, building, usage): https://github.com/NVIDIA/nccl/blob/master/ext-tuner/example/README.md

## Workload-aware config (8× A100, single-node)

The file `workload_aware_8gpu.conf` encodes a policy for **allreduce, 8 ranks, 1 node**, derived from our Phase 3 iteration-proxy results (Simple best mean iteration time for ~4 MB allreduce with compute):

- Prefer **tree + simple** for the size range used in the iteration proxy (e.g. ~1–8 MB), to minimize iteration time rather than raw bandwidth.
- Other sizes can mirror NCCL defaults or be extended as we add more data.

Use it with the official NCCL example plugin (build `libnccl-tuner-example.so`, then):

```bash
export NCCL_TUNER_PLUGIN=libnccl-tuner-example.so
export NCCL_TUNER_CONFIG_FILE=/path/to/phase4-tuner/workload_aware_8gpu.conf
# run your 8-GPU job, e.g. Phase 3 iteration proxy or nccl-tests
```

## RL bandit tuner plugin (online, workload-aware)

For a more ambitious design, `rl_bandit_tuner_plugin.c` implements a **multi-armed bandit** tuner that learns online which `(algorithm, protocol)` pair minimizes latency for a given workload context.

- **Key (state)** per bandit: `(collType, size_band, nNodes, nRanks)`.
- **Arms** per key: a small set of `(algo, proto)` candidates (currently `tree/simple`, `tree/ll128`, `ring/simple`).
- **Learning**: epsilon-greedy selection per key, updated from **latencies logged by the application**.

### Reward log format

The application (e.g. Phase 3 iteration proxy) should write **one line per completed collective / iteration**:

```text
collType,nBytes,nNodes,nRanks,latency_ms
```

Examples:

```text
allreduce,4194304,1,8,12.34
allreduce,4194304,1,8,11.98
```

- `collType`: `allreduce`, `broadcast`, etc.
- `nBytes`: message size in bytes for that collective.
- `nNodes`, `nRanks`: topology for the communicator.
- `latency_ms`: observed latency for that step (e.g. collective or iteration time).

By default, the plugin looks for:

- `NCCL_TUNER_REWARD_FILE` (env var), or
- `/tmp/nccl_tuner_rewards_<commId>.log` if unset.

It **attributes each latency to the last arm chosen for that key**, and updates running means accordingly.

### Using the RL tuner

1. Build a shared library from `rl_bandit_tuner_plugin.c` alongside NCCL (or in a separate project linked against NCCL headers) to produce something like `libnccl-tuner-rl-bandit.so`.
2. In your job launcher environment:

```bash
export NCCL_TUNER_PLUGIN=libnccl-tuner-rl-bandit.so
export NCCL_TUNER_EPS=0.1                       # optional, exploration rate in [0,1]
export NCCL_TUNER_REWARD_FILE=/tmp/nccl_rewards.log  # optional override
```

3. In your application, after each iteration / collective:
   - Measure the latency in milliseconds.
   - Append a line to the reward file in the format above.

Over time, the plugin will **shift probability mass toward the (algo, proto) combinations that minimize your observed latency** for each `(collType, size_band, nNodes, nRanks)` context, effectively performing **online workload-aware tuning**.

## Status

- **Design**: Documented; policy is size- and (optionally) env-based until NCCL exposes more workload context.
- **Config**: `workload_aware_8gpu.conf` added for 8× A100 single-node allreduce.
- **Plugin code**:
  - Static CSV-based tuner via NVIDIA's example plugin.
  - **RL bandit tuner** in `rl_bandit_tuner_plugin.c` for online workload-aware optimization.
- **Header discovery**: `a100-8gpu-new/get_nccl_tuner_info.py` — same Modal app (`browser-networking-tests`). Run `modal run get_nccl_tuner_info.py` from that directory to dump NCCL tuner headers from the torch/nccl runtime; output goes to `results/nccl_tuner_headers/` and `results/nccl_tuner_info.txt`. Use those to align `phase4-tuner/` headers for the plugin build.

