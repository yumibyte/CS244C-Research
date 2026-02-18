# NCCL Contention Analysis Plan

## Overview
This plan outlines the steps to measure and analyze the impact of GPU contention on NCCL AllReduce performance. You will compare baseline results (no contention) with results obtained while running competing workloads on the same GPUs.

---

## Steps

### 1. Baseline Measurement (Already Completed)
- Run `all_reduce_perf` on 2 GPUs with no other jobs.
- Save results to `results_2gpu_allreduce.txt`.
- Analyze bandwidth and latency graphs.

### 2. Prepare Contention Workload
- Choose a competing workload:
  - Deep learning training job (e.g., PyTorch, TensorFlow).
  - CUDA kernel stress test (e.g., matrix multiplication, memory copy).
  - GPU benchmarking tool (e.g., `cuda_memtest`, custom kernel).
- Ensure the workload uses significant GPU resources (compute, memory, or both).

### 3. Launch Contention Workload
- Start the competing workload on one or both GPUs.
- Use `nvidia-smi` to verify GPU utilization increases.

### 4. Rerun NCCL AllReduce Test
- While the contention workload is running, rerun:
  ```
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 | tee results_2gpu_allreduce_contended.txt
  ```
- Save the new results for comparison.

### 5. Monitor and Profile
- Use `nvidia-smi` to monitor GPU utilization, memory usage, and process activity.
- Optionally, use `nvprof` or Nsight to profile kernel activity and resource contention.

### 6. Analyze Results
- Parse and plot bandwidth and latency from `results_2gpu_allreduce_contended.txt`.
- Compare with baseline graphs.
- Identify performance degradation, increased latency, or reduced bandwidth.

### 7. Vary Contention
- Repeat steps 3–6 with different types and intensities of contention:
  - Compute-heavy vs memory-heavy workloads.
  - Contention on one GPU vs both GPUs.
  - Multiple competing jobs.

### 8. Document Findings
- Summarize the impact of contention on NCCL performance.
- Identify which types of contention are most detrimental.
- Suggest possible mitigation strategies (e.g., scheduling, resource-aware NCCL tuning).

---

## Deliverables
- Contention test results file(s) (e.g., `results_2gpu_allreduce_contended.txt`)
- Updated bandwidth and latency graphs
- Comparative analysis report
- Recommendations for NCCL optimization under contention
