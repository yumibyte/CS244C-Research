# Phase 2: GPU Contention and NCCL Performance

## Overview

This phase investigates how concurrent GPU compute (contention) affects NCCL AllReduce performance. You will:
- Run a CUDA-based GPU stress benchmark at different contention levels (low, medium, high).
- Simultaneously run NCCL AllReduce tests to measure communication performance under contention.
- Monitor and plot GPU utilization during these experiments.

---

## How to Run GPU Stress and NCCL AllReduce Tests

### 1. Run NCCL with GPU Contention

Use the provided bash script to launch both the GPU stress benchmark and the NCCL AllReduce test. The script accepts a contention level argument: `low`, `medium`, or `high`.

```bash
bash run_nccl_with_contention.sh [low|medium|high]
```

- Example for high contention:
  ```bash
  bash run_nccl_with_contention.sh high
  ```
- The script will:
  - Compile the CUDA stress benchmark if needed.
  - Start the stress benchmark at the specified contention level.
  - Run the NCCL AllReduce test (`all_reduce_perf`) and save results to `contention_results/results_2gpu_allreduce_contended_<level>.txt`.
  - Kill the stress benchmark after the test completes.

---

## How to Monitor and Plot GPU Utilization

### 2. Monitor GPU Utilization

To log GPU utilization while the stress benchmark is running, use:

```bash
bash check_gpu_utilization.sh [low|medium|high]
```

- This will:
  - Compile the stress benchmark.
  - Start the stress benchmark at the specified level.
  - Log GPU utilization and memory usage for 30 seconds to `gpu_utilization_logs/gpu_utilization_log_<level>.csv`.

---

### 3. Plot GPU Utilization

To generate a plot from the utilization log, use the Python plotting script:

```bash
python plot_gpu_utilization.py
```

- By default, the script reads `gpu_utilization_logs/gpu_utilization_log_high.csv`.
- To plot a different file, edit the `csv_file` variable at the top of `plot_gpu_utilization.py` or modify the script to accept a filename argument.
- The plot will be saved as a PNG in the same directory.

---

## Output Files

- **NCCL AllReduce results:**  
  `contention_results/results_2gpu_allreduce_contended_<level>.txt`
- **GPU utilization logs:**  
  `gpu_utilization_logs/gpu_utilization_log_<level>.csv`
- **Utilization plots:**  
  `gpu_utilization_logs/gpu_utilization_plot_<level>.png`

---

## Notes

- You can run and plot for all three contention levels (`low`, `medium`, `high`) to compare results.
- For further analysis, see the plotting and analysis scripts in this folder.
