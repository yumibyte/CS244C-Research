#!/bin/bash
# Run GPU stress benchmark and NCCL all_reduce_perf for contention analysis

set -e

# Compile the CUDA benchmark if not already present
if [ ! -f gpu_stress_benchmark ]; then
    echo "Compiling gpu_stress_benchmark.cu..."
    nvcc -lcublas gpu_stress_benchmark.cu -o gpu_stress_benchmark
fi

# Start the GPU stress benchmark in the background
echo "Starting GPU stress benchmark..."
./gpu_stress_benchmark &
BENCH_PID=$!

# Wait briefly to ensure contention starts
sleep 2

# Run NCCL all_reduce_perf
echo "Running NCCL all_reduce_perf..."
../nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 | tee results_2gpu_allreduce_contended.txt

# Kill the GPU stress benchmark after NCCL test finishes
echo "Killing GPU stress benchmark..."
kill $BENCH_PID || true

echo "Contention experiment complete. Results saved to results_2gpu_allreduce_contended.txt."