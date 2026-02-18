# Source user bashrc to ensure environment variables are set
source ~/.bashrc
#!/bin/bash
# Run GPU stress benchmark and NCCL all_reduce_perf for contention analysis

set -e

# Compile the CUDA benchmark if not already present
if [ ! -f gpu_stress_benchmark ]; then
    echo "Compiling gpu_stress_benchmark.cu..."
    nvcc -lcublas gpu_stress_benchmark.cu -o gpu_stress_benchmark
fi

# Accept contention parameter (low, medium, high)
CONTENTION="high"
if [ $# -ge 1 ]; then
    CONTENTION="$1"
fi

# Start the GPU stress benchmark with specified contention
echo "Starting gpu_stress_benchmark with $CONTENTION contention..."
./gpu_stress_benchmark $CONTENTION &
BENCH_PID=$!

# Wait briefly to ensure contention starts
sleep 2


# Ensure output directory exists
OUTDIR="contention_results"
mkdir -p "$OUTDIR"

# Run NCCL all_reduce_perf
echo "Running NCCL all_reduce_perf..."
OUTFILE="$OUTDIR/results_2gpu_allreduce_contended_${CONTENTION}.txt"
../nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 | tee "$OUTFILE"

# Kill the GPU stress benchmark after NCCL test finishes
echo "Killing GPU stress benchmark..."
kill $BENCH_PID || true

echo "Contention experiment complete. Results saved to $OUTFILE."