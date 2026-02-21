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





# Require contention and GPU count parameters
if [ $# -lt 2 ]; then
    echo "Usage: bash run_nccl_with_contention.sh <low|medium|high> <num_gpus>"
    echo "  <low|medium|high>: Contention level"
    echo "  <num_gpus>: Number of GPUs to use (e.g., 2, 4)"
    exit 1
fi
CONTENTION="$1"
NUM_GPUS="$2"


# Start the GPU stress benchmark with specified contention and GPU count
echo "Starting gpu_stress_benchmark with $CONTENTION contention on $NUM_GPUS GPUs..."
./gpu_stress_benchmark $CONTENTION $NUM_GPUS &
BENCH_PID=$!

# Wait briefly to ensure contention starts
sleep 2


# Ensure output directory exists
OUTDIR="contention_results"
mkdir -p "$OUTDIR"



# Get GPU name and tag
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_TAG=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | tr -d ' ')

# Run NCCL all_reduce_perf
echo "Running NCCL all_reduce_perf on $NUM_GPUS GPUs..."
OUTFILE="$OUTDIR/results_${GPU_TAG}-${NUM_GPUS}gpu_allreduce_contended_${CONTENTION}.txt"
../../nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g $NUM_GPUS | tee "$OUTFILE"

# Kill the GPU stress benchmark after NCCL test finishes
echo "Killing GPU stress benchmark..."
kill $BENCH_PID || true

echo "Contention experiment complete. Results saved to $OUTFILE."