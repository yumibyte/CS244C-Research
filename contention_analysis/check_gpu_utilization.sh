#!/bin/bash
# Monitor GPU utilization while running gpu_stress_benchmark

set -e

# Compile the benchmark if needed
if [ ! -f gpu_stress_benchmark ]; then
    echo "Compiling gpu_stress_benchmark.cu..."
    nvcc -lcublas gpu_stress_benchmark.cu -o gpu_stress_benchmark
fi

# Start the benchmark in the background
echo "Starting gpu_stress_benchmark..."
./gpu_stress_benchmark &
BENCH_PID=$!


# Monitor GPU utilization for 30 seconds and log to CSV
echo "Monitoring GPU utilization for 30 seconds..."
CSV_FILE="gpu_utilization_log.csv"
echo "timestamp,gpu_utilization,memory_used" > $CSV_FILE
for i in {1..30}; do
    TIMESTAMP=$(date +%s)
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | \
        awk -v ts="$TIMESTAMP" -F',' '{print ts "," $1 "," $2}' >> $CSV_FILE
    sleep 1
done

# Kill the benchmark
echo "Stopping gpu_stress_benchmark..."
kill $BENCH_PID || true

echo "Done."
