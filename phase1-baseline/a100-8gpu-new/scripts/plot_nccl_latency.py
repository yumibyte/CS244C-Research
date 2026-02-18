import re
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving without display
import matplotlib.pyplot as plt
import argparse
import os

def parse_nccl_results(filename):
    sizes = []
    out_of_place_times = []
    in_place_times = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 12:
                try:
                    size = int(parts[0])
                    out_time = float(parts[5])  # out-of-place latency (us)
                    in_time = float(parts[9])   # in-place latency (us)
                    sizes.append(size)
                    out_of_place_times.append(out_time)
                    in_place_times.append(in_time)
                except (ValueError, IndexError):
                    continue
    return sizes, out_of_place_times, in_place_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse NCCL test results and plot latency graphs.')
    parser.add_argument('input_file', type=str, help='Input NCCL results txt file')
    args = parser.parse_args()

    filename = args.input_file
    sizes, out_times, in_times = parse_nccl_results(filename)
    print("Sizes:", sizes)
    print("Out-of-place times:", out_times)
    print("In-place times:", in_times)

    # Create output directory if it doesn't exist
    output_dir = "latency_graphs"
    os.makedirs(output_dir, exist_ok=True)

    # Use input file base name for output
    base = os.path.splitext(os.path.basename(filename))[0]
    out_of_place_png = f"{base}_out_of_place.png"
    in_place_png = f"{base}_in_place.png"

    # Plot out-of-place latency vs message size
    plt.figure()
    plt.plot(sizes, out_times, color='blue')
    plt.xlabel('Message Size (Bytes)')
    plt.ylabel('Latency (us)')
    plt.xscale('log')
    plt.title('NCCL AllReduce Out-of-place Latency vs Message Size')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, out_of_place_png))
    plt.close()

    # Plot in-place latency vs message size
    plt.figure()
    plt.plot(sizes, in_times, color='green')
    plt.xlabel('Message Size (Bytes)')
    plt.ylabel('Latency (us)')
    plt.xscale('log')
    plt.title('NCCL AllReduce In-place Latency vs Message Size')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, in_place_png))
    plt.close()
