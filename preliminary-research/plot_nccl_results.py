#!/usr/bin/env python3
"""Parse and plot NCCL test results."""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_nccl_output(filename):
    """Parse NCCL test output file."""
    sizes = []
    out_of_place_bw = []
    in_place_bw = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Skip comments and headers
            if line.startswith('#') or not line.strip():
                continue
            
            # Parse data lines
            parts = line.split()
            if len(parts) >= 12:
                try:
                    size = int(parts[0])
                    oop_bw = float(parts[5])  # out-of-place algbw
                    ip_bw = float(parts[9])   # in-place algbw
                    
                    sizes.append(size)
                    out_of_place_bw.append(oop_bw)
                    in_place_bw.append(ip_bw)
                except (ValueError, IndexError):
                    continue
    
    return np.array(sizes), np.array(out_of_place_bw), np.array(in_place_bw)

def plot_bandwidth(sizes, oop_bw, ip_bw, title="NCCL All-Reduce Bandwidth"):
    """Plot bandwidth vs message size."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert bytes to human-readable
    size_labels = []
    for s in sizes:
        if s < 1024:
            size_labels.append(f"{s}B")
        elif s < 1024*1024:
            size_labels.append(f"{s//1024}KB")
        else:
            size_labels.append(f"{s//(1024*1024)}MB")
    
    # Plot both curves
    ax.plot(range(len(sizes)), oop_bw, 'o-', label='Out-of-place', linewidth=2, markersize=6)
    ax.plot(range(len(sizes)), ip_bw, 's-', label='In-place', linewidth=2, markersize=6)
    
    # Formatting
    ax.set_xlabel('Message Size', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set x-axis labels (show every few points to avoid crowding)
    step = max(1, len(sizes) // 10)
    ax.set_xticks(range(0, len(sizes), step))
    ax.set_xticklabels([size_labels[i] for i in range(0, len(sizes), step)], rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def print_summary(sizes, oop_bw, ip_bw):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("NCCL Performance Summary")
    print("="*60)
    
    # Find peak bandwidth
    peak_oop_idx = np.argmax(oop_bw)
    peak_ip_idx = np.argmax(ip_bw)
    
    print(f"\nPeak Out-of-Place Bandwidth: {oop_bw[peak_oop_idx]:.2f} GB/s")
    print(f"  at message size: {sizes[peak_oop_idx]:,} bytes ({sizes[peak_oop_idx]/(1024**2):.1f} MB)")
    
    print(f"\nPeak In-Place Bandwidth: {ip_bw[peak_ip_idx]:.2f} GB/s")
    print(f"  at message size: {sizes[peak_ip_idx]:,} bytes ({sizes[peak_ip_idx]/(1024**2):.1f} MB)")
    
    # Small message performance
    small_msg_threshold = 1024  # 1KB
    small_indices = sizes <= small_msg_threshold
    if np.any(small_indices):
        avg_small = np.mean(oop_bw[small_indices])
        print(f"\nAverage bandwidth for messages ≤ 1KB: {avg_small:.3f} GB/s")
    
    # Large message performance
    large_msg_threshold = 1024 * 1024  # 1MB
    large_indices = sizes >= large_msg_threshold
    if np.any(large_indices):
        avg_large = np.mean(oop_bw[large_indices])
        print(f"Average bandwidth for messages ≥ 1MB: {avg_large:.2f} GB/s")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_nccl_results.py <nccl_output_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Parse data
    sizes, oop_bw, ip_bw = parse_nccl_output(filename)
    
    if len(sizes) == 0:
        print("Error: No data found in file")
        sys.exit(1)
    
    # Print summary
    print_summary(sizes, oop_bw, ip_bw)
    
    # Create plot
    fig = plot_bandwidth(sizes, oop_bw, ip_bw, 
                        title="NCCL All-Reduce Bandwidth (2 GPUs, FarmShare oat-05)")
    
    # Save plot
    output_file = filename.replace('.txt', '_plot.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Show plot
    plt.show()
