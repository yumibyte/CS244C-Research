# NCCL Performance Baseline and Analysis on AWS

## Overview
This plan outlines how to:
1. Set up a cost-effective AWS GPU cluster for NCCL benchmarking.
2. Create a script to run baseline NCCL performance tests across message sizes.
3. Chart results to identify performance bottlenecks and opportunities for improvement.

## Requirements
- AWS account with permission to launch EC2 GPU instances.
- Use the cheapest cluster configuration that can run NCCL AllReduce tests reliably (e.g., single-GPU `g4dn.xlarge` or `g5.2xlarge`, or multi-GPU spot instances if needed).
- Access to spot instances for cost savings.
- Deep Learning AMI (NVIDIA) or similar with CUDA/NCCL pre-installed.
- Python (for scripting and plotting).
- `nccl-tests` benchmarking suite.
- Tools: `nvidia-smi`, `matplotlib` (for graphing).

## Implementation Steps
### 1. Setting Up the AWS Cluster
- Choose the cheapest instance type that can reliably run NCCL AllReduce tests:
  - For single-GPU: `g4dn.xlarge` or `g5.2xlarge`.
  - For multi-GPU: use spot instances such as `p3.8xlarge` or `p4d.24xlarge` only if required for your research.
- Launch instance using Deep Learning AMI (NVIDIA) for pre-installed CUDA/NCCL.
- SSH into the instance.
- Verify GPU topology with `nvidia-smi topo -m`.
- Install required packages (Python, matplotlib, git).

### 2. Creating a Baseline Measurement Script
- Clone and build `nccl-tests`:
  - `git clone https://github.com/NVIDIA/nccl-tests.git`
  - `cd nccl-tests && make MPI=1 CUDA_HOME=/usr/local/cuda`
- Write a shell or Python script to run `all_reduce_perf` across a range of message sizes:
  - Example: `./build/all_reduce_perf -b 8 -e 8G -f 2 -g <num_gpus>`
- Save output (latency, bandwidth) to CSV or text file for analysis.

### 3. Charting and Analyzing Results
- Parse the output files to extract message size, bandwidth, and latency.
- Use Python (matplotlib or seaborn) to plot:
  - Bandwidth vs. message size.
  - Latency vs. message size.
- Highlight regions where performance drops or latency spikes.
- Annotate graphs to indicate potential areas for NCCL tuning or algorithm changes.

## Testing
- Confirm instance launches and GPUs are visible.
- Validate `nccl-tests` runs successfully for all desired message sizes.
- Check that output files are correctly generated and parsed.
- Ensure graphs accurately reflect the data and highlight performance trends.

## Assumptions & Notes
- Spot instances may be interrupted; save results frequently.
- Deep Learning AMI simplifies setup but can use Ubuntu with manual CUDA/NCCL install.
- Multi-GPU tests require appropriate instance types and may incur higher costs.

## Deliverables
- Step-by-step AWS setup instructions.
- Baseline measurement script (shell or Python) for AllReduce only.
- Example output files and Python plotting script.
- Sample graphs with analysis notes.
