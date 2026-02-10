## SSH into farmshare
`ssh raigosa@login.farmshare.stanford.edu`

### Good to know functions for farmshare
View resources I've queued: `squeue -u raigosa`

## Requesting nodes on farmshare
`srun --partition=gpu --gres=gpu:2 --nodes=1 --pty bash`
provides the following result for `nvidia-smi`:
Mon Feb  9 15:35:44 2026       
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L40S                    Off |   00000000:CA:00.0 Off |                    0 |
| N/A   33C    P0             80W /  350W |       0MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA L40S                    Off |   00000000:E1:00.0 Off |                    0 |
| N/A   32C    P0             82W /  350W |       0MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

`nvidia-smi topo -m` provides:
```
vidia-smi topo -m
        GPU0    GPU1    NIC0    NIC1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NODE    NODE    NODE    1,33    1               N/A
GPU1    NODE     X      NODE    NODE    1,33    1               N/A
NIC0    NODE    NODE     X      PIX
NIC1    NODE    NODE    PIX      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
```

## Setting up the first nvcc experiment
`module load cuda/12.9.0` --> made it available on `which vncc`
`export CUDA_HOME=$(dirname $(dirname $(which nvcc)))` --> setup CUDA_HOME
`export NCCL_HOME="$CONDA_PREFIX"`
`export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH` --> make the cuda module available to run the build
`export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH` --> make nccl available

### I had needed to then install nccl...
- installed micromamba... to then be able to have nccl installed

### To actually run the experiment

*This is with MPI, I got some header issues and we should consider whether we want to use MPI later...*
`make MPI=1 CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"`

*This is without MPI. I'm doing this to get some quick baseline/preliminary results*
`make MPI=0 CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME"`

### I was just messing around with all_reduce. Save them to a file
`./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 | tee results_2gpu_allreduce.txt`