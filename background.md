Notes on NCCL


Tuners
What is a tuner?
Prior to launching a collective (group of GPU performing a single communication operation) it selects an algorithm and protocol then launches the kernels (which is out of our control). It’s NOT a scheduler. It’s selects a policy
A collective can be milliseconds and is a very quick call
Inputs:
Number of ranks (integer ID corresponding to one participant in our collective. 1 rank = 1 GPU)
Collective type (allreduce, broadcast, etc.)
AllReduce:
Every gpu contributes data, reduce the data, everyone gets the result
Broadcast:
Only one gpu has data, data gets sent to all gpus
Why might we choose AllReduce > Broadcast?
Well NCCL doesn’t actually select the collective type, the choice is either made by whatever DL framework or the developer. So think of this as “preset” in our case
Topology summary (PCIe, NVLink, etc.)
This is a map of how the GPUs are connected and their individual performances
This could be Ring/Tree/Split Tree
Something like ring might have ALL high-bandwidth links
Split-tree might exploit a couple high-speed paths… 
Example: https://github.com/NVIDIA/nccl/tree/master/ext-tuner/example



What does NCCL already do?:
Consider topology:
It might choose a ring for allreduce
Or a tree for broadcast

NCCL doesn’t change parameters mid-collective

What can we do that is different?
Temporal behavior:
Topology AND temporal information can allow some form of dynamic tuning (but we should note that you CANNOT dynamically change physical topology mid-collective, but it seems like you can change the algorithm…)
NCCL doesn’t consider prior iterations, history, or workload phases

Potential research question:
How can we dynamically choose the best algorithm DYNAMICALLY in a collective on a fixed topology of GPUs?
Consider if we used AllReduce in one treating it as a ring then treating it as a tree


Revised: How can we dynamically choose the best NCCL algorithm and protocol across iterations on a fixed GPU topology by leveraging temporal workload information and prior execution history?


