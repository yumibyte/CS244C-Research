# NCCL Benchmark Results Summary
## System Configuration
- **GPUs**: 8x NVIDIA A100-SXM4-80GB
- **NVLink**: Full NV12 topology (12 links per GPU @ 25 GB/s each = 300 GB/s per GPU)
- **NUMA**: 2 nodes (GPUs 0-3 on NUMA 0, GPUs 4-7 on NUMA 1)
- **NCCL Version**: 2.24.3

## Benchmark Results (Peak Performance)

### 1. All-Reduce ⭐ **MOST CRITICAL**
**Peak Bandwidth**: ~850 GB/s @ 2GB message size
- Small messages (1KB-1MB): 0.01-52 GB/s
- Medium messages (1MB-128MB): 100-700 GB/s
- Large messages (128MB-2GB): 700-850 GB/s

**Why it matters**: All-reduce is THE most important operation for distributed training. It's used in every gradient synchronization step during backpropagation. Performance here directly impacts training throughput.

### 2. All-Gather
**Peak Bandwidth**: ~860 GB/s @ 2GB message size
- Small messages (1KB-1MB): 0.01-100 GB/s
- Medium messages (1MB-128MB): 60-120 GB/s
- Large messages (128MB-2GB): 200-860 GB/s

**Why it matters**: Used in pipeline parallelism and some distributed training strategies (e.g., ZeRO optimizer). Less frequent than all-reduce but still important.

### 3. Reduce-Scatter
**Peak Bandwidth**: ~860 GB/s @ 2GB message size
- Small messages (1KB-1MB): 0.01-100 GB/s
- Medium messages (1MB-128MB): 40-80 GB/s
- Large messages (128MB-2GB): 200-860 GB/s

**Why it matters**: Used in advanced distributed training (ZeRO, FSDP). All-reduce = reduce-scatter + all-gather, so this tests half of the all-reduce operation.

### 4. Broadcast
**Peak Bandwidth**: ~850 GB/s @ 2GB message size
- Small messages (1KB-1MB): 0.01-90 GB/s
- Medium messages (1MB-128MB): 40-140 GB/s
- Large messages (128MB-2GB): 200-850 GB/s

**Why it matters**: Used for distributing model weights, broadcasting hyperparameters, and initialization. Less frequent but critical for correctness.

### 5. Send/Recv
**Peak Bandwidth**: ~760 GB/s @ 2GB message size
- Small messages (1KB-1MB): 0.01-200 GB/s
- Medium messages (1MB-128MB): 80-220 GB/s
- Large messages (128MB-2GB): 300-760 GB/s

**Why it matters**: Point-to-point communication for pipeline parallelism and custom communication patterns. Less common in standard data parallelism.

---

## Which Benchmarks Matter Most?

### 🥇 **#1: All-Reduce** (CRITICAL)
- **Why**: Used in EVERY training iteration for gradient synchronization
- **Impact**: Directly affects training speed
- **What to look for**: 
  - Large message performance (>100MB) for large models
  - Medium message performance (1-100MB) for typical models
  - Latency for small messages matters for frequent small updates

### 🥈 **#2: Reduce-Scatter + All-Gather** (IMPORTANT)
- **Why**: Used in ZeRO optimizer and FSDP (Fully Sharded Data Parallel)
- **Impact**: Critical for training very large models that don't fit in single GPU memory
- **What to look for**: Combined performance should match all-reduce

### 🥉 **#3: Broadcast** (MODERATE)
- **Why**: Used for initialization and occasional parameter distribution
- **Impact**: Affects startup time and checkpoint loading
- **What to look for**: Large message performance for model weight distribution

### **#4: Send/Recv** (SITUATIONAL)
- **Why**: Only critical if using pipeline parallelism
- **Impact**: Affects pipeline bubble time
- **What to look for**: Latency and bandwidth for activation sizes

---

## Performance Assessment: EXCELLENT ✅

Your system shows:
1. **Peak bandwidth ~850 GB/s** - excellent for 8 A100s with NVLink
2. **Good scaling** from small to large messages
3. **Consistent performance** across all collective operations
4. **No errors** in any tests (0 out-of-bounds values)

### Expected vs Actual Performance
- **Theoretical NVLink bandwidth per GPU**: 300 GB/s (12 links × 25 GB/s)
- **Achieved all-reduce bandwidth**: ~850 GB/s aggregate
- **Efficiency**: ~35% of theoretical peak (typical for collective operations)

This is **excellent performance** for NCCL on A100s. The efficiency is within expected range because:
- Collective operations require synchronization overhead
- Ring/tree algorithms don't use all links simultaneously
- NCCL optimizes for latency and correctness, not just raw bandwidth

---

## Key Takeaways for Distributed Training

1. **For standard data parallel training**: Focus on all-reduce performance
2. **For large model training (ZeRO/FSDP)**: Monitor reduce-scatter + all-gather
3. **Message size matters**: Your system performs best with >100MB messages
4. **NVLink is working perfectly**: Full connectivity verified and utilized

## Recommended Test Command for Quick Validation
```bash
# Quick all-reduce test (most important)
mpirun -np 8 ./build/all_reduce_perf -b 8M -e 512M -f 2 -g 1
```

This tests the typical gradient size range for most models.
