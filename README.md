# CUDA Global Memory Coalescing Profiling Trials with Bandwidth measurement

## Offset And Stride Effects

### This program (credits to NVIDIA) studies the effect of offset and stride on global memory access in GPUs.

### We profile using NVIDIA Nsight Compute with respect to various block sizes on the below GPU. The best results are observed for block sizes above 256.

### Irrespective of block sizes, the total number of warps remain the same. And there is no addtional overhead switching warps of different blocks on same SM and this is same as switching between warps of same block. There is some preamble and postamble overhead costs on launching blocks as it involves a few operations such as assigning/freeing registers and/or shared memory. However, the major overhead is that once a block is active on SM, it cannot be resigned from SM till it completes. Thus, in such cases, it is not possible for a warp from another block to be active on that SM. Since, the current program is memory bound (high latency), using smaller block sizes such as 64 prevents effective switching of warps. 

GPU: Tesla V100 - SXM2 - 32GB

Compilation: nvcc -arch=sm_70 OffsetAndStrideEffects.cu

Profiling offset kernel: ncu -k offset -s 1 -c 32 --set full -f --export out a.out

Profiling All the kernels: ncu --set full -f --export out a.out

## Required Colesced Size trials

### In this program, we study the effects of various coalesced global memory accesses of sizes 4, 8, 16, 32, 64 and 128 bytes.

### For instance, coalesced size of 16 bytes implies that 4 threads access first 4 float data type elements 0, 1, 2, 3, the next 4 threads access elements 8, 9, 10, 11 and so on. Similary for 32 bytes, first 8 threads access elements 0 to 7, the next 8 threads access 16 to 23 and so on.

### Based on our experiments on V100, we could conclude that optimal coalesced size is 64 bytes. This is likely since L1 cache line width is 32 bytes but L2 cache line width is 64 bytes (and so a single access with 512-bit memory controller). Ref: https://arxiv.org/pdf/1804.06826 (Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking).

GPU: Tesla V100 - SXM2 - 32GB

Compilation: nvcc -arch=sm_70 requiredCoalescedSize.cu