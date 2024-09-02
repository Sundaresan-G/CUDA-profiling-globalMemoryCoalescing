# CUDA Global Memory Coalescing Profiling Trials with Bandwidth measurement

## This program (credits to NVIDIA) studies the effect of offset and stride on global memory access in GPUs.

## We profile using NVIDIA Nsight Compute with respect to various block sizes on the below GPU. The best results are observed for block sizes above 256.

### Irrespective of block sizes, the total number of warps remain the same. And there is no addtional overhead switching warps of different blocks on same SM and this is same as switching between warps of same block. There is some preamble and postamble overhead costs on launching blocks as it involves a few operations such as assigning/freeing registers and/or shared memory. However, the major overhead is that once a block is active on SM, it cannot be resigned from SM till it completes. Thus, in such cases, it is not possible for a warp from another block to be active on that SM. Since, the current program is memory bound (high latency), using smaller block sizes such as 64 prevents effective switching of warps. 

GPU: Tesla V100 - SXM2 - 32GB

Compilation: nvcc -arch=sm_70 main.cu

Profiling offset kernel: ncu -k offset -s 1 -c 32 --set full -f --export out a.out

Profiling All the kernels: ncu --set full -f --export out a.out