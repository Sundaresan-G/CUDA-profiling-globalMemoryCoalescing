## CUDA Global Memory Coalescing Profiling Trials with Bandwidth measurement

# This program (credits to NVIDIA) studies the effect of offset and stride on global memory access in GPUs.

# We profile using NVIDIA Nsight Compute with respect to various block sizes on the below GPU. The best results are observed for block sizes above 256.

GPU: Tesla V100 - SXM2 - 32GB

Compilation: nvcc -arch=sm_70 main.cu

Profiling offset kernel: ncu -k offset -s 1 -c 32 --set full -f --export out a.out

Profiling All the kernels: ncu --set full -f --export out a.out