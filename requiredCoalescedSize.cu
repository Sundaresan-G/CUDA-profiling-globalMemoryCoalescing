/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


// In this program, consecutive group of threads access either 32-byte, 64-byte or 128-byte. 
// But subsequent group of threads shall access farther memory locations.
// For 32 byte example, threads 0-7 access 4 byte each at memory location 0, but threads 8-16 access 4 byte each at memory location 64 and so on.
// Use only float datatype
#include <stdio.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

template <typename T>
__global__ void coalescedAccessCheck(T* a, int s)
{
  // s is either 32, 64 or 128
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  s /= 4; // either 8 threads access 32 byte or 16 threads access 64 bytes
  int j = i%s + (i/s)*s*2;
  a[j] = a[j] + 1;
}

template <typename T>
void runTest(int deviceId, int nMB)
{
  int blockSize = 512;
  float ms;

  T *d_a;
  cudaEvent_t startEvent, stopEvent;
    
  int n = nMB*1024*1024/sizeof(T);

  // NB:  d_a(33*nMB) for stride case
  checkCuda( cudaMalloc(&d_a, n * 33 * sizeof(T)) );

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  printf("CoalescedBytes, Bandwidth (GB/s):\n");
  
  coalescedAccessCheck<<<n/blockSize, blockSize>>>(d_a, 32); // warm up

  // Here i specifies the number of bytes of contiguous access
  for (int i = 4; i <= 128; i = i << 1) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    coalescedAccessCheck<<<n/blockSize, blockSize>>>(d_a, i);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%13d, %f\n", i, 2*nMB/ms);
  }

  printf("\n");

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  cudaFree(d_a);
}

int main(int argc, char **argv)
{
  int nMB = 4 * 16;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {    
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }
  
  cudaDeviceProp prop;
  
  checkCuda( cudaSetDevice(deviceId) );

  checkCuda( cudaGetDeviceProperties(&prop, deviceId) );
  printf("Device: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", nMB);
  
  printf("%s Precision\n", bFp64 ? "Double" : "Single");
  
  if (bFp64) runTest<double>(deviceId, nMB);
  else       runTest<float>(deviceId, nMB);
}