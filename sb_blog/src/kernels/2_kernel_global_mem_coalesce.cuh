// Kernel: GMEM Coalescing, GFLOPS: 1986.5, Performance relative cuBLAS: 8.5% 

#pragma once

#include<cassert>
#include<cstdio>
#include<cstdlib>

#include<cublas_v2.h>
#include<cuda_runtime.h>


template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {

    const uint cRow = blockDim.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint cCol = blockDim.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if(cRow < M && cCol < N){

        float tmp = 0.0;

        for(int i=0;i<K;i++){
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }

}
