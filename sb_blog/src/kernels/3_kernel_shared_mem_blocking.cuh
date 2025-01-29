// Kernel: Shared memory Cache-Blocking, GFlops/s: 2980.3 , Performance relative to cuBLAS: 12.8%

#pragma once

#include<algorithm>
#include<cstdlib>
#include<cstdio>
#include<cublas_v2.h>
#include<cuda_runtime.h>


// To detetmine how many times we need to divide M by N to cover all elements, and rounding up if there is a remainder
#define CEIL_DIV(M,N) (((M) + (N) - 1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, flaot* A, float* B, int alpha, int beta, float* C){

    // Output block we want to compute in this threadblock
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // Allocate buffer for current block in fast shared mem
    // Shared mem is shared between all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSiZE * BLOCKSIZE];

    // Inner row and col we are accessing in this thread
    const uint threadCol = threadIdx.x / BLOCKSIZE;
    const uint threadRow = threadIdx.x % BLOCKSIZE;

    // Advance pointers to starting positions
    A += cRow * BLOCKSIZE * K; // row = cRow, col = 0
    B += cCol * BLOCKSIZE; // row = 0, col = cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row = cRow, col = cCol

    float tmp = 0.0;
    // The outer loop advances A along columns and B along rows until we have fully calculated results of C
    for(int bkIdx = 0;bkIdx < K; bkIdx++){

        // Have each thread load one of the elements in A & B from global memory into shared memory.
        // Make threadCol (=threadIdx.x) the consecutive index to allow global memory coalescing
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // Block threads in this block untill cache is fully populated
        __syncthreads();


        // Advance pointers into next chunck
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // Execute the dot product on the currently cached  block
        for(int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++){
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // Need to sync again to at the end to avoid faster threads fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}


