// Kernel: 1D Blocktiling for Calculating Multiple results per thread. GFlops/s: 8474.7, Performance relative to cuBLAS: 36.5%

#pragma once

#include<algorithm>
#include<cassert>
#include<cstdlib>
#include<cstdio>
#include<cublas_v2.h>
#include<cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))

// BM: No. of rows in Mat A each thread block will process
// BN: No. of columns in Mat B each thread block will process
// BK: Size of inner dimentions of the matrix tile that each thread block processes at at time. Determines the amount of shared memory used per block for tiles of A and B
// TM: No. of rows of the output matrix C that each thread computes. Determines the number of rows of computed by each thread, which influences how much register storage each thread uses.
// BM * BN: The size of the tile (block) of the output matrix C computed by each thread block.
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {

    // If we flip x and y here we get ~30% less performance for large matrices. The current, 30% faster configuration ensures that blocks with sequential. blockIDs access columns of B sequentially, while sharing the same row of A. The slower configuration would share columns of A, but access into B would be non-sequential. So the faster configuration has better spatial locality and hence a greater L2 hit rate.
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Each warp will calculate 32 * TM elements. with 32 being the columnar dim
    const int threadCol = threadIdx.x / BN;
    const int threadRow = threadIdx.x % BN;

    // Allocate space for current blocktile in SMEM
    __shared__ float As[BM * BN];
    __shared__ float Bs[BM * BN];

    // Move block tile to beginning of A's row and B's column.
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // We can adjust below so that each thread loads multiples entries and to better exploit the cache sizes
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    // how each thread in a thread block is assigned to load elements from global memory (GMEM) into shared memory (SMEM) for the matrices A and B
    // Determine which element of the input matrices A and B each thread is responsible for loading.
    // Enable coalesced memory access, where threads in the same warp access contiguous memory locations to maximize memory bandwidth.
    const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;

    // Allocate thread local cache for results in register file.
    float threadResults[TM] = {0.0};

    // Outer loop over block tiles
    for(uint bkIdx = 0;bkIdx < K;bkIdx++){

        // Populate the SMEM cache
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        // Advance blocktile
        A += BK ;
        B += BK * N;

        // Calculate per thread results
        for(uint dotIdx = 0; dotIdx < BK; dotIdx++){

            // we make the dotproduct loop the outside loop, which facilitates reuse of the Bs entry, which we can cache in a tmp var.
            float tmpB = Bs[dotIdx * BN + threadCol];
            for(uint resIdx = 0; resIdx < TM; resIdx++){
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    // Write the results
    for(uint resIdx = 0;resIdx < TM; resIdx++){
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadResults[resIdx] + beta * C[(threadIdx * TM + resIdx) * N + threadCol];
    }
}