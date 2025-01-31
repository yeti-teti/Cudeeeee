// Kernel: Incresing Arithmetic Intensity via 2D Blocktiling. GFLOPS/s: 15971.7. Performance relative to cuBLAS: 68.7%

#pragma once

#include<algorithm>
#include<cassert>
#include<cstdlib>
#include<cstdio>
#include<cublas_v2.h>
#include<cuda_runtime.h>


#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))

template <const int, BM, const int BN, const int BK, const int TM, const int TN>
__global__ void  __launch__bounds__((BM * BN) / (TM * TN), 1) sgemm2DBlocktiling(int M, int N, float* A, float* B, int alpha, int beta, float* C){

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    // A thread is responsible for calcualting TM*TN elements in the blocktile
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    // Results per block / Results per thread = Threads per block
    assert(numThreadsBlocktile == blockDim.x);

    // BN/TN are the number of threads to span the column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Calculating the indices that this thread will load into SMEM
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    // Calculating the number of rows of As that are being loaded in a single step by a single block
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    // For both As and Bs we want each load to span full column width instead of spanning full row width for better GMEM coalescing 
    const uint strideB = numThreadsBlocktile / BN;

    // Allocate thread-local cache for results in register file
    float threadResuts[TM * TN] = {0.0};

    // Register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    

    // Outer most loop over blocktiles 
    for(uint bkIdx = 0; bkIdx < K; bkIdx += BK){

        // Populate the SMEM caches
        for(uint loadOffset = 0; loadOffset < BK; loadOffset += strideA){
            As[(innerRowA + loadOffset) * BK + innerColA]  = A[(innerRowA + loadOffset) * K + innerColA];
        }
        for(uint loadOffset = 0; loadOffset < BK; loadOffset += strideB){
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        // Advance BLOCKTILE
        A += BK; // Move BK columns to right
        B += BK * N; // Move BK rows down

        
        // Calculate per-thread results
        for(uint dotIdx = 0;dotIdx < BK ; dotIdx++){

            // Block into registers
            for(uint i=0;i<TM;i++){
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for(uint i=0;i<TN;i++){
                regN[i] = BS[dotIdx * BN + threadCol * TN + i];
            }
            for(uint resIdxM = 0; resIdxM < TM ; resIdxM++){
                for(uint resIdxN = 0;resIdxN < TN; resIdxN++){
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // Write out results
    for(uint resIdxM = 0;resIdxM < TM;resIdxM++){
        for(uint resIdxN = 0;resIdxN < TN;resIdxN++){
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]; 
        }
    }

}
