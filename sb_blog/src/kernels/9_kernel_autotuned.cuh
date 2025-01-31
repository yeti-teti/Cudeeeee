#pragma once

#include<algorithm>
#include<cassert>
#include<cstdlib>
#include<cstdio>
#include<cublas_v2.h>
#include<cuda_runtime.h>


#define CEIL_DIV(M, N) (((M) + (N)-1) / N)
const K9_NUM_THREADS = 256;

// BM, BN and BK, which specify how much data we cache from GMEM into SMEM.
// TM and TN, which specify how much data we cache from SMEM into the registers.
// Optimal parameters vary quite a bit depending on the GPU model
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS) sgemmAutotuned(int M, int N, int K , float* A, float* B, int alpha, int beta, float *C){

    const uint cCol = blockIdx.x;
    const uint cRow = blockIdx.y;

    // Size of warptile
    const uint WM = TM * 16;
    const uint WN = TN * 16;
    // Iterations of warptile
    constexpr int WMITER = CEIL_DIV(BM, WM);
    constexpr int WNITER = CEIL_DIV(BN, WN);

    // Placement of the thread in the warptile
    const int threadCol = threadIdx.x % (WM / TN);
    const int threadRow = threadIdx.x / (WM / TN);

     // Allocate space for current blocks in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

     // Move blocktile to beginnings of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Calculating the indices that this thread will load in SMEM
    // We'll load 128bit / 4bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

    // Allocate thread local cache for results in register file
    float threadResult[WMITER * WNITER * TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        float4 tmp = reinterpret_cast<float4 *>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];
        // transpose A while storing it
        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        reinterpret_cast<float4 *>(
            &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        

    }

}
