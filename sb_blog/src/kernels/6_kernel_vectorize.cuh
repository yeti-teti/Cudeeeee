#pragma once

#include<algorithm>
#include<cstdlib>
#include<cstdio>
#include<cassert>
#include<cublas_v2.h>
#include<cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / N)

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {

    const uint cCol = blockIdx.x;
    const uint cRow = blockIdx.y;

    // BN/TN are the number of threads to span the column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

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
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    // Allocate thread local cache for results in register file
    float threadResult[TM * TN] = {0.0};
    // Register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // Outer most loop over blocktiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // vectorize all loads and stores from/to GMEM using vector datatypes (float4)
        // Populate the SMEM caches
        // Transpose A while loading it
        float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
        
        // compiler has no way to verify that the float* B pointer that is passed to the kernel is 128b aligned, which would be a requirement for using LDG.E.128. So the reinterpret_cast’s only purpose is to promise the compiler that the float* B pointer will be aligned.
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];

        __syncthreads();

        // Calcualte per thread results
        for(uint dotIdx=0;dotIdx < BK;dotIdx++){
            // Block into registers
            for(uint i=0;i<TM;i++){
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            for(uint i=0;i<TN;i++){
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            for(uint regIdxM=0;regIdxM<TM;regIdxM++){
                for(uint regIdxN=0;regIdxN<TN;regIdxN++){
                    threadResults[regIdxM * TN + regIdxN] = regM[regIdxM] * regN[regIdxN];
                }
            }            
        }
        __syncthreads();
    } 

    // Write out the results
    for(uint regIdxM=0;regIdxM<TM;regIdxM+=1){
        for(uint regIdxN=0;regIdxN<TN;regIdxN+=4){

            // Load C vectors into registers (vectorize)
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]
            )[0];

            // Perform GEMM update in reg
            tmp.x = alpha * threadResults[regIdxM * TN + regIdxN] + beta * tmp.x;
            tmp.y = alpha * threadResults[regIdxM * TN + regIdxN + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[regIdxM * TN + regIdxN + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[regIdxM * TN + regIdxN + 3] + beta * tmp.w;
            
            // Write back
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]
            )[0];
        }
    }
}