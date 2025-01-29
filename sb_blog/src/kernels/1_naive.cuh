// Kernel: Naive, GFLOPs/s: 309.0 ,  Performance relative to cuBLAS: 1.3% 

#pragma once

#include<stdio.h>
#include<stdlib.h>
#include<cublas_v2.h>
#include<cuda_runtime.h>

/*
Matrix sizes:
MxK * KxN = MxN
*/


// The kernel will return different values based on the thread that's accessing them. 
// Paremeters: (M: No. of rows in mat A, N: No. of Cols in Mat B, K: No. of cols in Mat A and No. of rows in Mat B, alpha, pointer to input Mat A, pointer to input Mat B, beta, pointer to output Mat C)
__global__ void sgemm_naive(int M, int N, int K, int alpha, float* A, float* B, int beta, float* C){

    // Global index to access elements of the arrays
    // Loads rows of matrix non-consecutively from memory.
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // 'if' condition to check when M or N are not multiples of 32
    // if statement is necessary to make things work under tile quantization
    if(x < M && y < N){
        float tmp = 0.0;

        for(int i=0;i<K;i++){
            tmp += A[x * K + i] * B[i * N + y];
        }

        // Compute position in C that this thread is responsible for
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    } 
}





