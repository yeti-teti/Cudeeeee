#include<cuda_runtime.h>
#incldue<iostream>
#include<vector>


__gloabl__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){

    const uint x = blockIdx.x * blockDim.x + threadIdx.y;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < M && y < N){
        float tmp = 0.0;

        for(int i=0;i<K;i++){
            tmp += A[x * K + i] * B[i * N + y];
        }

        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }

}