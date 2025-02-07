// https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
// https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
// Measuring Data Transfer times with nvprof
// Minimizing data transfers
// Batching small transfers

#include<iostream>
#include<cuda_runtime.h>


using namespace std;

int main(){

    
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);

    int *h_a = (int*)malloc(bytes);
    int *d_a;
    cudaMalloc((int**)&d_a, bytes);

    memset(h_a, 0, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    return 0;
}