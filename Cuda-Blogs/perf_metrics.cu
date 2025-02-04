// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

#include<cuda_runtime.h>
#include<iostream>

using namespace std;

__global__ void saxpy(int n, int a, float* x, float *y){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i < n ){
        y[i] = a * x[i] + y[i];
    }
}


// Host-Device Synchronization
// Data transfer between host and device using cudaMemcpy() are synchronous
// Kernel launch are asynchronous
// Timing Kernel 
    // Execution with CPU timers
    //  CUDA events
int main(){

    int N = 20 * (1 << 20);

    float *x, *y, *d_x, *d_y;

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Initialize the device arrays
    for(int i=0;i<N;i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    saxpy<<<(N+511)/512, 512>>>(N, 2.0f, d_x, d_y);
    cudaEventRecord(stop);

    // Copy result from device to host
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Error checking
    float maxError = 0.0f;
    for(int i=0;i<N;i++){
        maxError = max(maxError, abs(y[i] - 4.0f));
    }
    cout<<"Max Error: "<<maxError<<endl;
    cout<<"Effective Bandwidth (GB/s): "<<N*4*3/milliseconds/1e6<<endl;

    // Cleaning up
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

}



