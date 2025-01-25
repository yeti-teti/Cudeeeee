//https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/


#include<cuda_runtime.h>
#include<iostream>

using namespace std;

// Device Code
__global__
void saxpy(int n, float a , float *x, float *y){ // Parameters: size of array, constant, input array a, input array b

    // Global index to access elements of the arrays
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Value check for number of elements n
    if(i < n){
        y[i] = a * x[i] + y[i]; // // Element-wise work of SAXPY
    }
   
}

// Main function (Host code)
int main(){
    
    // Size of the arrays
    int N = 1<<20;

    // Allocate memory for host and device
    
    // Host arrays
    //(float*) casts from a pointer to void to a pointer to float
    float *x = (float*)malloc(N*sizeof(float));
    float *y = (float*)malloc(N*sizeof(float));

    // Device Arrays
    float *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    // Initialize the device arrays
    for(int i=0;i<N;i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Copy values from host ot device
    cudaMemcpy(d_x, x, N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float),cudaMemcpyHostToDevice);

    // Launching the SAXPY kernels
    // Perform Saxpy of 1M elements
    saxpy<<<(N+255) / 256, 256>>>(N, 2.0, d_x, d_y);

    // Copy result from device to host
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Error checking
    float maxError = 0.0f;
    for(int i=0;i<N;i++){
        maxError = max(maxError, abs(y[i] - 4.0f));
    }
    cout<<maxError<<endl;


    // Cleaning up
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}