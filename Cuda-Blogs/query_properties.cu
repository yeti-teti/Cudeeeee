// https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/

#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__ void saxpy(int n, int a, float* x, float *y){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i < n ){
        y[i] = a * x[i] + y[i];
    }
}


// Querying Device Properties
// Calculating the theoretical peak bandwidth by querying the attached device (or devices) for the needed information.
int main(){

    int nDevices;

    // Returns the number of cuda capable devices attached to the system
    cudaGetDeviceCount(&nDevices);

    // Loop through each device and calculate the theoretical peak bandwidth for each device.
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    } 


    // Handling cuda Errors
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


    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // checks for both synchronous and asynchronous errors. Invalid execution configuration parameters, e.g. too many threads per thread block, are reflected in the value of errSync returned by cudaGetLastError()
    // Asynchronous errors that occur on the device after control is returned to the host, such as out-of-bounds memory accesses, require a synchronization mechanism such as cudaDeviceSynchronize(), which blocks the host thread until all previously issued commands have completed. 
    // Any asynchronous error is returned by cudaDeviceSynchronize(). We can also check for asynchronous errors and reset the runtime error state by modifying the last statement to call cudaGetLastError().
    saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if(errSync != cudaSuccess){
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    }
    if(errSync != cudaSuccess){
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        printf("Async kernel error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }


    // Copy result from device to host
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Error checking
    float maxError = 0.0f;
    for(int i=0;i<N;i++){
        maxError = max(maxError, abs(y[i] - 4.0f));
    }
    printf("Max Error: %f\n", maxError);

    // Cleaning up
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

}