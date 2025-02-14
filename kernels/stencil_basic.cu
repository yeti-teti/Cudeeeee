#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

// Basic kernel that performs a stencil sweep

__global__ void stencil_kernel(float* in, float* out, unsigned int N){

    // Each thread block is responsible for calculating a tile of output grid values and that each thread is assigned to one of the output grid points.
    // blockIdx.x, blockIdx.y, blockIdx.z: The index of the current block in a 3D grid.
    // threadIdx.x, threadIdx.y, threadIdx.z: The index of the current thread within its block in a 3D thread block.
    // blockDim.x, blockDim.y, blockDim.z: The dimensions (in number of threads) of a 3D thread block.
    // Gives each thread a unique (i, j, k) coordinate for processing a cell in a 3D array of size N*N*N Each thread corresponds to one element (or “cell”) in the input and output arrays.
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread has been assigned to a 3D grid point, the input values at that grid point and all neighboring grid points are multiplied by different coefficients and added.
    // Bounday checks for stencil computation
    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1){
        // Flattening 3D indices into 1D
        // A 3D array arr[i][j][k] in row‐major ordering can be flattened into 1D as index=(i×N×N)+(j×N)+(k)
        // i is the “slowest‐varying” index (often treated like a z‐coordinate)
        // k is the “fastest‐varying” index (often treated like an x‐coordinate
        // j is in between (often treated like a y‐coordinate)
        out[i*N*N + j*N + k] = c0 * in[i*N*N + j*N + k] + c1 * in[i*N*N + j*N + (k - 1)] + c2 * in[i*N*N + j*N + (k + 1)] + c3 * in[i*N*N + (j - 1)*N + k] + c4 * in[i*N*N + (j + 1)*N + k] + c5 * in[(i - 1)*N*N + j*N + k] + c6 * in[(i + 1)*N*N + j*N + k];
    }
}





