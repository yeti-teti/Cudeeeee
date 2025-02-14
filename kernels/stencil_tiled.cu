
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void stencil_kernel(float* in, float* out, unsigned int N){

    // Thread Indexing and Global Memory Access
    // Each thread block works on a tile of size OUT_TILE_DIM³ within the full N × N × N grid
    // The indices (i, j, k) map the thread's location in the global memory space.
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    // Shared memory (in_s) is used instead of global memory to store the input tile
    // This reduces expensive global memory accesses, which significantly improves performance
    // The size is IN_TILE_DIM³, where IN_TILE_DIM = OUT_TILE_DIM + 2 (padding for halo cells)
    __shared__ float n_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // Threads cooperatively load their corresponding values from global memory into in_s
    // Boundary checking (i >= 0 && i < N) ensures that threads outside the array don't load invalid data
    // Each thread only loads one value, leading to coalesced memory access
    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N){
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    }
    
    // Ensures that all threads in the block finish loading before the stencil operation begins
    // Without this, some threads might read from in_s before others have finished writing, causing race conditions
    __syncthreads();

    // Conditions ensures we only compute for valid cells (excluding the boundary)
    if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {

        // condition further restricts computations inside shared memory boundaries
        // Ensures each thread only accesses valid shared memory indices (excluding the outer halo)
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 &&
           threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM-1 &&
           threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM-1) {
            out[i*N*N + j*N + k] =
                c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x]
              + c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x-1]
              + c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x+1]
              + c3 * in_s[threadIdx.z][threadIdx.y-1][threadIdx.x]
              + c4 * in_s[threadIdx.z][threadIdx.y+1][threadIdx.x]
              + c5 * in_s[threadIdx.z-1][threadIdx.y][threadIdx.x]
              + c6 * in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
        }
    }
    

}

// blocks are of the same size as input tiles and some of the threads are turned off in calculating output grid point values

// Like tiled convolution kernel, the tiled stencil sweep kernel first calculates the beginning x, y, and z coordinates of the input patch that is used for each thread

// value 1 that is subtracted in each expression is because the kernel assumes a 3D seven-point stencil with one grid point on each sides.
// value that is subtracted should be the order of the stencil