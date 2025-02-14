#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void stencil_kernel(float* in, float* out, unsigned int N){

    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    // Uses registers for inPrev, inCurr, and inNext, which are faster than shared memory
    float inPrev;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr;
    float inNext;

    // Loads the first i-1 layer into the register inPrev before entering the loop
    if(iStart - 1 >=0 && iStart - 1 < N && j >=0 && j < N && k >= 0 && k < N){
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }
    // Loads i into register inCurr and stores it in shared memory (inCurr_s)
    // Shared memory only holds the i layer, not the entire 3D tile
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr = in[iStart * N * N + j * N + k]; 
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    // The loop processes the z-dimension (i) one layer at a time
    for(int i=iStart;i<iStart + OUT_TILE_DIM;i++){
        // Only the i+1 layer is loaded per iteration, avoiding wasted memory transfers
        if(i + 1 >=0 && i + 1 < N && j >= 0 && j < N && k >=0 && k < N){
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
               threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                // Reads from registers (inPrev, inCurr, inNext) instead of shared memory whenever possible.
                // Uses shared memory (inCurr_s) only for in-plane accesses (j, k)
                // Minimizes redundant memory accesses
                out[i*N*N + j*N + k] =
                    c0 * inCurr
                  + c1 * inCurr_s[threadIdx.y][threadIdx.x-1]
                  + c2 * inCurr_s[threadIdx.y][threadIdx.x+1]
                  + c3 * inCurr_s[threadIdx.y+1][threadIdx.x]
                  + c4 * inCurr_s[threadIdx.y-1][threadIdx.x]
                  + c5 * inPrev
                  + c6 * inNext;
            }
        }
        __syncthreads();
        // Instead of reloading from global memory, shift registers
        // Only inCurr_s (shared memory) is updated per iteration, keeping memory bandwidth low
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;       
    }
}