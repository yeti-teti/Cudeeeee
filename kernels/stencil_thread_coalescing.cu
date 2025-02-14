#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

// kernel with thread coarsening in the z direction for a 3D seven-point stencil sweep
// The idea is for the thread block to iterate in the z direction, calculating the values of grid points in one x-y plane of the output tile during each iteration.
// Each inPrev_s and inNext_s element is used by only one thread in the calculation of the output tile grid point with the same x-y indices.
// Only the inCurr_s elements are accessed by multiple threads and truly need to be in the shared memory.
// The z neighbors in inPrev_s and inNext_s can instead stay in the registers of the single user thread
__global__ void stencil_kernel(float* in, float* out, unsigned int N){

    // iStart is the base index for the z-dimension of the current tile
    int iStart = blockIdx.z * OUT_TILE_DIM;
    // Each thread is responsible for a point (j, k) in a 2D plane and iterates through the third dimension (i) in a loop
    // kernel first assigns each thread to a grid point in an x-y plane of the output
    // The -1 in indexing accounts for halo cells (extra padding to include neighboring values for stencil computation)
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    // Instead of storing an entire 3D tile, it only maintains three 2D slices (inPrev_s, inCurr_s, inNext_s).
    // This significantly reduces shared memory usage since we only need three layers of the 3D array at a time
    // inPrev_s stores the previous layer (i-1)
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    // inCurr_s stores the current layer (i)
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    // inNext_s stores the next layer (i+1)
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

    // each block needs to load into the shared memory the three input tile planes that contain all the points that are needed to calculate the values of the output tile plane
    // First layer
    // The previous layer is loaded only once at the start.
    // This ensures that inPrev_s is prefilled for the first iteration in the for loop
    if(iStart - 1 >=0 && iStart - 1 < N && j >=0 && j < N && k >= 0 && k < N){
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }
    // The first "current" layer is preloaded before looping through subsequent layers
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }

    // During the first iteration, all threads in a block collaborate to load the third layer needed for the current output tile layer into the shared memory array inNext_s
    // i is the z index of the output tile grid point calculated by each thread. The loop iterates through OUT_TILE_DIM layers in the z-dimension (i)
    // During each iteration, all threads in a block will be processing an x-y plane of an output tile; thus they will all be calculating output grid points whose z indices are identical.
    for(int i=iStart;i<iStart + OUT_TILE_DIM;i++){
        // It fetches the next layer (i+1) in shared memory so that all three slices (inPrev_s, inCurr_s, and inNext_s) are always available
        if(i + 1 >=0 && i + 1 < N && j >= 0 && j < N && k >=0 && k < N){
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        // Ensures that all threads have loaded data into shared memory before starting computations
        // Prevents race conditions where some threads compute before others have finished loading
        __syncthreads();

        // Only valid grid points (i, j, k) within bounds perform computations (avoiding out-of-bounds accesses)
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            
            // Each thread computes a 7-point stencil using its own current layer (inCurr_s) and two adjacent layers (inPrev_s and inNext_s)
            // Neighbors are read from shared memory, making the computation very efficient
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
               threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                // Flattened 3D indexing (i*N*N + j*N + k) is used to write the result back to global memory
                out[i*N*N + j*N + k] =
                    c0 * inCurr_s[threadIdx.y][threadIdx.x]
                  + c1 * inCurr_s[threadIdx.y][threadIdx.x-1]
                  + c2 * inCurr_s[threadIdx.y][threadIdx.x+1]
                  + c3 * inCurr_s[threadIdx.y-1][threadIdx.x]
                  + c4 * inCurr_s[threadIdx.y+1][threadIdx.x]
                  + c5 * inPrev_s[threadIdx.y][threadIdx.x]
                  + c6 * inNext_s[threadIdx.y][threadIdx.x];
            }
        } 
        __syncthreads();
        // Instead of reloading everything from global memory, we shift shared memory buffers
        // The next iteration will load only inNext_s from global memory, significantly reducing memory traffic
        // This overlapping strategy is known as temporal blocking
        // This is because the roles that are played by the input tile planes change when the threads move by one output plane in the z direction
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }    
}