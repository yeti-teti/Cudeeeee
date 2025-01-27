// 2D convolution kernel
// PMPP Book

#include<iostream>
#include<cuda_runtime.h>


// ----------------------------------------------------------------------------------------------------------------------------------------------------// 

// Naive Kernel
// Parameters: Pointer to the input array (N), Pointer to the filter (F), Pointer ot the output array (P), Radius of the sqaure filter (r), width of the input and input and output arrays, height input and output arrays.
__global__ void convolution_2D_basic_kernel(float* N, float* F, float* P, int radius, int width, int height){

    // Mapping of threads to output elements
    // Calculating the output element indices
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    // The register variable Pvalue will accumulate all intermediate results to save DRAM bandwidth
    float Pvalue = 0.0f;

    // Doubly nested loop to iterate through all these index values and perform the calculation
    for(int fRow = 0;fRow < 2*radius+1;fRow++){
        for(int fCol = 0;fCol<2*radius+1;fCol++){
            // Input N elements that are needed for calculating the output element.
            // For all threads, outCol - r and outRow - r define the upper-left corner of the patch of input elements needed for P[outRow][outCol]
            int inRow = outRow - radius + fRow;
            int inCol = outCol - radius + fCol;

            // Checks whether any of the input N elements that are used are ghost cells on the left, right, top, or bottom side of the N array.
            // 0 values will be used for ghost cells, we can simply skip the multiplication and accumulation of the ghost cell element and its corresponding filter element
            if( inRow >=0 && inRow < height && inCol >=0 && inCol <width ){
                // The ratio of floating-point arithmetic calculation to global memory accesses is only about 0.25 OP/B (2 operations for every 8 bytes loaded)
                Pvalue += F[fRow][fCol] * N[inRow * width + inCol];
            }

        }
    }
    // We release the Pvalue into the output P element
    P[outRow][outCol] = Pvalue;
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------// 


// Declare an F array in constant memory, the host code declares it a global variable as follows:
// global variable declaration and should be outside any function in the source file
// The keyword __constant__ (two underscores on each side) tells the compiler that array F should be placed into the device constant memory
#define FILTER_RADIUS 2
__constant__ float F[2*FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// After Host code allocates and initializes the mask in a filter F_h array in the host memopry with (2*FILTER_RADIUS + 1) 2 elements
// The contents of the F_h can be transferred from the host memory to F in the device constant memory
// this is a special memory copy function that informs the CUDA runtime that the data being copied into the constant memory will not be changed during kernel execution
// cudaMemcpyToSymbol(F, F_h, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float))

// Const Mem Kernel
// Kernel functions access constant memory variables as global variables. Therefore their pointers do not need to be passed to the kernel as arguments
// Almost same as above kernel but F is no longer accessed through a pointer that is passed in as a parameter. It is now accessed as a global variable
__global__ void convolution_2D_const_mem_kernel(float* N, float* P, int radius, int width, int height){

    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;

    for(int fRow=0;fRow<2*radius+1;fRow++){
        for(int fCol=0;fCol<2*radius+1;fCol++){

            int inRow = outRow - radius + fRow;
            int inCol = outCol - radius + fCol;

            if( inRow >=0 && inRow < height && inCol >=0 && inCol < weight ){
                // With the use of constant memory and caching, we have effectively doubled the ratio of floating-point arithmetic to memory access to around 0.5 OP/B (2 operations for every 4 bytes)
                Pvalue += F[fRow][fCol] * N[inRow * width + inCol];
            }
        }
    }

    P[outRow * width + outCol] = Pvalue;
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------//

// Tiled convolution algorithms in which all threads in a block first collaboratively load the input tile into the shared memory before they calculate the elements of the output tile by accessing the input elements from the shared memory.
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ( (IN_TILE_DIM) -  2 * (FILTER_RADIUS) )

__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// Launches thread blocks whose dimension matches that of the input tiles
__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float* P, int width, int height){

    // Each thread calculates the column index (col) and row index (row) of the input or output elements that it is responsible for loading or computing
    // blockIdx.x*OUT_TILE_DIM and blockIdx.y*OUT_TILE_DIM are the horizontal and vertical P array indices, respectively, of the beginning of the output tile assigned to the block
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Loading input tile
    // Kernel allocates a shared memory array N_s whose size is the same as an input tile and loads the input tile to the shared memory array
    __shared__ N_s[IN_TILE_DIM][IN_TILE_DIM];
    // check whether the input tile element that a thread is attempting to load is a ghost cell
    if( row >= 0 && row < height && col >=0 && col < width ){
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    }else{
        N_s[threadIdx.y][threadIdx.y] = 0.0;
    }

    // Each thread performs barrier sync to ensure that the entire input tile is in place in the shared memory before any thread is allowed to proceed with the calcualtion of output elements.
    __syncthreads();
    // After all the threads are in the N_ds array each thread can calculate their output P element value using the N_ds elements

    // Calculating the output elements
    // threadIdx.x-r and threadIdx.y-r give the offset into the tile
    // design that deactivates FILTER_RADIUS exterior layers of threads
    // Active thread (tx, ty) will calculate output element (tx - FILTER_RADIUS, ty - FILTER_RADIUS) using a patch of input tile elements whose upper-left corner is element (tx - FILTER_RADIUS, ty - FILTER_RADIUS) of the input tile
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // Output tile is smaller than the input tile and that the blocks are of the same size as the input tiles, so only a subset of the threads in each block will be used to calculate the output tile elements
    // Turning off the threads at the edges of the block
    if( row >=0 && row < height && col >= 0 && col < width ){

        if( tileCol >=0 && tileCol < OUT_TILE_DIM && tileRow >=0 && tileRow < OUT_TILE_DIM{

            // iterates through the patch and generates the output elements.

            float Pvalue = 0.0f;
            for(int fRow=0;fRow < 2 * FILTER_RADIUS + 1; fRow++){
                for(int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++){
                    
                    //  arithmetic-to-global memory access ratio
                    // For our example with a 5 × 5 filter and 32 × 32 input tiles (28 × 28 output tiles), the ratio is 9.57 OP/B
                    Pvalue += F[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];      
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------//

// tiled convolution algorithm that uses the same dimension for input and output tiles and loads only the internal elements of each tile into the shared memory
// 2D convolution kernel using caching for halo cells
// Input tiles and output tiles are of the same dimension
// input tiles and output tiles are of the same size, the thread blocks can be launched with the same size of the input/output tiles
#define TILE_DIM 32
__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height){

    // Loading of the N_s elements becomes simpler, since each thread can simply load the input element that has the same x and y coordinates as its assigned output element

    int col = blockIdx.y * TILE_DIM + threadIdx.y;
    int row = blockIdx.x * TILE_DIM + threadIdx.x;

    // Loading input tile
    // the shared memory N_ds array needs to hold only the internal elements of the tile.
    __shared__ N_s[TILE_DIM][TILE_DIM];
    if( row < height && col < width ){
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    }else{
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Calcualting output elements
    // Turning off the threads at the edges of the block
    // check only for the usual boundary condition that a tile may extend beyond the valid range of the input data
    if( row < height && col < width ){

        float Pvalue = 0.0f;

        for(int fRow = 0;fRow < 2 * FILTER_RADIUS + 1; fRow++){
            for(int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++){

                // Conditions to check for use of both halo cells and ghost cells.

                // Handling of halo cells. Tests whether the input element falls within the interior of the input tile. If so, the element is accessed from the shared memory.
                if( threadIdx.x - FILTER_RADIUS + fCol >=0 && threadIdx - FILTER_RADIUS + fCol < TILE_DIM && threadIdx.y - FILTER_RADIUS + fRow >=0 && threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM){

                    Pvalue += F[fRow][fCol] * N_s[threadIdx.y + fRow][threadIdx.x + fCol];
                } else{
                    // Handling of ghost cells. Tests whether the input element falls within the interior of the input tile. If not, the conditions check whether the halo cells are ghost cells. If so, no action is taken for the element, since we assume that the ghost values are 0. Otherwise, the element is accessed from the global memory
                    if( row - FILTER_RADIUS + fRow >=0 && row - FILTER_RADIUS + fRow < height && col - FILTER_RADIUS + fCol >=0 && col - FILTER_RADIUS + fCol < width){
                        Pvalue += F[fRow][fCol] * N[( row - FILTER_RADIUS + fRow ) * width + (col - FILTER_RADIUS + fCol)];
                    }
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}