// https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F

#include<cuda_runtime.h>
#include<iostream>
#include<vector>
#include<cstdint>


// Implementation of Softmax - Reduction across Threads

// Wrap reduction code - When K is small
template <typename T, int NUM>
__inline__ __device__ T warpReduceMax(T* val, int thread_group_width=32){
    #pragma unroll
    for(int i=0;i<NUM;i++){
        #pragma unroll
        // Combines pairs of threads
        for(int mask=thread_group_width / 2; mask > 0; mask >>= 1){
            // Exchanges values between threads based on XOR of threads lane index and the mask
            val[i] = max(val[i], __shfl_xor_sync(0xffffffff, val[i], mask, thread_group_width));
        }
    }
    return (T)(0.0f);
}

// Vectorized reads, warp level parallelism and reduction operations
template<int cols_per_thread> // Number of column packs each thread is responsible for processing
__global void softmax_stored_locally_multi_dim(const half4* input, half4* output, size_t m, size_t n){ // CUDA kernel that can be called from the host and executed on the device (GPU).
    
    // pack_size = 4, k / 32 = cols_per_thread, num_packs = k/32/4
    constexpr int num_packs = (cols_per_thread + 3) / 4;
    float4 buf[num_packs];

    // global row index that this thread will process. Each block handles 4 rows
    const int m_idx = blockIdx.x * blockDim.y + threadIdx.y; // blockDim.y=4=thread_group_per_block
    const int tid = threadIdx.x; // thread index within the block in the x-dimension

    for(int64_t row=m_idx; row<m; row+=gridDim.x * blockDim.y){

        // Calculates the starting index of the current row in the input and output arrays.
        const int64_t row_offset = row * (n >> 2); // n >> 2: Equivalent to n / 4, since each half4 contains 4 elements.
        const half4* row_x = input + row_offset; // Pointer to the beginning of the current row in the input matrix.
        half4* row_y = output + row_offset; // Pointer to the beginning of the current row in the output matrix.
        float local_max[1] = {-Inf<float>()}; // Array to store the local maximum value for the current row.

        // Loading Data and Computing Local Maximum
        #pragma unroll
        for(int pack_id = 0;pack_id<num_packs;++pack_id){ // pack_id: Index of the current pack being processed by the thread.

            // pack_id * blockDim.x: Base column index for the current pack.
            // + tid: Offset within the pack based on the thread's x-index.
            // col: The global column index this thread processes in the current pack.
            const int col = pack_id * blockDim.x + tid;

            // Data Loading and Maximum Computation:
            if(col < n/4){ // Ensures that the column index does not exceed the matrix dimensions.
                buf[pack_id] = {
                    // Converts each half component (x, y, z, w) of half4 to float for computation.
                    // half2float / __half2float: Functions to convert half to float.
                    half2float(row_x[col].x),
                    __half2float(row_x[col].y),
                    __half2float(row_x[col].z),
                    __half2float(row_x[col].w)
                };
                // Updates the local maximum by comparing the current maximum with the maximum of the four elements in buf[pack_id].
                // Finds the maximum among the four elements of the current pack.
                local_max[0] = max(local_max[0], max(max(buf[pack_id].x, buf[pack_id].y), max(buf[pack_id].z, buf[pack_id].w)));
            } else{ // Handling Out-of-Bounds Columns:
                // If col exceeds the number of columns, initializes the buffer with negative infinity to exclude these elements from further calculations.
                buf[pack_id].x = -Inf<float>();
                buf[pack_id].y = -Inf<float>();
                buf[pack_id].z = -Inf<float>();
                buf[pack_id].w = -Inf<float>();   
            }            
        }
        // Performs a warp-level reduction to find the maximum value in local_max[0] across the warp.
        // After this operation, all threads in the warp have the global maximum value for their assigned columns within the row.
        warpReduceMax<float, 1>(local_max, blockDim.x); // cal the actual max among cols


        // Computing the Exponentials and Local Sum

        // local_sum: Array to store the sum of exponentials for the current row.
        float local_sum[1] = {0.0f};

        // Compute Exponentials: Applies the softmax exponential transformation to each element by subtracting the maximum value for numerical stability
        // Accumulate Sum: Adds the exponentials to local_sum[0]
        #pragma unroll
        for(int i=0;i<num_packs;i++){
            // Exponentials Calculation. Computes the exponential of the shifted value.
            buf[i].x = exp(buf[i].x - local_max[0]);
            buf[i].y = exp(buf[i].y - local_max[0]);
            buf[i].z = exp(buf[i].z - local_max[0]);
            buf[i].w = exp(buf[i].w - local_max[0]);
            // Sum Accumulation. Adds each exponential to local_sum[0].
            local_sum[0] += buf[i].x;
            local_sum[0] += buf[i].y;
            local_sum[0] += buf[i].z;
            local_sum[0] += buf[i].w;
        }
        warpReduceSum<float, 1>(local_sum, blockDim.x); // Warp-Level Reduction to Find the Global Sum

        // Normalizing the Exponentials to Get Softmax Probabilities
        for(int i=0;i<num_packs;i++){
            const int col = i * blockDim.x + tid;

            if(col < n/4){
                row_y[col] {
                    buf[i].x / local_sum[0],
                    buf[i].y / local_sum[0], 
                    buf[i].z / local_sum[0], 
                    buf[i].w / local_sum[0]
                };
            }
        }
    }
}


// Block Reduction - When K is large

// performs a block-wide sum reduction across all threads in a CUDA block. It aggregates values from each thread, ensuring that the final sum is available for use in computations like normalization (eg: Softmax)
template<typename T, int NUM> // Number of elements per thread to be reduced
__inline device T blockReduceSum(T* val){

    // Shared Memory Allocation: A 2D shared memory array to store partial sums from each warp.
    // NUM: Represents multiple elements per thread, supporting scenarios where each thread processes more than one element
    // 33: Accommodates up to 32 warps (assuming a maximum block size of 1024 threads) plus one for safety
    shared T shared[NUM][33];

    // lane: The thread's index within its warp (0-31)
    // Bitwise AND with 0x1f (31 in decimal) effectively computes threadIdx.x % 32
    int lane = threadIdx.x & 0x1f; // threadIdx.x % warp_size

    // wid: The warp's index within the block
    // Bitwise right shift by 5 (equivalent to division by 32)
    int wid = threadIdx.x >> 5; // threadIdx.x / warp_size
    
    // First Reduction: Each warp reduces its assigned subset of values
    // Aggregates the val array across threads in the warp, so each thread in the warp holds the partial sum
    warpReduceSum<T, NUM>(val);

    // Storing Partial Results: Threads with lane == 0 store their warp's partial sum into shared memory.
    // Only the first thread (lane 0) in each warp stores the warp's partial sum into shared memory
    if(lane == 0){
        #pragma unroll
        for(int i=0;i<NUM;i++){ // Stores each element of val into the shared memory array, indexed by NUM and wid
            shared[i][wid] = val[i];
        }
    }
    __syncthreads(); // ensures all partial results are written before proceeding

    // The first warp reads these partial sums and performs a final reduction to obtain the block-wide sum.
    #pragma unroll
    for(int i=0;i<NUM;i++){ // Loading Partial Sums for Final Reduction
        // Threads with threadIdx.x < (blockDim.x / 32) load the corresponding partial sum from shared memory.
        // Other threads set their val[i] to 0.0f to avoid interference in the final reduction
        val[i] = threadIdx.x < (blockDim.x / 32.f) ? shared[i][lane] : (T)(0.0f);
    }
    // The first warp (warp ID 0) performs a final reduction on the partial sums stored in val
    // After this reduction, the first warp holds the total sum across the entire block.
    if(wid==0) warpReduceSum<T, NUM>(val);

    return (T)0.0f;
}

// Softmax Kernel for Block Reduction
// block_size = blockDim.x = 128,256,512,1024
// performs the softmax operation on a matrix stored in half4 format (vector of four half-precision floats). It leverages block-wide reductions to compute the maximum and sum required for the softmax normalization
template<int block_size>
__global void softmax_block_smem_half(
   const half4* input,
   half4* output,
   size_t m,
   const size_t n) {

 const int m_idx = blockIdx.x; // Each block is responsible for a specific row index
 const int tid = threadIdx.x; // Thread's index within the block

 // Declares a shared memory buffer with alignment for float. This buffer is used to store intermediate values during reduction.
 // Declares a dynamically allocated shared memory buffer
 // Ensures that the buffer is aligned to the size of a float
 extern shared align(sizeof(float)) unsigned char shared_buf[];//size_t smem = nsizeof(float)

 // Casts the shared buffer to a float pointer for easy indexing.
 auto buf = reinterpret_cast<float*>(shared_buf);

 // Divides n by 4 (since each half4 contains four elements)
 // Determines how many half4 vectors represent each row
 const int num_packs = n >> 2;

 // Each block processes multiple rows, incrementing by the grid size to cover all rows
 for (int64_t row = m_idx; row < m; row += gridDim.x) {

    // Calculates the starting index for the current row in the input and output arrays
   const int64_t row_offset = row  (n>>2);

   // Points to the input data of the current row.
   const half4* row_x = input + row_offset;
   // Points to the output data of the current row
   half4* row_y = output + row_offset;
   // Initializes an array to store the local maximum for the current row.
   float local_max[1] = {-Inf<float>()};

    // Data Loading and Maximum Calculation Loop
    // Each thread processes multiple packs, spaced by the block size
   for (int pack_id = tid; pack_id < num_packs; pack_id += blockDim.x) {
     const int col = pack_id; //Current column pack index

     // store to local register, which is faster than shared memory
     // Vector Conversion
     // Converts each half component of half4 to float
     // half2float / __half2float: Converts half-precision floats to single-precision
     float4 pack = {
         half2float(row_x[col].x),
         __half2float(row_x[col].y),
         __half2float(row_x[col].z),
         __half2float(row_x[col].w)};

    // Distributes the four elements of float4 across different segments of the shared buffer for parallel processing.
     buf[col] = pack.x;
     buf[num_packs+col] = pack.y;
     buf[2 * num_packs+col] = pack.z;
     buf[3* num_packs+col] = pack.w;
     
     // Updates the local maximum by comparing the existing maximum with the maximum of the four elements in the current pack.
     local_max[0] = max(local_max[0], max(max(pack.x, pack.y), max(pack.z, pack.w)));
   }
   // Performs a block-wide reduction to find the maximum value across all threads.
   blockReduceMax<float, 1>(local_max);//reduce on a block of #blockDim.x
   
   // shared variable to store the block-wide maximum
   __shared float s_max;
   if (threadIdx.x == 0) { // Thread 0 writes the block-wide maximum to shared memory.
     s_max = local_max[0];
   }
   syncthreads(); // Synchronizes threads to ensure s_max is available to all
   
   // Initializes an array to store the local sum of exponentials.
   float local_sum[1] = {0.0f};
   for (int i = tid; i < n; i += blockDim.x) { // Each thread processes multiple elements, spaced by the block size.
     float local_val = exp(buf[i]-s_max); // Computes the exponential of the shifted value for numerical stability.
     buf[i] = local_val; // Stores the exponential back into the shared buffer.
     local_sum[0] += local_val; // Accumulates the sum of exponentials.
   }
   blockReduceSum<float, 1>(local_sum); // Performs a block-wide reduction to compute the total sum of exponentials.
   
   // Storing and Broadcasting Sum
   __shared float s_sum; // shared variable to store the block-wide sum
   if (threadIdx.x == 0) {
     s_sum = local_sum[0]; // Thread 0 writes the block-wide sum to shared memory
   }
   syncthreads(); // Synchronizes threads to ensure s_sum is available to all

   // Each thread processes multiple packs.
   for (int i = tid; i < num_packs; i += blockDim.x) {
     const int col = i; // Current column index
     // Softmax Calculation and Conversion
     // Divides each exponential by the total sum to obtain softmax probabilities
     // Converts the results back to half precision using __float2half_rn (round to nearest)
     row_y[col] = {
       __float2half_rn(buf[i]/s_sum),
       __float2half_rn(buf[num_packs+i]/s_sum),
       __float2half_rn(buf[2*num_packs+i]/s_sum),
       __float2half_rn(buf[3*num_packs+i]/s_sum)};
   }
 }
}

// TODO: 
// Complete this and run
// Profiling ? 

