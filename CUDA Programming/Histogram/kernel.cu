#include <stdio.h>
#define BLOCK_SIZE 512
#define MAX_BLOCK  16

__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
    __shared__ unsigned int hist_private[4096];

    for (int it = threadIdx.x; it < num_bins; it = it + BLOCK_SIZE)
        hist_private[it] = 0;
    __syncthreads();

    int threads = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (threads < num_elements)
    {
        atomicAdd(&(hist_private[input[threads]]), 1);
        threads = threads + stride;
    }
    __syncthreads();
    
    for (int a = threadIdx.x; a < num_bins; a =a + BLOCK_SIZE)
        atomicAdd(&(bins[a]), hist_private[a]);
}

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {
		 dim3 grid_dim, block_dim;
   		 block_dim.x = BLOCK_SIZE;
		 block_dim.y = 1; 
		 block_dim.z = 1;
    		grid_dim.x = (((num_elements-1)/BLOCK_SIZE+1) > MAX_BLOCK ? MAX_BLOCK : (num_elements-1)/BLOCK_SIZE+1);
    		grid_dim.y = 1; 
		grid_dim.z = 1;

    		histo_kernel<<<grid_dim, block_dim>>>(input, bins, num_elements, num_bins);

}


