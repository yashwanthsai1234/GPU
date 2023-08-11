#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

   	__shared__ float matA[TILE_SIZE][TILE_SIZE];
	__shared__ float matB[TILE_SIZE][TILE_SIZE];

	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;
	int a = thread_y + block_y* blockDim.y;

	int b = thread_x + (block_x * blockDim.x);
	float pvalue = 0;
	
	for(int i = 0; i < (TILE_SIZE + k-1)/TILE_SIZE ; i++){
		if(a < m && i * TILE_SIZE + thread_x < k)
			matA[thread_y][thread_x] = A[thread_x + (a * k) + (i * TILE_SIZE)];
		else
			matA[thread_y][thread_x] = 0.0;
		if((thread_y + (TILE_SIZE * i)) < k && b < n)
			matB[thread_y][thread_x] = B[((i * TILE_SIZE + thread_y) * n) + b];
		else
			matB[thread_y][thread_x] = 0.0;
		__syncthreads();
		if(b < n && a < m){
			for (int it = 0; it < TILE_SIZE; ++it)
				pvalue += matA[thread_y][it] * matB[it][thread_x];
			
			
		}
		__syncthreads();
	
	}
	if(a < m && b < n)
		C[((block_y*blockDim.y + thread_y)*n) + (block_x*blockDim.x) + thread_x] = pvalue;
	
	
	 
  
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE,1);	
	dim3 grid_size((n-1) / BLOCK_SIZE+1 , (m-1) / BLOCK_SIZE+1,1);

	mysgemm<<<grid_size, block_size>>>(m, n, k, A, B, C);
}

