#include <stdio.h>

#define TILE_SIZE 16

__global__ void matAdd(int dim, const float *A, const float *B, float* C) {

    int x_axis; int y_axis;
    y_axis =(blockIdx.y*blockDim.y) + threadIdx.y;
    x_axis =(blockIdx.x*blockDim.x) + threadIdx.x;
    										
    if ( y_axis < dim && x_axis < dim)
    {
       C[x_axis + (y_axis * dim)]=A[x_axis + (y_axis * dim)]+B[x_axis + (y_axis * dim)];
    }



};

void basicMatAdd(int dim, const float *A, const float *B, float *C)
{
    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
  
    dim3 Grid_of_blocks(BLOCK_SIZE, BLOCK_SIZE);
  
    dim3 Grid_size((int)(dim-1)/Grid_of_blocks.x+1,(int)(dim-1)/Grid_of_blocks.y+1);


  
    matAdd<<<Grid_size,Grid_of_blocks>>>(dim, A, B, C);

}

