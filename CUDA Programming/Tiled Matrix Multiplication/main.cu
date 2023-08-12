#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

int main(int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...");
    fflush(stdout);
    startTime(&timer);

    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;


    if (argc == 1)
    {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    }
    else if (argc == 2)
    {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    }
    else if (argc == 4)
    {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    }
    else
    {
        printf("\n    Invalid input parameters!"
               "\n    Usage: ./sgemm-tiled                # All matrices are 1000 x 1000"
               "\n    Usage: ./sgemm-tiled <m>            # All matrices are m x m"
               "\n    Usage: ./sgemm-tiled <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
               "\n");
        exit(0);
    }

    A_sz = matArow * matAcol;
    B_sz = matBrow * matBcol;
    C_sz = matArow * matBcol;

    //=========================== Unified memory=============================

    float *X, *Y, *Z;
    cudaMallocManaged(&X, A_sz * sizeof(float));
    cudaMallocManaged(&Y, B_sz * sizeof(float));
    cudaMallocManaged(&Z, C_sz * sizeof(float));
  for (unsigned int t = 0; t < A_sz; t++)
    {
        X[t] = (rand() % 100) / 100.00;
    }
    for (unsigned int m = 0; m < B_sz; m++)
    {
        Y[m] = (rand() % 100) / 100.00;
    }



    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol, matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables...");
    fflush(stdout);
    startTime(&timer);

    //CUDA STREAMS CREATION
    cudaStream_t x;
    cudaStream_t y;
    cudaStreamCreate(&x);
    cudaStreamCreate(&y);

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel...");
    fflush(stdout);
    startTime(&timer);
    //=============================CUDA STREAMS ===============================

    int gpu = -1;
    cudaGetDevice(&gpu);

    cudaMemAdvise(&X, A_sz * sizeof(float), cudaMemAdviseSetReadMostly, gpu);
    cudaMemAdvise(&Y, B_sz * sizeof(float), cudaMemAdviseSetReadMostly, gpu);
    cudaMemPrefetchAsync(X, A_sz * sizeof(float), gpu, x);
    cudaMemPrefetchAsync(Y, B_sz * sizeof(float), gpu, x);

    basicSgemm(matArow, matBcol, matBrow, X, Y, Z);

    cudaMemPrefetchAsync(Z, C_sz * sizeof(float), -1,y);

    cudaStreamSynchronize(x);
    cudaStreamSynchronize(y);

    //=========================================================================

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
        printf("Unable to launch kernel");
    startTime(&timer);

    // Copy device variables from host ----------------------------------------
    printf("\nCopying data from device to host...");fflush(stdout);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

   ///=========================== Unified memory=============================

    // Verify correctness -----------------------------------------------------

    printf("Verifying results...");
    fflush(stdout);
    verify(X, Y, Z, matArow, matAcol, matBcol);

    //=========================================================================

    // Free memory ------------------------------------------------------------


    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    //=========================================================================
    /*************************************************************************/

    cudaStreamDestroy(x);
    cudaStreamDestroy(y);

    return 0;
}