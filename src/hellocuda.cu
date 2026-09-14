#include <stdio.h>
#include <cuda_runtime.h>

__global__ void showID()
{
    printf(
        "block=%d thread=%d\n",
        blockIdx.x,
        threadIdx.x
    );
}

int main()
{
    showID<<<3, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}