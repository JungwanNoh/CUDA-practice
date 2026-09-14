#include <stdio.h>
#include <cuda_runtime.h>

__global__ void showThread()
{
    printf("threadIdx.x = %d\n", threadIdx.x);
}

int main()
{
    showThread<<<1, 256>>>();

    cudaDeviceSynchronize();

    return 0;
}