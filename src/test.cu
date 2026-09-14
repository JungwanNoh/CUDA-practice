#include <cstdio>
#include <cuda_runtime.h>

__global__ void helloKernel()
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread %d\n", id);
}

int main()
{
    printf("Hello from CPU\n");

    helloKernel<<<1, 8>>>();

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}