#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);

    printf("GPU Name : %s\n", prop.name);
    printf("SM Count : %d\n", prop.multiProcessorCount);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM   : %d\n", prop.maxThreadsPerMultiProcessor);

    return 0;
}