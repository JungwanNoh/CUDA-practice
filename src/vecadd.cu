#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 1024

__global__ void vecAdd(int* _a, int* _b, int* _c)
{
    int tID = threadIdx.x;

    _c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
    int* a;
    int* b;
    int* c;

    int* d_a;
    int* d_b;
    int* d_c;

    int memSize = sizeof(int) * NUM_DATA;

    // Host memory
    a = new int[NUM_DATA];
    b = new int[NUM_DATA];
    c = new int[NUM_DATA];

    memset(a, 0, memSize);
    memset(b, 0, memSize);
    memset(c, 0, memSize);

    // Initialize
    for (int i = 0; i < NUM_DATA; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // Device memory
    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);

    // Host → Device
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    // Kernel
    vecAdd<<<1, NUM_DATA>>>(d_a, d_b, d_c);

    // Device → Host
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

    // Free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}