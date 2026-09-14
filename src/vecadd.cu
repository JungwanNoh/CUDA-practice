#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_DATA 134217728


__global__ void vectorAdd(int* a, int* b, int* c)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tId < _size)
        c[tid] = a[tid] + b[tid];
}


int main(void)
{
    int* a;
    int* b;
    int* c;
    int* hc;

    int* d_a;
    int* d_b;
    int* d_c;

    int memSize = sizeof(int) * NUM_DATA;


    // Allocate host memory
    a  = new int[NUM_DATA];
    b  = new int[NUM_DATA];
    c  = new int[NUM_DATA];
    hc = new int[NUM_DATA];


    // Initialize host memory
    memset(a,  0, memSize);
    memset(b,  0, memSize);
    memset(c,  0, memSize);
    memset(hc, 0, memSize);


    // Make input data
    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;

        // CPU result
        hc[i] = a[i] + b[i];
    }


    // Allocate device memory
    cudaMalloc((void**)&d_a, memSize);
    cudaMalloc((void**)&d_b, memSize);
    cudaMalloc((void**)&d_c, memSize);


    // Copy Host -> Device
    cudaMemcpy(
        d_a,
        a,
        memSize,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_b,
        b,
        memSize,
        cudaMemcpyHostToDevice
    );


    // Run GPU kernel
    dim3 dimGrid(ceil((float)NUM_DATA / 256), 1, 1)
    dim3 dimBlock(256,1,1);
    vectorAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, NUM_DATA);


    // Copy Device -> Host
    cudaMemcpy(
        c,
        d_c,
        memSize,
        cudaMemcpyDeviceToHost
    );


    // Check results
    bool result = true;

    for (int i = 0; i < NUM_DATA; i++) {

        if (hc[i] != c[i]) {

            printf(
                "[%d] The result is not matched! (%d, %d)\n",
                i,
                hc[i],
                c[i]
            );

            result = false;
        }
    }


    if (result)
        printf("GPU works well!\n");


    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    // Free host memory
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] hc;


    return 0;
}