#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_DATA 1024


// GPU Kernel
__global__ void vectorAdd(const int* A, const int* B, int* C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NUM_DATA)
    {
        C[idx] = A[idx] + B[idx];
    }
}


int main(void)
{
    // ========================================
    // 1. CPU(Host) 메모리 할당
    // ========================================

    int* h_A = (int*)malloc(NUM_DATA * sizeof(int));
    int* h_B = (int*)malloc(NUM_DATA * sizeof(int));
    int* h_C = (int*)malloc(NUM_DATA * sizeof(int));

    // 데이터 초기화
    for (int i = 0; i < NUM_DATA; i++)
    {
        h_A[i] = i;
        h_B[i] = i * 2;
    }


    // ========================================
    // 2. GPU(Device) 메모리 할당
    // ========================================

    int* d_A;
    int* d_B;
    int* d_C;

    cudaMalloc((void**)&d_A, NUM_DATA * sizeof(int));
    cudaMalloc((void**)&d_B, NUM_DATA * sizeof(int));
    cudaMalloc((void**)&d_C, NUM_DATA * sizeof(int));


    // ========================================
    // 3. Host → Device
    // ========================================

    cudaMemcpy(
        d_A,
        h_A,
        NUM_DATA * sizeof(int),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        NUM_DATA * sizeof(int),
        cudaMemcpyHostToDevice
    );


    // ========================================
    // 4. Kernel 실행
    // ========================================

    int threadsPerBlock = 256;
    int numBlocks =
        (NUM_DATA + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();


    // ========================================
    // 5. Device → Host
    // ========================================

    cudaMemcpy(
        h_C,
        d_C,
        NUM_DATA * sizeof(int),
        cudaMemcpyDeviceToHost
    );


    // ========================================
    // 6. 결과 확인
    // ========================================

    for (int i = 0; i < 10; i++)
    {
        printf(
            "%d + %d = %d\n",
            h_A[i],
            h_B[i],
            h_C[i]
        );
    }


    // ========================================
    // 7. 메모리 해제
    // ========================================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}