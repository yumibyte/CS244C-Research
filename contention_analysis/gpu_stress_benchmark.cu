#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>

void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(result));
        exit(1);
    }
}

void checkCublas(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %s\n", msg);
        exit(1);
    }
}

int main() {
    float *A, *B, *C;
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    // Allocate matrices on GPU
    checkCuda(cudaMalloc((void**)&A, N*N*sizeof(float)), "cudaMalloc A");
    checkCuda(cudaMalloc((void**)&B, N*N*sizeof(float)), "cudaMalloc B");
    checkCuda(cudaMalloc((void**)&C, N*N*sizeof(float)), "cudaMalloc C");

    float alpha = 1.0f, beta = 0.0f;

    printf("Starting GPU stress benchmark: infinite matrix multiplication of %dx%d\n", N, N);
    int i = 0;
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, A, N, B, N, &beta, C, N), "cublasSgemm");
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (i % 10 == 0) {
            printf("Iteration %d: %.2f ms\n", i, ms);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++i;
    }
    // Never reached, but cleanup for completeness
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);
    return 0;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);
    return 0;
}