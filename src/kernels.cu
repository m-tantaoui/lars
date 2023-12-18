#include "kernels.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// These are the inline versions for all of the SDK helper functions
inline void checkCuBLAS(const cublasStatus_t error)
{
    if (cudaSuccess != error)
    {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error, cublasGetStatusString(error));
        exit(1);
    }
}

inline void check(const cudaError_t error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

void cuda_matmul(int m, int n, int k, const float *A, const float *B, float *C)
{

    // Initialize CUDA and cuBLAS
    cublasHandle_t handle;
    checkCuBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 1.0f;

    float *d_A;
    float *d_B;
    float *d_C;

    check(cudaMalloc((void **)&d_A, (m * k) * sizeof(float)));
    check(cudaMalloc((void **)&d_B, (k * n) * sizeof(float)));
    check(cudaMalloc((void **)&d_C, (m * n) * sizeof(float)));

    checkCuBLAS(cublasSetVector((m * k), sizeof(float), A, 1, d_A, 1));
    checkCuBLAS(cublasSetVector((k * n), sizeof(float), B, 1, d_B, 1));
    checkCuBLAS(cublasSetVector((m * n), sizeof(float), C, 1, d_C, 1));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    checkCuBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
    checkCuBLAS(cublasGetVector(m * n, sizeof(float), d_C, 1, C, 1));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy cuBLAS handle
    cublasDestroy(handle);
}