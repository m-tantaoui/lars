#include "kernels.h"
#include <cuda_runtime.h>

inline void check(const cudaError_t error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

void cuda_dot(float *x, float *y, int n)
{

    float *d_x, *d_y;
    cudaError_t err;

    check(cudaMalloc(&d_x, n * sizeof(float)));
    cudaMalloc(&d_y, n * sizeof(float));

    err = cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("WTF, Why ??? \n");
    }

    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(n + 255) / 256, 256>>>(n, 10.0f, d_x, d_y);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        printf("%f\n", y[i]);

    cudaFree(d_x);
    cudaFree(d_y);
    // free(x);
    // free(y);
}