#include <iostream>
#include "kernels.h"

using namespace std;

extern "C"
{

    void matmul(int m, int n, int k, const float *A, const float *B, float *C)
    {
        cuda_matmul(m, n, k, A, B, C);
    }
}