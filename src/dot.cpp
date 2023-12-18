#include <iostream>
#include "kernels.h"

using namespace std;

extern "C"
{
    float dot(float *x, float *y, int N)
    {
        float res = 0.0;
        cuda_dot(x, y, N);
        return res;
    }
}