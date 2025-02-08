#include "Nvidia/helper_math.h"

extern "C" {
  __global__ void threshold_f(float *heightmap, int *out, int width, int height, float threshold) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
      return;

    int i = x + y * width;
    out[i] = (int) (heightmap[i] > threshold);
  }
}
