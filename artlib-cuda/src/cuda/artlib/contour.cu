#include "Nvidia/helper_math.h"

extern "C" {
__global__ void threshold_f(float *heightmap, int *out, int width, int height,
                            float threshold) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height)
    return;

  int i = x + y * width;
  out[i] = (int)(heightmap[i] > threshold);
}

// Re-map number from one range to [0.0, 1.0].
// https://docs.arduino.cc/language-reference/en/functions/math/map/
__device__ inline float map(float x, float in_min, float in_max) {
  return (x - in_min) / (in_max - in_min);
}

__global__ void calculate_line_segments(float *heightmap, float *out,
                                        int cells_x, int cells_y,
                                        float threshold) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= cells_x || y >= cells_y)
    return;
  int i = x + y * cells_x;

  // because cells are 2x2, the heightmap is 1 unit wider and higher than
  //  the cells we operate on
  int heightmap_width = cells_x + 1;

  // one cell is a collection of 2x2 pixels, indexed as the following:
  //  - a -> top left
  //  - b -> top right
  //  - c -> bottom left
  //  - d -> bottom right
  float a = heightmap[x + y * heightmap_width];
  float b = heightmap[x + 1 + y * heightmap_width];
  float c = heightmap[x + (y + 1) * heightmap_width];
  float d = heightmap[x + 1 + (y + 1) * heightmap_width];
  float m = (a + b + c + d) / 4.0;
  int configuration = (((int)(a < threshold)) << 3) |
                      (((int)(b < threshold)) << 2) |
                      (((int)(c < threshold)) << 1) | ((int)(d < threshold));

  // TODO: DEBUG symbol in constant memory to flag logging
  // printf("config: %d cell: %dx%d index: %d\n", configuration, x, y, i);
  // printf("values a, b, c, d are: %.2f %.2f %.2f %.2f\n", a, b, c, d);

  switch (configuration) {
  case 1:
    out[i * 8 + 0] = (float)(x + 1) + 0.5;
    out[i * 8 + 1] = map(threshold, b, d) + (float)y + 0.5;
    out[i * 8 + 2] = map(threshold, c, d) + (float)x + 0.5;
    out[i * 8 + 3] = (float)(y + 1) + 0.5;
    break;
  case 2:
    out[i * 8 + 0] = map(threshold, c, d) + (float)x + 0.5;
    out[i * 8 + 1] = (float)(y + 1) + 0.5;
    out[i * 8 + 2] = (float)x + 0.5;
    out[i * 8 + 3] = map(threshold, a, c) + (float)y + 0.5;
    break;
  case 3:
    out[i * 8 + 0] = (float)(x + 1) + 0.5;
    out[i * 8 + 1] = map(threshold, a, d) + (float)y + 0.5;
    out[i * 8 + 2] = (float)x + 0.5;
    out[i * 8 + 3] = map(threshold, b, c) + (float)y + 0.5;
    break;
  case 4:
    out[i * 8 + 0] = map(threshold, a, b) + (float)x + 0.5;
    out[i * 8 + 1] = (float)y + 0.5;
    out[i * 8 + 2] = (float)(x + 1) + 0.5;
    out[i * 8 + 3] = map(threshold, b, d) + (float)y + 0.5;
    break;
  case 5:
    out[i * 8 + 0] = map(threshold, a, b) + (float)x + 0.5;
    out[i * 8 + 1] = (float)y + 0.5;
    out[i * 8 + 2] = map(threshold, c, d) + (float)x + 0.5;
    out[i * 8 + 3] = (float)(y + 1) + 0.5;
    break;
  case 6:
    if (m < threshold) {
      // same as 4
      out[i * 8 + 0] = map(threshold, a, b) + (float)x + 0.5;
      out[i * 8 + 1] = (float)y + 0.5;
      out[i * 8 + 2] = (float)(x + 1) + 0.5;
      out[i * 8 + 3] = map(threshold, b, d) + (float)y + 0.5;

      // same as 2
      out[i * 8 + 4] = map(threshold, c, d) + (float)x + 0.5;
      out[i * 8 + 5] = (float)(y + 1) + 0.5;
      out[i * 8 + 6] = (float)x + 0.5;
      out[i * 8 + 7] = map(threshold, a, c) + (float)y + 0.5;
    } else {
      // same as 7
      out[i * 8 + 0] = map(threshold, a, b) + (float)x + 0.5;
      out[i * 8 + 1] = (float)y + 0.5;
      out[i * 8 + 2] = (float)x + 0.5;
      out[i * 8 + 3] = map(threshold, a, c) + (float)y + 0.5;

      // same as 14
      out[i * 8 + 4] = map(threshold, c, d) + (float)x + 0.5;
      out[i * 8 + 5] = (float)(y + 1) + 0.5;
      out[i * 8 + 6] = (float)(x + 1) + 0.5;
      out[i * 8 + 7] = map(threshold, b, d) + (float)y + 0.5;
    }
    break;
  case 7:
    out[i * 8 + 0] = map(threshold, a, b) + (float)x + 0.5;
    out[i * 8 + 1] = (float)y + 0.5;
    out[i * 8 + 2] = (float)x + 0.5;
    out[i * 8 + 3] = map(threshold, a, c) + (float)y + 0.5;
    break;
  case 8:
    out[i * 8 + 0] = (float)x + 0.5;
    out[i * 8 + 1] = map(threshold, a, c) + (float)y + 0.5;
    out[i * 8 + 2] = map(threshold, a, b) + (float)x + 0.5;
    out[i * 8 + 3] = (float)y + 0.5;
    break;
  case 9:
    if (m < threshold) {
      // same as 8
      out[i * 8 + 0] = (float)x + 0.5;
      out[i * 8 + 1] = map(threshold, a, c) + (float)y + 0.5;
      out[i * 8 + 2] = map(threshold, a, b) + (float)x + 0.5;
      out[i * 8 + 3] = (float)y + 0.5;

      // same as 1
      out[i * 8 + 4] = (float)(x + 1) + 0.5;
      out[i * 8 + 5] = map(threshold, b, d) + (float)y + 0.5;
      out[i * 8 + 6] = map(threshold, c, d) + (float)x + 0.5;
      out[i * 8 + 7] = (float)(y + 1) + 0.5;
    } else {
      // same as 11
      out[i * 8 + 0] = (float)(x + 1) + 0.5;
      out[i * 8 + 1] = map(threshold, b, d) + (float)y + 0.5;
      out[i * 8 + 2] = map(threshold, a, b) + (float)x + 0.5;
      out[i * 8 + 3] = (float)y + 0.5;

      // same as 13
      out[i * 8 + 4] = (float)x + 0.5;
      out[i * 8 + 5] = map(threshold, a, c) + (float)y + 0.5;
      out[i * 8 + 6] = map(threshold, c, d) + (float)x + 0.5;
      out[i * 8 + 7] = (float)(y + 1) + 0.5;
    }
    break;
  case 10:
    out[i * 8 + 0] = map(threshold, c, d) + (float)x + 0.5;
    out[i * 8 + 1] = (float)(y + 1) + 0.5;
    out[i * 8 + 2] = map(threshold, a, b) + (float)x + 0.5;
    out[i * 8 + 3] = (float)y + 0.5;
    break;
  case 11:
    out[i * 8 + 0] = (float)(x + 1) + 0.5;
    out[i * 8 + 1] = map(threshold, b, d) + (float)y + 0.5;
    out[i * 8 + 2] = map(threshold, a, b) + (float)x + 0.5;
    out[i * 8 + 3] = (float)y + 0.5;
    break;
  case 12:
    out[i * 8 + 0] = (float)x + 0.5;
    out[i * 8 + 1] = map(threshold, a, c) + (float)y + 0.5;
    out[i * 8 + 2] = (float)(x + 1) + 0.5;
    out[i * 8 + 3] = map(threshold, b, d) + (float)y + 0.5;
    break;
  case 13:
    out[i * 8 + 0] = (float)x + 0.5;
    out[i * 8 + 1] = map(threshold, a, c) + (float)y + 0.5;
    out[i * 8 + 2] = map(threshold, c, d) + (float)x + 0.5;
    out[i * 8 + 3] = (float)(y + 1) + 0.5;
    break;
  case 14:
    out[i * 8 + 0] = map(threshold, c, d) + (float)x + 0.5;
    out[i * 8 + 1] = (float)(y + 1) + 0.5;
    out[i * 8 + 2] = (float)(x + 1) + 0.5;
    out[i * 8 + 3] = map(threshold, b, d) + (float)y + 0.5;
    break;
  }
}
}
