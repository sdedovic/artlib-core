#include <helper_math.h>

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

__constant__ int vertex_lut[2][16][4] = {{{},
                                          {2, 3},
                                          {3, 4},
                                          {2, 4},
                                          {1, 2},
                                          {1, 3},
                                          {1, 2, 3, 4},
                                          {1, 4},
                                          {4, 1},
                                          {4, 1, 2, 3},
                                          {3, 1},
                                          {2, 1},
                                          {4, 2},
                                          {4, 3},
                                          {3, 2}},
                                         {{},
                                          {2, 3},
                                          {3, 4},
                                          {2, 4},
                                          {1, 2},
                                          {1, 3},
                                          {1, 4, 3, 2},
                                          {1, 4},
                                          {4, 1},
                                          {2, 1, 4, 3},
                                          {3, 1},
                                          {2, 1},
                                          {4, 2},
                                          {4, 3},
                                          {3, 2}}};

// Re-map number from one range to [0.0, 1.0].
// https://docs.arduino.cc/language-reference/en/functions/math/map/
__device__ inline float map(float x, float in_min, float in_max) {
  return (in_min == in_max && in_min == 0.0) ? 0.0
                                             : (x - in_min) / (in_max - in_min);
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
  int center = (int)(m > threshold);

  float2 top = make_float2((float)x + map(threshold, a, b), y) + 0.5;
  float2 right = make_float2(x + 1, (float)y + map(threshold, b, d)) + 0.5;
  float2 bottom = make_float2((float)x + map(threshold, c, d), y + 1) + 0.5;
  float2 left = make_float2(x, (float)y + map(threshold, a, c)) + 0.5;
  float2 vertices[] = {make_float2(0.0, 0.0), top, right, bottom, left};

  // TODO: DEBUG symbol in constant memory to flag logging
  // printf("config: %d cell: %dx%d index: %d\n", configuration, x, y, i);
  // printf("values a, b, c, d are: %.2f %.2f %.2f %.2f\n", a, b, c, d);

  out[i * 8 + 0] = vertices[vertex_lut[center][configuration][0]].x;
  out[i * 8 + 1] = vertices[vertex_lut[center][configuration][0]].y;
  out[i * 8 + 2] = vertices[vertex_lut[center][configuration][1]].x;
  out[i * 8 + 3] = vertices[vertex_lut[center][configuration][1]].y;
  out[i * 8 + 4] = vertices[vertex_lut[center][configuration][2]].x;
  out[i * 8 + 5] = vertices[vertex_lut[center][configuration][2]].y;
  out[i * 8 + 6] = vertices[vertex_lut[center][configuration][3]].x;
  out[i * 8 + 7] = vertices[vertex_lut[center][configuration][3]].y;
}
}
