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

// 4 bit number, represents which edges of the cell have points for
//  the contour line segments.
//
// bit order is: top, right, bot, left
//  example,      1     0     1     0    = 0xA
//  example,      1     1     1     1    = 0xF
const int edgeLut[16] = {
    0x0, 0x6, 0x3, 0x5, 0xC, 0xA, 0xF, 0x9,
    0x9, 0xF, 0xA, 0xC, 0x5, 0x3, 0x6, 0x0,
};

// represents the winding order of the points, i.e. if the first
//  processed point is the start or end of the directed segment.
//
// 0 means points are processed in the same order as the edges,
//  top, right, bottom, left.
//
// 1 means points are processed in reverse order,
//  left, bottom, right, top
//
// -1 is a special case for saddle points and is undefined behavior
const int windingLut[16] = {
    0, 0, 0, 0, 0, 0, -1, 0, 1, -1, 1, 1, 1, 1, 1, 0,
};

// linearly interpolate point based on values
__device__ float2 interpolate(float x1, float y1, float v1, float x2, float y2,
                              float v2, float threshold) {
  float t = (threshold - v1) / (v2 - v1);
  return make_float2(fma(t, x2 - x1, x1), fma(t, y2 - y1, y1));
}

__global__ void calculate_line_segments(float *heightmap, float *out,
                                        int *segmentCount, int width,
                                        int height, float threshold) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width - 1 || y >= height - 1)
    return;

  // one cell is a collection of 2x2 pixels, indexed as the following:
  //  - a -> top left
  //  - b -> top right
  //  - c -> bottom left
  //  - d -> bottom right
  float a = heightmap[x + y * width];
  float b = heightmap[x + 1 + y * width];
  float c = heightmap[x + (y + 1) * width];
  float d = heightmap[x + 1 + (y + 1) * width];

  // configuration is which corners are above / below threshold
  int configuration = 0;
  if (a < threshold)
    configuration |= 8;
  if (b < threshold)
    configuration |= 4;
  if (c < threshold)
    configuration |= 2;
  if (d < threshold)
    configuration |= 1;

  int edges = edgeLut[configuration];
  if (edges == 0)
    return;

  // offset to center of cell
  float x1 = x + 0.5;
  float y1 = y + 0.5;
  float x2 = x + 1.5;
  float y2 = y + 1.5;

  float2 points[4];
  int pointCount = 0;

  // top
  if (edges & 8)
    points[pointCount++] = interpolate(x1, y1, a, x2, y1, b, threshold);
  // right
  if (edges & 4)
    points[pointCount++] = interpolate(x2, y1, b, x2, y2, d, threshold);
  // bottom
  if (edges & 2)
    points[pointCount++] = interpolate(x1, y2, c, x2, y2, d, threshold);
  // left
  if (edges & 1)
    points[pointCount++] = interpolate(x1, y1, a, x1, y2, c, threshold);

  if (pointCount == 2) {
    int idx = atomicAdd(segmentCount, 1);
    int firstIdx = windingLut[configuration] == 0 ? 0 : 1;
    int secondIdx = firstIdx == 0 ? 1 : 0;

    out[idx * 4 + 0] = points[firstIdx].x;
    out[idx * 4 + 1] = points[firstIdx].y;
    out[idx * 4 + 2] = points[secondIdx].x;
    out[idx * 4 + 3] = points[secondIdx].y;
  } else if (pointCount == 4) {
    float center = (a + b + c + d) / 4.0;
    int idx = atomicAdd(segmentCount, 1);
    if (configuration == 6) {
      if (center < threshold) {
        out[idx * 4 + 0] = points[0].x;
        out[idx * 4 + 1] = points[0].y;
        out[idx * 4 + 2] = points[3].x;
        out[idx * 4 + 3] = points[3].y;

        idx = atomicAdd(segmentCount, 1);
        out[idx * 4 + 0] = points[2].x;
        out[idx * 4 + 1] = points[2].y;
        out[idx * 4 + 2] = points[1].x;
        out[idx * 4 + 3] = points[1].y;
      } else {
        out[idx * 4 + 0] = points[0].x;
        out[idx * 4 + 1] = points[0].y;
        out[idx * 4 + 2] = points[1].x;
        out[idx * 4 + 3] = points[1].y;

        idx = atomicAdd(segmentCount, 1);
        out[idx * 4 + 0] = points[2].x;
        out[idx * 4 + 1] = points[2].y;
        out[idx * 4 + 2] = points[3].x;
        out[idx * 4 + 3] = points[3].y;
      }
    }
    if (configuration == 9) {
      if (center < threshold) {
        out[idx * 4 + 0] = points[1].x;
        out[idx * 4 + 1] = points[1].y;
        out[idx * 4 + 2] = points[0].x;
        out[idx * 4 + 3] = points[0].y;

        idx = atomicAdd(segmentCount, 1);
        out[idx * 4 + 0] = points[3].x;
        out[idx * 4 + 1] = points[3].y;
        out[idx * 4 + 2] = points[2].x;
        out[idx * 4 + 3] = points[2].y;
      } else {
        out[idx * 4 + 0] = points[1].x;
        out[idx * 4 + 1] = points[1].y;
        out[idx * 4 + 2] = points[2].x;
        out[idx * 4 + 3] = points[2].y;

        idx = atomicAdd(segmentCount, 1);
        out[idx * 4 + 0] = points[3].x;
        out[idx * 4 + 1] = points[3].y;
        out[idx * 4 + 2] = points[0].x;
        out[idx * 4 + 3] = points[0].y;
      }
    }
  }
}
}
