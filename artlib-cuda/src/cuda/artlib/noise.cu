#include <helper_math.h>

extern "C" {

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
__device__ float pcg_hash(float input) {
    uint i = (uint) input;
    uint state = i * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint out = (word >> 22u) ^ word;
    return (float) out;
}

// https://www.shadertoy.com/view/XdXBRH
// this hash is not production ready, please replace this by something better
__device__ float2 hash2(float2 in, float random) {
    const float2 k = make_float2(0.3183099, 0.3678794);
    in = in * k + make_float2(k.y, k.x);
    return -1.0 + 2.0 * fracf(16.0 * k * fracf( in.x * in.y * (in.x + in.y)));
}

// https://iquilezles.org/articles/gradientnoise/
__device__ float3 calcNoised2(float2 pos, float random) {
  float2 i = floorf(pos);
  float2 f = fracf(pos);

  float2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
  float2 du = 30.0*f*f*(f*(f-2.0)+1.0);

  float2 ga = hash2(i + make_float2(0.0,0.0), random);
  float2 gb = hash2(i + make_float2(1.0,0.0), random);
  float2 gc = hash2(i + make_float2(0.0,1.0), random);
  float2 gd = hash2(i + make_float2(1.0,1.0), random);

  float va = dot(ga, f - make_float2(0.0,0.0));
  float vb = dot(gb, f - make_float2(1.0,0.0));
  float vc = dot(gc, f - make_float2(0.0,1.0));
  float vd = dot(gd, f - make_float2(1.0,1.0));

  float value = va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd);
  float2 derivatives = ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +
                       du * (make_float2(u.y, u.x)*(va-vb-vc+vd) + make_float2(vb,vc) - va);
  return make_float3(value, derivatives.x, derivatives.y);
}

__global__ void noise2(float *buffer, float *rngBuffer, int2 resolution, float2 scale, float2 offset) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= resolution.x || y >= resolution.y)
    return;
  int idx = x + y * resolution.x;

  float random = rngBuffer[idx];
  float2 pos = (make_float2(x, y) / make_float2(resolution.x, resolution.y)) * scale + offset;
  buffer[idx] = calcNoised2(pos, random).x;
}

__global__ void noised2(float *buffer, float *rngBuffer, int2 resolution, float2 scale, float2 offset) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= resolution.x || y >= resolution.y)
    return;
  int idx = x + y * resolution.x;

  float random = rngBuffer[idx];
  float2 pos = (make_float2(x, y) / make_float2(resolution.x, resolution.y)) * scale + offset;
  float3 n = calcNoised2(pos, random);

  buffer[idx*3+0] = n.x;
  buffer[idx*3+1] = n.y;
  buffer[idx*3+2] = n.z;
}
}
