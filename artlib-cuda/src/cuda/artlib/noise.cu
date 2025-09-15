#include <helper_math.h>

extern "C" {
// this hash is not production ready, please replace this by something better
__device__ float2 hash2(float2 x) {
  const float2 k = make_float2( 0.3183099, 0.3678794 );
  x = x*k + make_float2(k.y, k.x);
  return -1.0 + 2.0*fracf( 16.0 * k*fracf( x.x*x.y*(x.x+x.y)) );
}

// https://iquilezles.org/articles/gradientnoise/
__device__ float3 calcNoised2(float2 pos) {
  float2 i = floorf(pos);
  float2 f = fracf(pos);

  float2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
  float2 du = 30.0*f*f*(f*(f-2.0)+1.0);

  float2 ga = hash2( i + make_float2(0.0,0.0) );
  float2 gb = hash2( i + make_float2(1.0,0.0) );
  float2 gc = hash2( i + make_float2(0.0,1.0) );
  float2 gd = hash2( i + make_float2(1.0,1.0) );

  float va = dot(ga, f - make_float2(0.0,0.0));
  float vb = dot(gb, f - make_float2(1.0,0.0));
  float vc = dot(gc, f - make_float2(0.0,1.0));
  float vd = dot(gd, f - make_float2(1.0,1.0));

  float value = va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd);
  float2 derivatives = ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +
                       du * (make_float2(u.y, u.x)*(va-vb-vc+vd) + make_float2(vb,vc) - va);
  return make_float3(value, derivatives.x, derivatives.y);
}

__global__ void noise2(float *buffer, int2 resolution, float2 scale, float2 offset) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= resolution.x || y >= resolution.y)
    return;
  int idx = x + y * resolution.x;

  float2 pos = (make_float2(x, y) / make_float2(resolution.x, resolution.y)) * scale + offset;
  buffer[x + y * resolution.x] = calcNoised2(pos).x;
}
}
