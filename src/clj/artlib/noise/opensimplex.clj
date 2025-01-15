(ns artlib.noise.opensimplex
  "Clojure bindings for OpenSimplex2 noise functions.
  All functions take an optional seed value and return noise in the range [-1, 1].
  When seed is not provided, uses value from global seed atom."
  (:import (opensimplex2 OpenSimplex2S)))

(def ^:private seed (atom 42))

;; 2D noise functions
(defn noise2
  "2D Simplex noise with standard lattice orientation.
   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y]
   (noise2 @seed x y))
  ([seed x y]
   (OpenSimplex2S/noise2 (long seed) (double x) (double y))))

(defn noise2-norm
  "2D Simplex noise with standard lattice orientation.
   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y]
   (noise2-norm @seed x y))
  ([seed x y]
   (Math/fma (OpenSimplex2S/noise2 (long seed) (double x) (double y)) (float 0.5) (float 0.5))))

(defn noise2-improve-x
  "2D Simplex noise with Y pointing down the main diagonal.
   Better suited for 2D sandbox games where Y is vertical.
   Consider using standard noise2 for heightmaps or continent maps
   unless your map is centered around an equator.
   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y]
   (noise2-improve-x @seed x y))
  ([seed x y]
   (OpenSimplex2S/noise2_ImproveX (long seed) (double x) (double y))))

(defn noise2-improve-x-norm
  "2D Simplex noise with Y pointing down the main diagonal.
   Better suited for 2D sandbox games where Y is vertical.
   Consider using standard noise2 for heightmaps or continent maps
   unless your map is centered around an equator.
   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y]
   (noise2-improve-x-norm @seed x y))
  ([seed x y]
   (Math/fma (OpenSimplex2S/noise2_ImproveX (long seed) (double x) (double y)) (float 0.5) (float 0.5))))

;; 3D noise functions
(defn noise3-improve-xy
  "3D OpenSimplex2 noise with better visual isotropy in (X, Y).
   Recommended for 3D terrain and time-varied animations where Z is vertical
   or represents time.

   Usage patterns:
   - If Y is vertical: (noise3-improve-xz x z y)
   - If Z is vertical: (noise3-improve-xy x y z)
   - For time animation: (noise3-improve-xy x y t)

   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z]
   (noise3-improve-xy @seed x y z))
  ([seed x y z]
   (OpenSimplex2S/noise3_ImproveXY (long seed) (double x) (double y) (double z))))

(defn noise3-improve-xy-norm
  "3D OpenSimplex2 noise with better visual isotropy in (X, Y).
   Recommended for 3D terrain and time-varied animations where Z is vertical
   or represents time.

   Usage patterns:
   - If Y is vertical: (noise3-improve-xz x z y)
   - If Z is vertical: (noise3-improve-xy x y z)
   - For time animation: (noise3-improve-xy x y t)

   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z]
   (noise3-improve-xy-norm @seed x y z))
  ([seed x y z]
   (Math/fma (OpenSimplex2S/noise3_ImproveXY (long seed) (double x) (double y) (double z)) (float 0.5) (float 0.5))))

(defn noise3-improve-xz
  "3D OpenSimplex2 noise with better visual isotropy in (X, Z).
   Recommended for 3D terrain and time-varied animations where Y is vertical
   or represents time.

   Usage patterns:
   - If Y is vertical: (noise3-improve-xz x y z)
   - If Z is vertical: Use noise3-improve-xy instead
   - For time animation: (noise3-improve-xz x t z) or use noise3-improve-xy

   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z]
   (noise3-improve-xz @seed x y z))
  ([seed x y z]
   (OpenSimplex2S/noise3_ImproveXZ (long seed) (double x) (double y) (double z))))

(defn noise3-improve-xz-norm
  "3D OpenSimplex2 noise with better visual isotropy in (X, Z).
   Recommended for 3D terrain and time-varied animations where Y is vertical
   or represents time.

   Usage patterns:
   - If Y is vertical: (noise3-improve-xz x y z)
   - If Z is vertical: Use noise3-improve-xy instead
   - For time animation: (noise3-improve-xz x t z) or use noise3-improve-xy

   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z]
   (noise3-improve-xz-norm @seed x y z))
  ([seed x y z]
   (Math/fma (OpenSimplex2S/noise3_ImproveXZ (long seed) (double x) (double y) (double z)) (float 0.5) (float 0.5))))

(defn noise3
  "3D OpenSimplex2 noise using fallback rotation option.
   Consider using noise3-improve-xy or noise3-improve-xz instead where appropriate,
   as they have less diagonal bias. Best used as a fallback option.
   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z]
   (noise3 @seed x y z))
  ([seed x y z]
   (OpenSimplex2S/noise3_Fallback (long seed) (double x) (double y) (double z))))

(defn noise3-norm
  "3D OpenSimplex2 noise using fallback rotation option.
   Consider using noise3-improve-xy or noise3-improve-xz instead where appropriate,
   as they have less diagonal bias. Best used as a fallback option.
   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z]
   (noise3-norm @seed x y z))
  ([seed x y z]
   (Math/fma (OpenSimplex2S/noise3_Fallback (long seed) (double x) (double y) (double z)) (float 0.5) (float 0.5))))

;; 4D noise functions
(defn noise4-improve-xyz-improve-xy
  "4D OpenSimplex2 noise with XYZ oriented like noise3-improve-xy
   and W providing an extra degree of freedom (repeats eventually).
   Recommended for time-varied animations texturing 3D objects (W=time)
   in spaces where Z is vertical.
   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4-improve-xyz-improve-xy @seed x y z w))
  ([seed x y z w]
   (OpenSimplex2S/noise4_ImproveXYZ_ImproveXY (long seed) (double x) (double y) (double z) (double w))))

(defn noise4-improve-xyz-improve-xy-norm
  "4D OpenSimplex2 noise with XYZ oriented like noise3-improve-xy
   and W providing an extra degree of freedom (repeats eventually).
   Recommended for time-varied animations texturing 3D objects (W=time)
   in spaces where Z is vertical.
   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4-improve-xyz-improve-xy-norm @seed x y z w))
  ([seed x y z w]
   (Math/fma (OpenSimplex2S/noise4_ImproveXYZ_ImproveXY (long seed) (double x) (double y) (double z) (double w)) (float 0.5) (float 0.5))))

(defn noise4-improve-xyz-improve-xz
  "4D OpenSimplex2 noise with XYZ oriented like noise3-improve-xz
   and W providing an extra degree of freedom (repeats eventually).
   Recommended for time-varied animations texturing 3D objects (W=time)
   in spaces where Y is vertical.
   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4-improve-xyz-improve-xz @seed x y z w))
  ([seed x y z w]
   (OpenSimplex2S/noise4_ImproveXYZ_ImproveXZ (long seed) (double x) (double y) (double z) (double w))))

(defn noise4-improve-xyz-improve-xz-norm
  "4D OpenSimplex2 noise with XYZ oriented like noise3-improve-xz
   and W providing an extra degree of freedom (repeats eventually).
   Recommended for time-varied animations texturing 3D objects (W=time)
   in spaces where Y is vertical.
   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4-improve-xyz-improve-xz-norm @seed x y z w))
  ([seed x y z w]
   (Math/fma (OpenSimplex2S/noise4_ImproveXYZ_ImproveXZ (long seed) (double x) (double y) (double z) (double w)) (float 0.5) (float 0.5))))

(defn noise4-improve-xyz
  "4D OpenSimplex2 noise with XYZ oriented like noise3
   and W providing an extra degree of freedom (repeats eventually).
   Recommended for time-varied animations texturing 3D objects (W=time)
   where there isn't a clear distinction between horizontal and vertical.
   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4-improve-xyz @seed x y z w))
  ([seed x y z w]
   (OpenSimplex2S/noise4_ImproveXYZ (long seed) (double x) (double y) (double z) (double w))))

(defn noise4-improve-xyz-norm
  "4D OpenSimplex2 noise with XYZ oriented like noise3
   and W providing an extra degree of freedom (repeats eventually).
   Recommended for time-varied animations texturing 3D objects (W=time)
   where there isn't a clear distinction between horizontal and vertical.
   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4-improve-xyz-norm @seed x y z w))
  ([seed x y z w]
   (Math/fma (OpenSimplex2S/noise4_ImproveXYZ (long seed) (double x) (double y) (double z) (double w)) (float 0.5) (float 0.5))))

(defn noise4-improve-xy-improve-zw
  "4D OpenSimplex2 noise with XY and ZW forming orthogonal triangular-based planes.
   Recommended for:
   - 3D terrain where X and Y (or Z and W) are horizontal
   - noise(x, y, sin(time), cos(time)) animations
   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4-improve-xy-improve-zw @seed x y z w))
  ([seed x y z w]
   (OpenSimplex2S/noise4_ImproveXY_ImproveZW (long seed) (double x) (double y) (double z) (double w))))

(defn noise4-improve-xy-improve-zw-norm
  "4D OpenSimplex2 noise with XY and ZW forming orthogonal triangular-based planes.
   Recommended for:
   - 3D terrain where X and Y (or Z and W) are horizontal
   - noise(x, y, sin(time), cos(time)) animations
   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4-improve-xy-improve-zw-norm @seed x y z w))
  ([seed x y z w]
   (Math/fma (OpenSimplex2S/noise4_ImproveXY_ImproveZW (long seed) (double x) (double y) (double z) (double w)) (float 0.5) (float 0.5))))

(defn noise4
  "4D OpenSimplex2 noise using fallback lattice orientation.
   Consider using one of the other 4D noise functions if their
   specific characteristics match your use case.
   Returns values in the range [-1, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4 @seed x y z w))
  ([seed x y z w]
   (OpenSimplex2S/noise4_Fallback (long seed) (double x) (double y) (double z) (double w))))

(defn noise4-norm
  "4D OpenSimplex2 noise using fallback lattice orientation.
   Consider using one of the other 4D noise functions if their
   specific characteristics match your use case.
   Returns values in the range [0, 1].
   Optional seed parameter - uses global seed atom if not provided."
  ([x y z w]
   (noise4-norm @seed x y z w))
  ([seed x y z w]
   (Math/fma (OpenSimplex2S/noise4_Fallback (long seed) (double x) (double y) (double z) (double w)) (float 0.5) (float 0.5))))

;; Seed management
(defn set-seed!
  "Set the global seed used by all noise functions when no explicit seed is provided."
  [^long new-seed]
  (reset! seed new-seed))

(defn get-seed
  "Get the current global seed value."
  ^long []
  @seed)