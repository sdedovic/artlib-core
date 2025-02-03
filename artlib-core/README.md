# artlib-core
[![Clojars Project](https://img.shields.io/clojars/v/com.dedovic/artlib-core.svg)](https://clojars.org/com.dedovic/artlib-core)

A Clojure library for creating computational artwork. Inspired by [thobbs/genartlib](https://github.com/thobbs/genartlib) and created as a supplement.

Very expiremental. See also [artlib-cuda](https://github.com/sdedovic/artlib-cuda).

## Contents 
### 3d
- **`3d.native`** - 3D projection a la OpenGL and GLM. Backed by `vectorz-clj` and runs on the CPU

### color
- **`color.dictionary`** - implementation of [A Dictionary of Color combinations by Sanzo Wada](https://en.wikipedia.org/wiki/Sanzo_Wada)
- **`color.model`** - various color model transforms, e.g. `rgb->hsb`
- **`color.palettes`** - oscillating color palletes and color pallete generator

### geometry
- **`geometry.barycentric`** - barycentric coordinate conversions
- **`geometry.jts`** - a wrapper around the JTS Topology Suite containing various high-level utilities
- **`geometry.triangle`** - picking random points uniformly on a triangle

### keyframing
- **`keyframe.core`** - basic keyframing

### midi
- **`midi.core`** - low level access to .mid files
- **`midi.data`** - higher-level, more useful access to .mid file data

### noise
- **`noise.opensimplex`** - bindings to OpenSimplex library

### numeric
- **`numeric.solver`** - solver for multidimensional ODEs using numeric methods

### quil
- **`quil.global`** - macros I typically require with `:refer :all`
- **`quil.middleware`** - a useful animation middleware built on top of [quil's `fun-mode` middleware](https://github.com/quil/quil/wiki/Functional-mode-%28fun-mode%29)

### misc
- **`modulation`** - A set of modulation sources based on the audio production concepts
- **`waveform`** - Simple periodic waveforms, e.g. sine, cosine, triangle