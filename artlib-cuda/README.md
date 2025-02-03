# artlib-cuda
[![Clojars Project](https://img.shields.io/clojars/v/com.dedovic/artlib-cuda.svg)](https://clojars.org/com.dedovic/artlib-cuda)


> [!WARNING]
> This is old and subject to immediate change


A Clojure library for creating GPU accelerated computational artwork via CUDA. Heavily based on [uncomplicate/neanderthal](https://github.com/uncomplicate/neanderthal) and built on top of [uncomplicate/clojurecuda](https://github.com/uncomplicate/clojurecuda).

See also [artlib-core](https://github.com/sdedovic/artlib-core).

## Contents
### cuda
- **`cuda.curand`** - idiomatic wrapper around [cuRAND](https://developer.nvidia.com/curand) a la clojurecuda. 
- **`cuda.particles`** - wrapper around low-level particle simulation kernels, e.g. all-pairs charged particle simulation