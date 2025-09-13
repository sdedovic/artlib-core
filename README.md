# artlib
[![Clojars Project](https://img.shields.io/clojars/v/com.dedovic/artlib-core.svg)](https://clojars.org/com.dedovic/artlib-core)
</br>
[![Clojars Project](https://img.shields.io/clojars/v/com.dedovic/artlib-cuda.svg)](https://clojars.org/com.dedovic/artlib-cuda)

Monorepo for generative / computation art tooling in Clojure.

## Libraries
- [artlib-core](./artlib-core/) - all kinds of cross-platform utilities for quil (processing) and generative art
- [artlib-cuda](./artlib-cuda/) - accelerators that only work on linux machines with NVIDIA CUDA devices
- [artlib-common](./artlib-common/) - -probably not consumable on its own - more for exposing interface the higher-level packages consume


# development
## testing
```bash
lein monolith each check
lein monolith each test
```

## release
```bash
# given
direnv allow

lein release
```
