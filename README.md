# artlib
Monorepo for generative / computation art tooling in Clojure.

## Libraries
- [artlib-core](./artlib-core/) - all kinds of cross-platform utilities for quil (processing) and generative art
- [artlib-cuda](./artlib-cuda/) - accelerators that only work on linux machines with NVIDIAs CUDA-enabled GPUs
- [artlib-common](./artlib-common/) - not directly consumable - provides interface for the other libraries


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
