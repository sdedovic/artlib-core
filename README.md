# artlib-core

A Clojure library for creating computational artwork. Inspired by [thobbs/genartlib](https://github.com/thobbs/genartlib) and created as a supplement.

This is not published, yet. See also [artlib-core](https://github.com/sdedovic/artlib-cuda).

## Contents 
### 3d
- **`3d.native`** - 3D projection a la OpenGL and GLM. Backed by `vectorz-clj` and runs on the CPU.

### midi
- **`midi.core`** - low level access to .mid files
- **`midi.data`** - higher-level, more useful access to .mid file data

### quil
- **`quil.global`** - macros I typically require with `:refer :all`
- **`quil.middleware`** - a useful animation middleware built on top of [quil's `fun-mode` middleware](https://github.com/quil/quil/wiki/Functional-mode-%28fun-mode%29)

### misc
- **`modulation`** - A set of modulation sources based on the audio production concepts
