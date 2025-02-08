(ns artlib.cuda.core
  (:require [clojure.java.io :as io]))

(defn default-headers
  "Returns the set of default headers to include when compiling a CUDA program."
  []
  {"Nvidia/helper_math.h" (slurp (io/resource "vendor/helper_math.h"))})

(defn default-nvcc-args
  "Returns a list of default compiler args."
  []
  (if-let [cuda-path (System/getenv "CUDA_PATH")]
    [(str "-I " cuda-path "/include")]
    []))

