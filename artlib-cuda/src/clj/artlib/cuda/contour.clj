(ns artlib.cuda.contour
  (:require [clojure.java.io :as io]
            [artlib.cuda.core :refer [default-headers default-nvcc-args]] 
            [uncomplicate.clojurecuda.core :refer [in-context] :as cuda]
            [uncomplicate.commons.core :refer [release with-release let-release]]))

(defn create-module
  "Compiles and load the contour kernel returning a CUDA module."
  [ctx]
  (in-context ctx
    (let [src (slurp (io/resource "artlib/contour.cu"))
          headers (default-headers)
          args (default-nvcc-args)]
      (with-release [program (cuda/compile! (cuda/program src headers) args)]
        (let-release [module (cuda/module program)]
          module)))))

(defn threshold-f
  [module heightmap out width height threshold]
  (with-release [kernel (cuda/function module "threshold_f")]
    (cuda/launch!
      kernel
      (cuda/grid-2d width height)
      (cuda/parameters heightmap out (int width) (int height) (float threshold)))))

(defn calculate-line-segments
  [module heightmap out width height threshold]
  (with-release [kernel (cuda/function module "calculate_line_segments")]
    (cuda/launch!
      kernel
      (cuda/grid-2d width height)
      (cuda/parameters heightmap out (int width) (int height) (float threshold)))))
