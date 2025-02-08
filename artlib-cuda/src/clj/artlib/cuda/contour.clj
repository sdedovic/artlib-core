(ns artlib.cuda.contour
  (:require [artlib.cuda.core :refer [default-headers default-nvcc-args]] 
            [uncomplicate.clojurecuda.core :refer [in-context] :as cuda]
            [uncomplicate.commons.core :refer [release with-release let-release]]))

(defn create-module
  "Compiles and load the contour kernel returning a CUDA module."
  [ctx]
  (in-context ctx
    (let [src (slurp "src/cuda/artlib/contour.cu")
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
      (cuda/parameters heightmap out width height threshold))))
