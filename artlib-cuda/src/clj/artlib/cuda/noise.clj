(ns artlib.cuda.noise
  (:require [artlib.cuda.core :refer [default-headers default-nvcc-args]]
            [clojure.java.io :as io]
            [uncomplicate.clojurecuda.core :refer [in-context] :as cuda]
            [uncomplicate.commons.core :refer [let-release with-release]])
  (:import (org.bytedeco.cuda.cudart float2 int2)))

(defn create-module
  "Compiles and load the contour kernel returning a CUDA module."
  [ctx]
  (in-context ctx
              (let [src (slurp (io/resource "artlib/noise.cu"))
                    headers (default-headers)
                    args (default-nvcc-args)]
                (with-release [program (try
                                         (cuda/compile! (cuda/program src headers) args)
                                         (catch Exception e
                                           (when-let [data (ex-data e)]
                                             (when-let [details (:details data)]
                                               (println details)))
                                           e))]
                  (let-release [module (cuda/module program)]
                               module)))))

(defn noise-2d
  "Calculate 2D noise."
  [module buffer rng-buffer {[width height] :resolution [scale-x scale-y] :scale [offset-x offset-y] :offset}]
  (with-release [kernel (cuda/function module "noise2")
                 resolution (doto (int2.)
                              (.x (int width))
                              (.y (int height)))
                 scale (doto (float2.)
                         (.x (float scale-x))
                         (.y (float scale-y)))
                 offset (doto (float2.)
                          (.x (float offset-x))
                          (.y (float offset-y)))]
    (cuda/launch!
      kernel
      (cuda/grid-2d width height)
      (cuda/parameters buffer rng-buffer (.asByteBuffer resolution) (.asByteBuffer scale) (.asByteBuffer offset)))))

(defn noised-2d
  "Calculate 2D noise and its analytical derrivative."
  [module buffer rng-buffer {[width height] :resolution [scale-x scale-y] :scale [offset-x offset-y] :offset}]
  (with-release [kernel (cuda/function module "noised2")
                 resolution (doto (int2.)
                              (.x (int width))
                              (.y (int height)))
                 scale (doto (float2.)
                         (.x (float scale-x))
                         (.y (float scale-y)))
                 offset (doto (float2.)
                          (.x (float offset-x))
                          (.y (float offset-y)))]
    (cuda/launch!
      kernel
      (cuda/grid-2d width height)
      (cuda/parameters buffer rng-buffer (.asByteBuffer resolution) (.asByteBuffer scale) (.asByteBuffer offset)))))
