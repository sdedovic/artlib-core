(ns artlib.cuda.noise
  (:require [clojure.java.io :as io]
            [artlib.cuda.core :refer [default-headers default-nvcc-args]]
            [uncomplicate.clojurecuda.core :refer [in-context] :as cuda]
            [uncomplicate.commons.core :refer [release with-release let-release]])
  (:import [org.bytedeco.cuda.cudart float2 int2]))

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
                                 (if-let [data (ex-data e)]
                                   (if-let [details (:details data)]
                                     (println details)))
                                 e))]
        (let-release [module (cuda/module program)]
          module)))))

(defn noise-2d
  [module buffer { [width height] :resolution [scale-x scale-y] :scale [offset-x offset-y] :offset }]
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
      (cuda/parameters buffer (.asByteBuffer resolution) (.asByteBuffer scale) (.asByteBuffer offset)))))
