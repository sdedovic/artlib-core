(ns artlib.acceleration.cuda
  (:refer-clojure :exclude [name])
  (:require [artlib.acceleration.core :as accel]
            [clojure.pprint :refer [pprint]]
            [uncomplicate.clojurecuda.core :as cuda]
            [uncomplicate.commons.core :refer [info with-release]]
            [mikera.image.core :as img])
  (:import [java.awt.image DataBuffer]))

;; Acceleration

(defrecord CudaAcceleration [device ctx]
  :load-ns true
  
  accel/HardwareAcceleration
  (info [this]
    (info (:device this)))
  
  accel/ContourHelper
  (compute-contour-lines
    [this heightmap threshold]
    (let [channels (->> heightmap .getRaster .getNumBands)]
      (when (> channels 1)
        (throw (new IllegalArgumentException "heightmap has more than 1 color channel")))

      (let [dtype (->> heightmap
                       .getRaster
                       .getDataBuffer
                       .getDataType)]
        (when (not= dtype DataBuffer/TYPE_FLOAT)
          (throw (new IllegalArgumentException "heightmap data is not type float")))
          
        (let [width (img/width heightmap) 
              height (img/height heightmap) 
              pixels (img/get-pixels heightmap)] 
          (with-release [pixel-buffer (cuda/mem-alloc-driver (* width height Float/BYTES))
                         out-buffer (cuda/mem-alloc-driver (* width height Integer/BYTES))]
            (println "beep boop bop .... zzzzz..... r..f.g..a.....g..gggg....  pretending to do gpu stuff")
            []))))))

;; Acceleration Provider

(defrecord CudaAccelerationProvider []
  :load-ns true

  accel/HardwareAccelerationProvider
  (name [this] "CUDAAccelerationProvider")
  (create [this] (accel/create this {}))
  (create [this _]
    (cuda/init)
    (let [device (cuda/device 0)
          ctx (cuda/context device)]
      (->CudaAcceleration device ctx))))
