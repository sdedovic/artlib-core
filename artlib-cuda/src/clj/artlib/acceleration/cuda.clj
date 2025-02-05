(ns artlib.acceleration.cuda
  (:refer-clojure :exclude [name])
  (:require [artlib.acceleration.core :as accel]
            [clojure.pprint :refer [pprint]]
            [uncomplicate.clojurecuda.core :as cuda]
            [uncomplicate.commons.core :refer [info]]
            ;;[uncomplicate.clojurecuda.info :refer :all]
            ))

;; Acceleration

(defrecord CudaAcceleration [device ctx]
  :load-ns true
  
  accel/HardwareAcceleration
  (info [this]
    (info (:device this)))
  
  accel/ContourHelper
  (compute-contour-lines
    [this heightmap threshold]
    []))

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
