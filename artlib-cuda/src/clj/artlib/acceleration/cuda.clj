(ns artlib.acceleration.cuda
  (:refer-clojure :exclude [name])
  (:require [artlib.acceleration.core :as accel]
            [artlib.cuda.contour :as contour-kernel]
            [uncomplicate.clojurecuda.core :as cuda]
            [uncomplicate.commons.core :refer [info with-release]]
            [uncomplicate.clojure-cpp :refer [float-pointer int-pointer pointer-seq] :as cpp]
            [mikera.image.core :as img])
  (:import [java.awt.image DataBuffer]
           [org.bytedeco.cuda.global cudart]))

;; Acceleration

(defrecord CudaAcceleration [device ctx modules]
  :load-ns true

  accel/HardwareAcceleration
  (info [this]
    (info (:device this)))

  accel/ContourHelper
  (compute-contour-lines
    [this heightmap threshold]
    (let [channels (->> heightmap .getSampleModel .getNumBands)]
      (when (> channels 1)
        (throw (new IllegalArgumentException "heightmap has more than 1 color channel")))

      (let [dtype (->> heightmap .getRaster .getDataBuffer .getDataType)]
        (when (not= dtype DataBuffer/TYPE_FLOAT)
          (throw (new IllegalArgumentException "heightmap data is not type float")))

        (let [width (img/width heightmap) 
              height (img/height heightmap) 

              ;; NOTE: since cells are calculated 2x2, we iterate over the width and 
              ;;  height minus one
              num-cells (* (dec width) (dec height))

              pixels (img/get-pixels heightmap)
              module (get-in this [:modules ::contour :module])
              buffers-atom (get-in this [:modules ::contour :buffers])]
          (cuda/in-context 
            (:ctx this)

            (when (or (nil? @buffers-atom)
                      (> num-cells (:capacity @buffers-atom)))
              (reset! buffers-atom {:capacity num-cells

                                    :pixel-buffer 
                                    (cuda/mem-alloc-driver (* width height Float/BYTES))

                                    ;; NOTE: output is one or two line segments, i.e. [[x1 y1] [x2 y2] ...] which 
                                    ;;   means it requires, at worst, 8 floats to store, 4 per line segment
                                    :polygon-buffer
                                    (cuda/mem-alloc-driver (* num-cells 8 Float/BYTES))
                                    
                                    ;; atomic counter
                                    :segment-counter
                                    (cuda/mem-alloc-driver Integer/BYTES)}))

            (let [{pixel-buffer :pixel-buffer 
                   polygon-buffer :polygon-buffer
                   segment-counter :segment-counter} @buffers-atom]
              (cuda/memcpy-host! pixels pixel-buffer)
              (let [result (->> [threshold]
                                flatten
                                (map (fn [threshold]
                                       (cuda/memcpy-host! 0 segment-counter) ;; reset counter
                                       (contour-kernel/calculate-line-segments module 
                                                                               pixel-buffer polygon-buffer 
                                                                               segment-counter
                                                                               width height 
                                                                               threshold)
                                       (with-release [counter-ptr (int-pointer 1)]
                                         (cuda/memcpy-host! segment-counter counter-ptr)
                                         (let [num-segments (cpp/get-entry counter-ptr)]
                                           (with-release [polygon-ptr (float-pointer (* num-segments 4))]
                                             (cuda/memcpy-host! polygon-buffer polygon-ptr)
                                             (let [segments (->> (pointer-seq polygon-ptr)
                                                                 (partition 2)
                                                                 (map vec)
                                                                 (partition 2)
                                                                 (map vec)
                                                                 ;; this is required to consume the data otherwise it will
                                                                 ;;  be released when returning the sequence
                                                                 doall)]
                                               (with-meta segments {:threshold threshold})))))))
                                doall)]
                (if (sequential? threshold)
                  result
                  (first result))))))))))

(defn new-cuda-acceleration [device]
  (let [ctx (cuda/context device)
        modules {::contour 
                 {:module (contour-kernel/create-module ctx)
                  :buffers (atom nil)}}]
    (->CudaAcceleration device ctx modules)))

;; Acceleration Provider

(defrecord CudaAccelerationProvider []
  :load-ns true

  accel/HardwareAccelerationProvider
  (name [this] "CUDAAccelerationProvider")
  (create [this] (accel/create this {:device-index 0}))
  (create [this {device :device-index}]
    (cuda/init)
    (new-cuda-acceleration (cuda/device device))))
