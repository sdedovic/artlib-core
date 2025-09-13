(ns artlib.geometry.contour
  (:require [artlib.core :as core]
            [artlib.acceleration.core :as accel]))

(defn calc-contour-lines [heightmap threshold]
  (if-let [accelerator (core/accelerator)]
    (if (satisfies? accel/ContourHelper accelerator)
      (accel/compute-contour-lines accelerator heightmap threshold)
      (println "WARNING This accelerator does not implement the ContourHelper protocol."))
    (println "WARNING No accelerator found, cannot compute contour lines.")))
