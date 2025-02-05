(ns artlib.acceleration.core
  (:refer-clojure :exclude [name])
  (:import [java.util ServiceLoader]))

;; Acceleration Provider

(defprotocol HardwareAccelerationProvider
  "This protocol is implemented to provide access to hardware acceleration algorithms."
  (name [this] "Returns the name of the the provider.")
  (create [this] [this opts] "Creates the provider. Opts are a provider-specific map."))

(defn providers
  "Get all hardware acceleration providers"
  []
  (->> HardwareAccelerationProvider
       :on-interface
       ServiceLoader/load
       (.iterator)
       iterator-seq))

;; Acceleration and Helpers

(defprotocol HardwareAcceleration
  "This protocol is implemented to provide hardware acceleration."
  (info [this] "Returns information for debugging."))

(defprotocol ContourHelper
  "ContourHelper provides utilities for computing contour lines using a marching squares 
    algorithm, based on on ContourPy.

  See: https://contourpy.readthedocs.io/en/latest/config.html"
  (compute-contour-lines
    [this heightmap threshold] 
    "Given a heightmap, returns the contour lines at the specified threshold as a seq of line segments."))
