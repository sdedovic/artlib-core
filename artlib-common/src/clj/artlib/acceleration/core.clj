(ns artlib.acceleration.core
  (:refer-clojure :exclude [name])
  (:import [java.util ServiceLoader]
           [java.awt.image BufferedImage]))

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
    "Given a heightmap, returns the contour lines at the specified threshold as a seq of 
      line segments. If threshold is sequential (vec or seq), then returns a seq of seq
      of line segments in the same order."))

(defprotocol NoiseHelper
  "NoiseHelper provides utilities for computing noise functions."

  (noise2
    [this resolution opts]
    "Compute 2D noise and return a vec of values. The parameters are as follows:
      resolution [x y] - the size of the generated noise image in pixels
      opts { [x y] :scale [x y] :offset } - noise generation options
      scale [x y] - the scale of the noise, default is [1.0 1.0]
      offset [x y] - the noise offset, default is [0.0 0.0]")

  (noised2
    [this resolution opts]
    "Compute 2D noise and and its analytical derrivative, returning a seq. See `noise2` for parameter documentation. The
     resulting seq is packed [value-1 derrivative-x-1 derrivative-y-1 ... value-n derrivative-x-n derrivative-y-n"))