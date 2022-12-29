(ns artlib.3d.native
  (:require [clojure.core.matrix :refer [cross dot mmul select distance normalise]]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :refer [norm]]))

(defn mat4 
  "Generates a 4x4 identity matrix."
  []
  [[1 0 0 0] 
   [0 1 0 0] 
   [0 0 1 0] 
   [0 0 0 1]])

(defn translate 
  "Translate the matrix. Returns a new matrix."
  [matrix [x y z]]
  (let [transform [[1 0 0 x]
                   [0 1 0 y]
                   [0 0 1 z]
                   [0 0 0 1]]]
    (mmul matrix transform)))

(defn glm-scale
  "Scale the matrix. Returns a new matrix."
  [matrix [x y z]]
  (let [transform [[x 0 0 0]
                   [0 y 0 0]
                   [0 0 z 0]
                   [0 0 0 1]]]
    (mmul matrix transform)))

(defn perspective-fov 
  "Create a perpective projection matrix. FOV is in radians."
  ([]
   (perspective-fov (* 1/2 Math/PI) 1.0 0.1 100))
  ([fov aspect-ratio near-plane far-plane]
   (let [n (/ 1.0 (Math/tan (/ fov 2.0)))
         num-9 (/ n aspect-ratio)
         sum (+ near-plane far-plane)
         diff (- near-plane far-plane)]
     [[num-9 0 0 0]
      [0     n 0 0]
      [0     0 (/ sum diff) (/ (* 2 near-plane far-plane) diff)]
      [0     0 -1 0]])))

(defn look-at 
  "Create a view matrix."
  ([camera]
   (look-at camera [0 0 0] [0 1 0]))
  ([camera-pos camera-target up]
   (let [v (- camera-pos camera-target)
         v (/ v (norm v))
         v2 (cross up v)
         v2 (/ v2 (norm v2))
         v3 (cross v v2)]

     [[(select v2 0) (select v2 1) (select v2 2) (- (dot v2 camera-pos))]
      [(select v3 0) (select v3 1) (select v3 2) (- (dot v3 camera-pos))]
      [(select v  0) (select v  1) (select v  2) (- (dot v  camera-pos))]
      [0             0             0             1]])))

(defn project 
  "Convenience function to perform vertex projection. Result will have a Z value.
    Equivalent to: Vout = Mproject * Mview* Mmodel * Vin"
  [projection view model [x y z]] 
  (let [[x y z w] (mmul projection view model [x y z 1])
        x (+ 0.5 (* 0.5 (/ x w)))
        y (- 0.5 (* 0.5 (/ y w)))
        z (- 0.5 (* 0.5 (/ z w)))] 
    [x y z]))

