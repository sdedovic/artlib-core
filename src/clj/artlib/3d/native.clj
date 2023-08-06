;; See: https://github.com/g-truc/glm

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

(defn glm-translate
  "Builds a translation 4 * 4 matrix created from a vector of 3 components.
  matrix - Input matrix multiplied by this translation matrix.
  v - Coordinates of a translation vector."
  [matrix [x y z :as v]]
  (let [transform [[1 0 0 x]
                   [0 1 0 y]
                   [0 0 1 z]
                   [0 0 0 1]]]
    (mmul matrix transform)))

(defn ^:deprecated translate 
  "Use glm-translate instead."
  [matrix v]
  (glm-translate matrix v))

(defn glm-scale
  "Builds a scale 4 * 4 matrix created from 3 scalars.
  matrix - Input matrix multiplied by this scale matrix.
  v - Ratio of scaling for each axis."
  [matrix [x y z :as v]]
  (let [transform [[x 0 0 0]
                   [0 y 0 0]
                   [0 0 z 0]
                   [0 0 0 1]]]
    (mmul matrix transform)))

(defn glm-rotate
  "Builds a rotation 4 * 4 matrix created from an axis vector and an angle.
  matrix - Input matrix multiplied by this rotation matrix.
  angle - Rotation angle expressed in radians.
  axis - Rotation axis, normalized."
  [matrix angle [x y z :as axis]]
  (let [c (Math/cos angle)
        s (Math/sin angle)
        cc (- 1 c)
        transform [[(+ (* x x cc) c)        (- (* y x cc) (* z s))  (+ (* z x cc) (* y s))  0]
                   [(+ (* x y cc) (* z s))  (+ (* y y cc) c)        (- (* z y cc) (* x s))  0]
                   [(- (* x z cc) (* y s))  (+ (* y z cc) (* x s))  (+ (* z z cc) c)        0]
                   [0                       0                       0                       1]]]
    (mmul matrix transform)))

(defn perspective-fov 
  "Create a perpective projection matrix. FOV is in radians.
  See: mat4#perspective https://glmatrix.net/docs/mat4.js.html"
  ([]
   (perspective-fov (* 1/2 Math/PI) 1.0 0.1 10))
  ([fovy aspect-ratio near-plane far-plane]
   (let [f (/ 1.0 (Math/tan (/ fovy 2.0)))
         nf (/ 1.0 (- near-plane far-plane))]
     [[(/ f aspect-ratio) 0 0 0]
      [0 f 0 0]
      [0 0 (* (+ far-plane near-plane) nf) (* 2 far-plane near-plane nf)]
      [0 0 -1 0]])))

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
  "Convenience function to perform vertex transformation. Performs viewport transformation and perspective division, 
    i.e. divides by the w value and normalizes to [0, 1]."
  [projection view model [x y z :as point]] 
  (let [[x y z w] (mmul projection view model [x y z 1])
        x (+ 0.5 (* 0.5 (/ x w)))
        y (- 0.5 (* 0.5 (/ y w)))
        z (/ z w)] 
    (with-meta [x y z] (meta point))))
