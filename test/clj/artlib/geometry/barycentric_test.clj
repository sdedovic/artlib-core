(ns artlib.geometry.barycentric-test
  (:require [artlib.geometry.barycentric :refer [point->barycentric]]
            [artlib.testutils :refer [eq-vecf]]
            [clojure.test :refer [is deftest]]))

(deftest basic-on-vertex
  (let [tri-1 (partial point->barycentric [-1 1] [1 1] [0 -1])
        tri-2 (partial point->barycentric [-13.2 24] [93.1 0.233] [0.031 -100])
        tri-3 (partial point->barycentric [1 1] [-1 1] [0 -1])]

    ;; basic test
    (is (eq-vecf [1 0 0] (tri-1 [-1 1])))
    (is (eq-vecf [0 1 0] (tri-1 [1 1])))
    (is (eq-vecf [0 0 1] (tri-1 [0 -1])))
    
    ;; using floats and larger values
    (is (eq-vecf [1 0 0] (tri-2 [-13.2 24])))
    (is (eq-vecf [0 1 0] (tri-2 [93.1 0.233])))
    (is (eq-vecf [0 0 1] (tri-2 [0.031 -100])))

    ;; counter-clockwise triangle
    (is (eq-vecf [1 0 0] (tri-3 [1 1])))
    (is (eq-vecf [0 1 0] (tri-3 [-1 1])))
    (is (eq-vecf [0 0 1] (tri-3 [0 -1])))))

(deftest inside
  (let [h (/ (Math/sqrt 3.0) 2)
        apothem (/ h 3)
        tri (partial point->barycentric [-1/2 (- h)] [0 (* 2 h)] [1/2 (- h)])]

    ;; center is [1/3, 1/3, 1/3]
    (is (eq-vecf [1/3 1/3 1/3] (tri [0 0])))))
