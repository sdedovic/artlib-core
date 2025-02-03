(ns artlib.geometry.triangle-test
  (:require [clojure.test :refer [deftest is]]
            [artlib.geometry.triangle :as tri]))

(defn-
  sign
  "Helper for point-in-triangle? test."
  [[ax ay] [bx by] [cx cy]]
  (-
   (* (- ax cx) (- by cy))
   (* (- bx cx) (- ay cy))))

(defn-
  point-in-triangle?
  "Tests whether the point p is in the triangle defined by vertices a b c."
  [a b c p]
  (let [d1 (sign p a b)
        d2 (sign p b c)
        d3 (sign p c a)
        has_neg (->> [d1 d2 d3] (map neg?) (some true?))
        has_pos (->> [d1 d2 d3] (map pos?) (some true?))]
    (not (and has_neg has_pos))))

(deftest test-random-points
  (let [a [-3 0]
        b [0 10]
        c [10 5]]
    (dotimes [n 50]
      (let [point (tri/random-point a b c)]
        (is (point-in-triangle? a b c point)))))
  
  (let [a [-1.3 0.9]
        b [2.04 10/17]
        c [100 500]]
    (dotimes [n 50]
      (let [point (tri/random-point a b c)]
        (is (point-in-triangle? a b c point))))))
