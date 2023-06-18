(ns artlib.color.dictionary-test
  (:require [artlib.color.dictionary :as dictionary]
            [clojure.test :refer [is deftest]]))

(let [epsilon 0.000001]
  (defn- eq-float
    "Truthiness of equality between two numbers using maching epsilon."
    [a b]
    (< (Math/abs (- (float a) (float b))) epsilon)))

(defn- eq-int
  "Truthiness of equality between two numbers using rounding to nearest int."
  [a b]
  (= 
    (Math/round (float a))
    (Math/round (float b))))

(defn- eq-vecf
  "Trutiness of equality of vector by testing float equality component wise."
  [a b]
  (if (not= (count a) (count b))
    false
    (->> (map vector a b)
         (map #(apply eq-float %))
         (filter false?)
         (empty?))))

(defn- eq-veci
  "Trutiness of equality of vector by testing integer equality component wise."
  [a b]
  (if (not= (count a) (count b))
    false
    (->> (map vector a b)
         (map #(apply eq-int %))
         (filter false?)
         (empty?))))

(deftest cmyk->rgb
  (let [black [20 10 15 100]
        white [0 0 0 0]
        sepia [48 60 100 40]]
    (is (eq-veci [0 0 0] (apply dictionary/cmyk->rgb black)))
    (is (eq-veci [255 255 255] (apply dictionary/cmyk->rgb white)))
    (is (eq-veci [80 61 0] (apply dictionary/cmyk->rgb sepia)))))

(deftest color->rgb
  (let [dict (dictionary/init)]
    (is (eq-veci [0 0 0] (dictionary/get-color-rgb dict "black")))
    (is (eq-veci [255 255 255] (dictionary/get-color-rgb dict "white")))
    (is (eq-veci [80 61 0] (dictionary/get-color-rgb dict "sepia")))
    (is (eq-veci [255 255 0] (dictionary/get-color-rgb dict "yellow")))))

(deftest color->hsb
  (let [dict (dictionary/init)]
    (is (eq-veci [0 0 0] (dictionary/get-color-hsb dict "black")))
    (is (eq-veci [0 0 100] (dictionary/get-color-hsb dict "white")))
    (is (eq-veci [46 100 31] (dictionary/get-color-hsb dict "sepia")))
    (is (eq-veci [60 100 100] (dictionary/get-color-hsb dict "yellow")))))
