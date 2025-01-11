(ns artlib.color.dictionary-test
  (:require [artlib.color.dictionary :as dictionary]
            [artlib.testutils :refer [eq-veci]]
            [clojure.test :refer [is deftest]]))

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
