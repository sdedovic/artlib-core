
(ns artlib.color.dictionary-test
  (:require [artlib.color.spaces :refer :all]
            [artlib.testutils :refer [eq-veci]]
            [clojure.test :refer [is deftest]]))


(deftest cmyk->rgb
  (let [black [20 10 15 100]
        white [0 0 0 0]
        sepia [48 60 100 40]]
    (is (eq-veci [0 0 0] (apply cmyk->rgb black)))
    (is (eq-veci [255 255 255] (apply cmyk->rgb white)))
    (is (eq-veci [80 61 0] (apply cmyk->rgb sepia)))))
