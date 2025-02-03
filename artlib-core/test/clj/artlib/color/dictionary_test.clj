(ns artlib.color.dictionary-test
  (:require [artlib.color.dictionary :as dictionary]
            [artlib.testutils :refer [eq-veci]]
            [clojure.test :refer [is deftest testing]]))

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

; This test is the same as the above but does not pass the
;   initialized dictionary object as a parameter to the various
;   get-color-* functions. Instead, it relies on the default dict,
;   which is the same as simply executing the init function.
(deftest default-dict
  (is (eq-veci [0 0 0] (dictionary/get-color-hsb "black")))
  (is (eq-veci [0 0 100] (dictionary/get-color-hsb "white")))
  (is (eq-veci [46 100 31] (dictionary/get-color-hsb "sepia")))
  (is (eq-veci [60 100 100] (dictionary/get-color-hsb "yellow"))))


(deftest bugs
  (testing "bug genuary 2025 15"

    (is (=
          (->> (dictionary/get-combination-hsb :252)
               (map #(map int %)))
           [[337 80 90]
            [30 100 72]
            [53 100 72]
            [163 76 73]]))))