(ns artlib.image-test
  (:require [artlib.image.core :as image]
            [mikera.image.core :as imagez]
            [clojure.test :refer :all])
  (:import [java.awt.image BufferedImage]))

(deftest image-test
  (testing "gray-32f"
    (let [img (image/make-gray-32f 3 3)]
      (imagez/set-pixels img (into-array Float/TYPE [0 0 0
                                                     0 1 0
                                                     0 0 0]))))
  (testing "rgb-32f"
    (let [img (image/make-rgb-32f 2 2)]
      (imagez/set-pixels img (into-array Float/TYPE [0 0.5 1  0 0.5 1
                                                     0.5 1 0  0.5 1 0])))))
