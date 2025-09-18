(ns artlib.noise-test
  (:require [artlib.acceleration.core :as acceleration]
            [artlib.cuda.noise :as noise]
            [artlib.image.core :refer [make-rgb-32f convert-f32-to-int-rgb]]
            [mikera.image.core :as imagez]
            [clojure.test :refer :all])
  (:import [java.awt.image BufferedImage]))

(deftest noise-test
  (let [accel (acceleration/create (first (acceleration/providers)))
        {ctx :ctx device :device} accel]

    (testing "module creation"
      (is (some? (noise/create-module ctx))))

    (testing "basic execution flow"
      (let [out (acceleration/noise2 accel [5 5] {:scale [0.2 0.2] :offset [1.35 7.23]})]
        (is (some? out))
        (is (seq? out))
        (is (= (count out) 25))
        (is (every? (complement zero?) out)))

      (let [out (acceleration/noised2 accel [2 2] {:scale [0.2 0.2] :offset [1.35 7.23]})]
        (is (some? out))
        (is (seq? out))
        (is (= (count out) 12))
        (is (every? (complement zero?) out))))

    (testing "basic image creation"
      (let [dims [1920 1920]
            [width height] dims
            image (make-rgb-32f width height)
            out (acceleration/noised2 accel dims {:scale [2 2] :offset [0 0]})
            as-array (into-array Float/TYPE out)]
        (imagez/set-pixels image as-array)
        (imagez/save (convert-f32-to-int-rgb image) "test.png")))))
