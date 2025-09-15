(ns artlib.noise-test
  (:require [artlib.acceleration.core :as acceleration]
            [artlib.cuda.noise :as noise]
            [artlib.image.core :refer [make-rgb-32f make-gray-32f]]
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
        (println out)
        ))

    #_(testing "sample cases"
      (testing "have metadata"
        (let [heightmap (make-gray-32f 3 3)]
          (imagez/set-pixels heightmap (into-array Float/TYPE [0 0 0
                                                               0 1 0
                                                               0 0 0]))
          (testing "from single threshold"
            (let [output (acceleration/compute-contour-lines accel heightmap 0.5)]
              (is (some? (meta output)))
              (is (some? (:threshold (meta output))))
              (is (= (:threshold (meta output)) 0.5))))

          (testing "from seq of thresholds"
            (let [output (acceleration/compute-contour-lines accel heightmap [0.2 0.5 0.8])]
              (is (every? #(some? (meta %)) output))
              (is (every? #(some? (:threshold (meta %))) output))
              (is (= [0.2 0.5 0.8] (mapv #(:threshold (meta %)) output)))))))

      (testing "3x3 with center pixel high"
        (let [heightmap (make-gray-32f 3 3)]
          (imagez/set-pixels heightmap (into-array Float/TYPE [0 0 0 
                                                               0 1 0 
                                                               0 0 0]))
          (let [output (acceleration/compute-contour-lines accel heightmap 0.5)]
            (is (some? output))
            (is (pos? (count output)))
            (is (= 4 (count output)))
            (is (= output
                   '([[1.0 1.5] [1.5 1.0]] 
                     [[1.5 1.0] [2.0 1.5]] 
                     [[1.5 2.0] [1.0 1.5]] 
                     [[2.0 1.5] [1.5 2.0]])))))))

    ))
