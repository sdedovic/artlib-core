(ns artlib.contour-test
  (:require [artlib.acceleration.core :as acceleration]
            [artlib.cuda.contour :as contour]
            [artlib.image.core :refer [make-rgb-32f make-gray-32f]]
            [mikera.image.core :as imagez]
            [clojure.test :refer :all])
  (:import [java.awt.image BufferedImage]))

(deftest contour-test
  (let [accel (acceleration/create (first (acceleration/providers)))
        {ctx :ctx device :device} accel]

    (testing "module creation"
      (is (some? (contour/create-module ctx))))

    (testing "basic execution flow"
      (let [rgb-32f (make-rgb-32f 10 10)
            gray-i8 (new BufferedImage 10 10 BufferedImage/TYPE_BYTE_GRAY)
            gray-32f (make-gray-32f 10 10)]
        (is (thrown? IllegalArgumentException (acceleration/compute-contour-lines accel rgb-32f 0.5)))
        (is (thrown? IllegalArgumentException (acceleration/compute-contour-lines accel gray-i8 0.5)))

        ;; set random data inside image
        (imagez/set-pixels gray-32f (into-array 
                                      Float/TYPE 
                                      (take 100 (repeatedly rand))))
        (is (some? (acceleration/compute-contour-lines accel gray-32f 0.5)))
        (is (pos? (count (acceleration/compute-contour-lines accel gray-32f 0.5))))
        (is (= 3 (count (acceleration/compute-contour-lines accel gray-32f [0.2 0.5 0.7]))))))


    (testing "sample cases"
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
                     [[2.0 1.5] [1.5 2.0]])))))))))
