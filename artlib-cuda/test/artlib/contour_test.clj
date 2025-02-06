(ns artlib.contour-test
  (:require [artlib.acceleration.core :as acceleration]
            [artlib.cuda.contour :as contour]
            [clojure.test :refer :all]))

(deftest contour-test
  (let [accel (acceleration/create (first (acceleration/providers)))
        {ctx :ctx device :device} accel]
    (testing "module creation"
      (is (some? (contour/create-module ctx))))))
