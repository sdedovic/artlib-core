(ns artlib.noise.opensimplex-test
  (:require [artlib.testutils]
            [artlib.noise.opensimplex :refer :all]
            [clojure.test :refer [is deftest testing]]))

(deftest opensimplex-test
  (testing "Bindings are functioning"
    (testing "Range of regular noise functions is [-1, 1]."
      (dotimes [n 100]
        (is (> 1 (noise2 (rand) (/ n 43)) -1))
        (is (> 1 (noise2-improve-x 1.3 (/ n 43)) -1))

        (is (> 1 (noise3 (rand) (/ n 43) -3.2) -1))
        (is (> 1 (noise3-improve-xy 1.3 (/ n 43) -5.2) -1))
        (is (> 1 (noise3-improve-xz 2.3 (/ n 43) -7.2) -1))

        (is (> 1 (noise4 (rand) (/ n 43) -3.2 1.1) -1))
        (is (> 1 (noise4-improve-xyz (rand) (/ n 43) -1.2 3.1) -1))
        (is (> 1 (noise4-improve-xyz-improve-xy (rand) (/ n 43) -2.2 5.1) -1))
        (is (> 1 (noise4-improve-xyz-improve-xz (rand) (/ n 43) -4.2 7.1) -1))
        (is (> 1 (noise4-improve-xy-improve-zw (rand) (/ n 43) -5.2 9.1) -1))))

    (testing "Range of normalized noise functions is [-1, 1]."
      (dotimes [n 100]
        (is (> 1 (noise2-norm (rand) (/ n 43)) 0))
        (is (> 1 (noise2-improve-x-norm 1.3 (/ n 43)) 0))

        (is (> 1 (noise3-norm (rand) (/ n 43) -3.2) 0))
        (is (> 1 (noise3-improve-xy-norm 1.3 (/ n 43) -5.2) 0))
        (is (> 1 (noise3-improve-xz-norm 2.3 (/ n 43) -7.2) 0))

        (is (> 1 (noise4-norm (rand) (/ n 43) -3.2 1.1) 0))
        (is (> 1 (noise4-improve-xyz-norm (rand) (/ n 43) -1.2 3.1) 0))
        (is (> 1 (noise4-improve-xyz-improve-xy-norm (rand) (/ n 43) -2.2 5.1) 0))
        (is (> 1 (noise4-improve-xyz-improve-xz-norm (rand) (/ n 43) -4.2 7.1) 0))
        (is (> 1 (noise4-improve-xy-improve-zw-norm (rand) (/ n 43) -5.2 9.1) 0))))))