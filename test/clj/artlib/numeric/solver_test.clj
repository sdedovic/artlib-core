(ns artlib.numeric.solver-test
  (:require [artlib.testutils :refer [eq-float]]
            [artlib.numeric.solver :refer :all]
            [clojure.test :refer [is deftest testing]]))

(deftest simple-harmonic-oscillator
  (let [
        ;; These equations are derived from the Lagrangian of
        ;;  a simple harmonic oscillator. L = T - V.
        ;;  T is the potential energy defined as 1/2*k*x^2 where k is the spring constant.
        ;;  V is the kinetic energy defined as 1/2*m*v^2 where v is the velocity.
        ;;
        ;; The first-order differential equations stat the dx=v_x(t), dy=v_y(t),
        ;;  dv_x=(-kx)/m, and dv_y=(-1ky)/m. The following equations represent this.
        eq (fn [m k]
             (->first-order
               (fn [t [x y vx vy]]
                 vx)
               (fn [t [x y vx vy]]
                 vy)
               (fn [t [x y vx vy]]
                 (/ (* -1 k x) m))
               (fn [t [x y vx vy]]
                 (/ (* -1 k y) m))))]
    (testing "solve future state"
      (let [solve (solver eq 2.0 1.0)
            [t x y vx vy] (solve 1 [1 2 3 4] 11)]

        ;; solved in SICM Chapter 1.7 (page 74)
        (is (eq-float t  11.0))
        (is (eq-float x  3.71279166))
        (is (eq-float y  5.42062082))
        (is (eq-float vx 1.61480309))
        (is (eq-float vy 1.81891037))))

    (testing "evolver with intermediate steps"
      (let [evolve ((evolver eq 2.0 1.0)
                    1 [1 2 3 4] 11)]
        (is (fn? evolve))

        (testing "n=0, initial state of evolver"
          (let [[t x y vx vy] (evolve 0)]
            (is (eq-float t  1))
            (is (eq-float x  1))
            (is (eq-float y  2))
            (is (eq-float vx 3))
            (is (eq-float vy 4))))

        (testing "n=1, final state of evolver"
          (let [[t x y vx vy] (evolve 1)]
            (is (eq-float t  11.0))
            (is (eq-float x  3.71279166))
            (is (eq-float y  5.42062082))
            (is (eq-float vx 1.61480309))
            (is (eq-float vy 1.81891037))))

        (testing "n=0.3, intermediate state of evolver"
          (let [[t x y vx vy] (evolve 0.3)]
            (is (eq-float t   4))
            (is (eq-float x   3.09265878))
            (is (eq-float y   3.77478912))
            (is (eq-float vx -2.17203379))
            (is (eq-float vy -3.29779980))))))))
