(ns artlib.waveform-test
  (:require [artlib.testutils :refer [eq-float]]
            [artlib.waveform :as waveform]
            [clojure.test :refer [deftest is]]))

(deftest sin
  (is (eq-float (waveform/sin 2.123) (Math/sin 2.123)))
  (is (eq-float (waveform/sin 0)     (Math/sin 0)))
  (is (eq-float (waveform/sin 31.3)  (Math/sin 31.3))))

(deftest sin-norm
  (is (eq-float (waveform/sin-norm 2.123) (+ 0.5 (* 0.5 (Math/sin 2.123)))))
  (is (eq-float (waveform/sin-norm 0)     (+ 0.5 (* 0.5 (Math/sin 0)))))
  (is (eq-float (waveform/sin-norm 31.3)  (+ 0.5 (* 0.5 (Math/sin 31.3))))))

(deftest cos
  (is (eq-float (waveform/cos 2.123) (Math/cos 2.123)))
  (is (eq-float (waveform/cos 0)     (Math/cos 0)))
  (is (eq-float (waveform/cos 31.3)  (Math/cos 31.3))))

(deftest cos-norm
  (is (eq-float (waveform/cos-norm 2.123) (+ 0.5 (* 0.5 (Math/cos 2.123)))))
  (is (eq-float (waveform/cos-norm 0)     (+ 0.5 (* 0.5 (Math/cos 0)))))
  (is (eq-float (waveform/cos-norm 31.3)  (+ 0.5 (* 0.5 (Math/cos 31.3))))))

(deftest tri
  (is (eq-float (waveform/tri 0)                  0))
  (is (eq-float (waveform/tri (* Math/PI 1/2))    1))
  (is (eq-float (waveform/tri Math/PI)            0))
  (is (eq-float (waveform/tri (* Math/PI 3/2))    -1))
  (is (eq-float (waveform/tri (* Math/PI 2))      0))
  (is (eq-float (waveform/tri (* Math/PI 5/2))    1))
  (is (eq-float (waveform/tri (* Math/PI 3))      0))
  (is (eq-float (waveform/tri (* Math/PI 7/2))    -1)))

(deftest tri-norm
  (is (eq-float (waveform/tri-norm 0)                  0))
  (is (eq-float (waveform/tri-norm (* Math/PI 1/2))    0.5))
  (is (eq-float (waveform/tri-norm Math/PI)            1))
  (is (eq-float (waveform/tri-norm (* Math/PI 3/2))    0.5))
  (is (eq-float (waveform/tri-norm (* Math/PI 2))      0))
  (is (eq-float (waveform/tri-norm (* Math/PI 5/2))    0.5))
  (is (eq-float (waveform/tri-norm (* Math/PI 3))      1))
  (is (eq-float (waveform/tri-norm (* Math/PI 7/2))    0.5)))