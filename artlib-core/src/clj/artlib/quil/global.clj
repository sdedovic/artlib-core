(ns artlib.quil.global
  (:require [quil.core :as q]))

(defmacro with-matrix [& body]
  `(do
     (q/push-matrix)
     (try
       ~@body
       (finally (q/pop-matrix)))))

(defmacro with-style [& body]
  `(do
     (q/push-style)
     (try
       ~@body
       (finally (q/pop-style)))))

; TODO: move to new artlib.quil.random namespace?
;; Copyright (c) 2016 Tyler Hobbs
;; https://github.com/thobbs/genartlib/blob/master/LICENSE
;; https://github.com/thobbs/genartlib/blob/master/src/clj/genartlib/random.clj#L6
(defn gauss
  "Samples a single value from a Gaussian distribution with the given mean
   and variance"
  [mean variance]
  (+ mean (* variance (q/random-gaussian))))
