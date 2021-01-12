(ns sketch.dynamic
  (:require [clojure.java.shell :refer [sh]]
            ; [genartlib.algebra :refer :all]
            ; [genartlib.curves :refer :all]
            ; [genartlib.geometry :refer :all]
            ; [genartlib.random :refer :all]
            [artlib.quil.global :refer :all]
            [genartlib.util :refer [w h]]
            [quil.core :as q]))

(defn draw [state]
  (with-style
    (q/stroke 0 0 0 1.0) ;; HSBA - Hue [0-360), Saturation [0-100],
                         ;;        Brightness [0-100], Alpha [0.0-1.0]

    (q/rect (w 0.1) (h 0.1) (w 0.8) (h 0.8))))
