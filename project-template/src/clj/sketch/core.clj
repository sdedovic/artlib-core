(ns sketch.core
  (:require [quil.core :as q]
            [quil.middleware :as qm]
            [artlib.quil.middleware :refer [animation-mode]]
            [sketch.dynamic :as dynamic]))

(q/defsketch example
             :title "Sketch"
             :setup dynamic/setup
             :draw dynamic/draw
             :update dynamic/update-state
             :size [1400 1400]
             :animation {:render? false :dirname "EXAMPLE/1"}
             :middleware [qm/fun-mode animation-mode ])


