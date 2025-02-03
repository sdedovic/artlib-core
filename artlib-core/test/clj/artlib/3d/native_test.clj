(ns artlib.3d.native_test
  (:require [artlib.3d.native :as n]
            [clojure.test :refer [is deftest]]))

(deftest project-preserve-metadata
  (let [point (with-meta [0 1 2] {:foo true :bar "false" :baz 0.3})
        projected (n/project (n/perspective-fov) (n/look-at [0 5 5]) (n/mat4) point)]
    (is (= (meta point) (meta projected)))))
