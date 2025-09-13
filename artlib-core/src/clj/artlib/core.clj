(ns artlib.core
  (:require [artlib.acceleration.core :as accel]))

(def ^:private accelerator-atom (atom nil))

;; TODO: another fn to filter accelerator by name / gen-interface
;; TODO: support options passed to acceleration/create
(defn create-accelerator
  "Create and cache a harware accelerator, returning it. If none is available, returns nil."
  []
  (if-let [provider (first (accel/providers))]
    (do 
      ;; TODO: switch to logger instead of stdout
      (println "Hardware accelerator found:" (accel/name provider))

      (let [impl (accel/create provider)]
        (reset! accelerator-atom impl)))
    nil))

(defn accelerator
  "Get hardware accelerator or nil if one is not provided."
  []
  @accelerator-atom)
