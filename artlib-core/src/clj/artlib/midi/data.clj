(ns artlib.midi.data
  (:require [artlib.midi.core :as core]))

(defn- ->note-ms
  [ms-per-tick note]
  (-> note
      (update :start-time * ms-per-tick)
      (update :duration * ms-per-tick)
      (update :end-time * ms-per-tick)))

(defn get-notes-ms
  "Finds all pairs on note on and note off events in the `track-idx` indexed
  track that corresponds to the same note values (pitch).

  Notice that the first track of a MIDI file typically doesn't have any notes."
  [mid-file track-idx]
  (let [ms-per-tick (core/ms-per-tick mid-file)
        notes (core/get-notes mid-file track-idx)]
    (mapv #(->note-ms ms-per-tick %) notes)))

(defn get-notes-ms-grouped
  "Same as get-notes-ms but will return a map of pitch to notes instead."
  [mid-file track-idx]
  (group-by #(:pitch %) (get-notes-ms mid-file track-idx)))

(defn note-nearest-ms
  "Return the most recently played note to the given timestamp, in millis."
  [notes ts]
  (last (first (filter
          (fn [[idx {start :start-time end :end-time}]]
            (and
             (>= ts start)
             (< ts (:start-time (nth notes (inc idx))))))
          (zipmap (range) notes)))))

(defn get-bpm
  "Query the BPM of the supplied mid file."
  [mid-file]
  (core/bpm mid-file))
