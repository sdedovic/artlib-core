(ns artlib.midi.core
  (:require [clojure.java.io :as jio])
  (:import [javax.sound.midi MetaMessage ShortMessage MidiSystem MidiFileFormat Sequence Track]
           [java.nio ByteBuffer ByteOrder]
           [artlib.midi Note Midi]
           (java.io File)))

(defn get-info
  "Return the MIDI file format of the specified argument. Argument will be coerced to file."
  [mid-file]
  (let [file-format (MidiSystem/getMidiFileFormat ^File (jio/as-file mid-file))
        byte-length (.getByteLength file-format)
        division-type (.getDivisionType file-format)
        microsecond-length (let [value (.getMicrosecondLength file-format)]
                             (if (= MidiFileFormat/UNKNOWN_LENGTH value)
                               :unknown-length
                               value))
        resolution (.getResolution file-format)
        f-type (.getType file-format)]
    { :byte-length byte-length 
      :division-type division-type
      :microsecond-length microsecond-length
      :resolution resolution
      :type f-type }))

(defn- ->division-type [mid-file]
  (case mid-file
    Sequence/PPQ :ppq
    Sequence/SMPTE_24 :smpte-24
    Sequence/SMPTE_25 :smpte-25
    Sequence/SMPTE_30DROP :smpte-30-drop
    Sequence/SMPTE_30 :smpte-30-drop
    :unknown))

;; Note-off event:
;; MIDI may send both Note-On and Velocity 0 or Note-Off.
;;
;; http://www.jsresources.org/faq_midi.html#no_note_off
(defn- ->midi-msg
  "Make a clojure map out of a midi ShortMessage object."
  [^ShortMessage obj & [ts]]
  (let [ch     (.getChannel obj)
        cmd    (.getCommand obj)
        d1     (.getData1 obj)
        d2     (.getData2 obj)
        status (.getStatus obj)]
  { :type :message
    :channel   ch
    :command   cmd 
    :note      d1
    :velocity  d2
    :data-1     d1
    :data-2     d2
    :status    status
    :timestamp ts }))

(defn- ->meta-msg
  [^MetaMessage obj]
  { :type :meta-message
    :data (.getData obj)
    :meta-type (.getType obj) })


; TODO: Figure out how to detect the strange end-of-track msg
; TODO: Find better documentation for the meta messages so we can
; either make sense of them or disregard them if unimportant.
(defn- midi-event
  [event]
  (let [msg (.getMessage event)
        msg (cond
              (= (type msg) MetaMessage) (->meta-msg msg)
              (instance? ShortMessage msg) (->midi-msg msg)
              :default {:type :end-of-track})]
    (assoc msg :timestamp (.getTick event))))

(defn- ->track [^Track track]
  (let [size (.size track)]
    { :type :midi-track
      :size size 
      :events (into [] (for [i (range size)] (midi-event (.get track i)))) }))

(defn- ->sequence
  "Return a MIDI sequence from the supplied argument. Argument will be coerced to file.
  See: https://github.com/JuliaMusic/MIDI.jl/blob/master/src/midifile.jl#L122"
  [mid-file]
  (let [midi-sequence (MidiSystem/getSequence ^File (jio/as-file mid-file))
        division (.getDivisionType midi-sequence)
        microsecond-length (.getMicrosecondLength midi-sequence)
        resolution (.getResolution midi-sequence)
        tick-length (.getTickLength midi-sequence)
        tracks (.getTracks midi-sequence)]
    { :division-type (->division-type division)
      :microsecond-length microsecond-length
      :resolution resolution
      :tick-length tick-length 
      :tracks (mapv ->track (vec tracks)) }))

(defn- byte-array-as-int
  "Returns an integer represented by the supplied byte array"
  [^bytes in]
  (let [size (int Integer/BYTES)
        padded (concat (repeat (- size (count in)) 0x00) in)
        as-int (.getInt (ByteBuffer/wrap (byte-array padded)))]
    as-int))

(defn bpm
  "Determine BPM of file by searching for Set Tempo MIDI event (0xFF 0x51 0x03) 
  and parsing the payload if possible."
  [mid-file]
  (let [midi-sequence (MidiSystem/getSequence ^File (jio/as-file mid-file))
        track (->track (aget (.getTracks midi-sequence) 0))
        event (first
                (filter 
                  #(and (= :meta-message (:type %)) (= 0x51 (:meta-type %))) 
                  (:events track)))
        data (:data event)
        data (if-not (= 0x00 (aget data 0)) 
               (concat [0x00] data) ; first entry should be 0x00
               (let [[end _] (last (filter (fn [[k v]] (= 0x00 v)) (zipmap (range) data)))]
                 (drop end data)))
        as-byte-array (byte-array data)
        bpm-float (float (/ 60000000 (byte-array-as-int as-byte-array)))]
    (/ (Math/round (* 100.0 bpm-float)) 100.0)))

(defn ms-per-tick "Returns the number of milliseconds per tick. This is affected by BPM"
  [mid-file]
  (let [bpm-v (bpm mid-file)
        ;; TODO: lets not assume ticks per quarter note and actually read that
        ;; info from the file and handle correctly
        ticks-per-quarternote (:resolution (get-info mid-file))]
    (/ (* 60 1000) (* bpm-v ticks-per-quarternote))))

(defn- ->note
  [^Note note]
  { :pitch (.getData1 note)
    :velocity (.getData2 note)
    :start-time (.getTrackTime note)
    :duration (.getDuration note)
    :end-time (+ (.getTrackTime note) (.getDuration note)) })

(defn get-notes 
  "Find all NOTEON and NOTEOFF MIDI events in the `track-idx` indexed track
  that correspond to the same note value (pitch).
  
  There are special cases where NOTEOFF is actually
  encoded as NOTEON with 0 velocity, but `get-notes` takes care of this.
 
  Notice that the first track of a MIDI file typically doesn't have any notes."
  [mid-file track-idx]
  (let [midi-sequence (MidiSystem/getSequence ^File (jio/as-file mid-file))
        notes (Midi/getNotes (aget (.getTracks midi-sequence) track-idx))] 
    (mapv ->note notes)))

(defn- get-notes-grouped
  "Same as get-notes but will return a map of pitch to notes instead."
  [mid-file track-idx]
  (group-by #(:pitch %) (get-notes mid-file track-idx)))

(defn- note-nearest [notes tick-num]
  (first (filter 
          (fn [[idx {start :start-time end :end-time}]]
            (and
             (>= tick-num start)
             (< tick-num (:start-time (nth notes (inc idx))))))
          (zipmap (range) notes))))

