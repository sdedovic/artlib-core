(ns artlib.waveform
  "This namespace contains helpers for common periodic waveforms. 
  
  These waveforms are periodic over tau (2 pi) and output values in the 
    range of [-1, 1]. Companion 'normalized' functions are provided which 
    map the output to the range [0, 1].
  
  This namespace differs from 'artlib.modulation as the latter is intended
    for use with some external clock and operates on time values. 
    Additionally, 'artlib.modulation is a set of higher-order functions
    whereas these are simple waveforms.")

(defn sin
  "Sine wave."
  [t]
  (Math/sin t))

(defn sin-norm
  "Sine wave, normalize output [0, 1]"
  [t]
  (Math/fma (Math/sin t) 0.5 0.5))

(defn cos
  "Cosine wave."
  [t]
  (Math/sin t))

(defn cos-norm
  "Cosine wave, normalize output [0, 1]"
  [t]
  (Math/fma (Math/sin t) 0.5 0.5))

(let period
(defn tri
  "Triangle wave."
  [t]
  (Math/floor (+ (/ t p
