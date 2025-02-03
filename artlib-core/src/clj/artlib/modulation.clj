(ns artlib.modulation
 "This namespace houses higher-order functions for creating
    modulation generators. A modulation generator is a function
    hat accepts timing information and returns a value between
    0.0 and 1.0. Think an ADSR Envelope. 

  Supplied values should be considered durations and not
    instances.
  
  The returned modulation generators expect inputs relative to 
    their lifecycle, not relative to the start of the application.
    I.e. pass a diff from when the event happens, not the event 
    time.

  Time units are in milliseconds.")

(defn make-ad
  "Returns a parameterized Attack-Decay envelope generator.
  
  Inputs are durations in millis.
  
  The returned function accepts an input, in millis, relative
  to the start of the envelope."
  {:test #(let [env (make-ad 1.25 2.0)] 
            (assert (= 0.0 (env 0.0)))
            (assert (= 0.8 (env 1.0)))
            (assert (= 0.5 (env 2.25)))
            (assert (= 0.0 (env 4.0)))) }
  [attack decay]
  (fn [input]
    (cond
     (<= input 0) 0.0
     (<= input attack) (/ input attack)
     :else (max 0.0 (/ (- input attack decay) (- decay))))))

(defn make-a
  "Special case of `make-ad` where the decay value is set to 0.0"
  [attack]
  (make-ad attack 0.0))

(defn make-d
  "Special case of `make-ad` where the attack value is set to 0.0"
  [decay]
  (make-ad 0.0 decay))

(defn make-sin
  "Returns a parameterized sine wave LFO.

  Input `freq` is in hertz. Input `phase` is normalized, [0, 1].

  The returned function accepts an input, in millis, relative to
  the start of the wave generator."
  {:test #(let [lfo (make-sin 0.5 0.25)]
            (assert (= 1.0 (lfo 0.0)))
            (assert (= 0.5 (lfo 500.0)))
            (assert (= 0.0 (lfo 1000.0)))
            (assert (= 0.5 (lfo 1500.0)))
            (assert (= 1.0 (lfo 2000.0)))
            (assert (= 1.0 (lfo 4000.0))))}
  ([freq]
   (make-sin freq 0))
  ([freq phase]
   (fn [input]
     (let [input (/ input 1000) ; because we want millis
           theta (* 2 Math/PI (+ phase (* input freq)))]
     (float (Math/fma 0.5 (Math/sin (double theta)) 0.5))))))

(defn make-sin-sync
  "Returns a parameterized sine wave LFO. See `make-sin`.

  Input `division` is a fraction of a beat. Input phase is 
    normalized, [0, 1].

  The returned function accepts an input, in millis, relative to
  the start of the wave generator."
  {:test #(let [lfo (make-sin-sync 120 1/4)]
            ; 120 bpm, 1/4 division so
            ; quarter-note is 125 ms
            (assert (= 0.5 (lfo 0)))
            (assert (= 1.0 (lfo 31.25)))
            (assert (= 0.5 (lfo 62.5)))
            (assert (= 0.0 (lfo 93.75)))
            (assert (= 0.5 (lfo 125))))}
  ([bpm]
   (make-sin-sync bpm 1 0))
  ([bpm division]
   (make-sin-sync bpm division 0))
  ([bpm division phase]
   (let [freq (/ bpm 60 division)]
     (make-sin freq phase))))

