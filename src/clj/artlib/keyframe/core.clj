(ns artlib.keyframe.core)

(def keyframe
  [{:frame 0  :value 0}
   {:frame 60 :value 100}
   {:frame 120 :value 250}])

(defn- make-4p-bezier
  "Given a set of 4 bezier curve control points,
    returns a function that takes in a t value [0-1]
    and returns the value of the bezier curve defined
    by the control points at the t value."
  [a b c d]
  (fn [t]
    (+
     (* (Math/pow (- 1 t) 3) a)
     (* 3 (Math/pow (- 1 t) 2) t b)
     (* 3 (- 1 t) (Math/pow t 2) c)
     (* (Math/pow t 3) d))))

(defn- make-bezier
  "Gien a set of 4 2-dimentional bezier curve control points,
    returns a function that takes in a value [0-1] and 
    returns the 2d coordinate of the bezier curve as
    defined by the control points at the t value."
  [a b c d]
  (let [[ax ay] a
        [bx by] b
        [cx cy] c
        [dx dy] d
        x-fn (make-4p-bezier ax bx cx dx)
        y-fn (make-4p-bezier ay by cy dy)]
    (fn [t]
      [(x-fn t) (y-fn t)])))

(defn lerp 
  "Linearly interpolates a point between [start, finish] at point t, where 
    t is between 0.0 and 1.0" 
  [start finish t] 
  (+ (* (- 1.0 t) start) (* t finish)))

(defn value-at
  "Given a keyframe lane and the frame number, returns the
    value at the supplied frame."
  [lane frame]
  (cond 
    (<= frame (:frame (first lane)))
    (:value (first lane))
    
    (>= frame (:frame (last lane)))
    (:value (last lane))

    :else
    (if-let [exact (->> lane
                        (filter #(= (:frame %) frame))
                        first)]
      (:value exact)
      (let [[a b] (->> (partition 2 1 lane)
                       (filter (fn [[a b]]
                                 (and 
                                   (> frame (:frame a))
                                   (< frame (:frame b)))))
                       first)
             t (/ (- frame (:frame a)) (- (:frame b) (:frame a)))
             p-linear (lerp (:value a) (:value b) t)
             quarter (/ (- (:frame b) (:frame a)) 4)
             bezier (make-bezier [(:frame a) (:value a)]
                                 [(+ (:frame a) quarter) (:value a)]
                                 [(- (:frame b) quarter) (:value b)]
                                 [(:frame b) (:value b)])
             [_ p-bezier] (bezier t) 
             left-amount (if (nil? (:bezier a)) 0 1)
             right-amount (if (nil? (:bezier b)) 0 1)
             blend-amt (lerp left-amount right-amount t)
             value (lerp p-linear p-bezier blend-amt)]
         value))))
