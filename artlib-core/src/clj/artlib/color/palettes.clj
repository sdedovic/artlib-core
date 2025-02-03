(ns artlib.color.palettes
  "This namespace contains an implementation of procedural 
    periodic color palettes. It is based on a Inigo Quilez
    article from 1999.

  A color palette is simply a unary function that accepts an
    input value and returns an RGB vec. As the input runs 
    from 0 to 1 (normalized palette index), the palette
    oscilates.

  See https://iquilezles.org/articles/palettes/")

(defn make-palette
  "Returns a color palette based on the following function:
    color(t) = a + b ⋅ cos[ 2π(c⋅t+d)]

  This returns a unary function. All inputs and outpus are 
    normalized (0.0 to 1.0)."
  
  ([[a1 a2 a3] [b1 b2 b3] [c1 c2 c3] [d1 d2 d3]] 
   (make-palette a1 a2 a3 b1 b2 b3 c1 c2 c3 d1 d2 d3)) 
  ([a1 a2 a3 b1 b2 b3 c1 c2 c3 d1 d2 d3] 
   (fn [t]
    (let [o1 (+ a1 (* b1 (Math/cos (* Math/PI 2 (* c1 (+ t d1))))))
          o2 (+ a2 (* b2 (Math/cos (* Math/PI 2 (* c2 (+ t d2))))))
          o3 (+ a3 (* b3 (Math/cos (* Math/PI 2 (* c3 (+ t d3))))))]
      [o1 o2 o3]))))

(def rainbow
  "Sample palette."
  (make-palette 
    0.5   0.5   0.5	
    0.5   0.5   0.5	
    1.0   1.0   1.0	
    0.00  0.33  0.67))

(def a
  "Sample palette."
  (make-palette 
    0.5   0.5   0.5	
    0.5   0.5   0.5	
    1.0   1.0   1.0
    0.00  0.10  0.20))

(def b
  "Sample palette."
  (make-palette
    0.5   0.5   0.5
    0.5   0.5   0.5	
    1.0   1.0   1.0	
    0.30  0.20  0.20))

(def c
  "Sample palette."
  (make-palette
    0.5   0.5   0.5	
    0.5   0.5   0.5	
    1.0   1.0   0.5	
    0.80 0.90 0.30))

(def d
  "Sample palette."
  (make-palette
    0.5   0.5   0.5	
    0.5   0.5   0.5	
    1.0   0.7   0.4	
    0.00  0.15  0.20))

(def e
  "Sample palette."
  (make-palette
    0.5   0.5   0.5	
    0.5   0.5   0.5	
    2.0   1.0   0.0	
    0.50  0.20  0.25))

(def f
  "Sample palette."
  (make-palette
    0.8   0.5   0.4	
    0.2   0.4   0.2	
    2.0   1.0   1.0	
    0.00  0.25  0.25))
