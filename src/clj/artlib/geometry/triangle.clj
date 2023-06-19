(ns artlib.geometry.triangle
  (:import [java.util SplittableRandom]))

(defn-
  gen-u-vals
  "Helper function. Generates uniformly random [u1 u2] ~ U(0,1).
  If u1 + u2 > 1, apply the transformation u1 → 1 - u1 and u2 → 1 - u2."
  ([rng]
   (let [u1 (.nextDouble rng)
         u2 (.nextDouble rng)]
     (if (> (+ u1 u2) 1)
       [(- 1 u1) (- 1 u2)]
       [u1 u2])))
  ([]
   (gen-u-vals (SplittableRandom.))))

(defn
  random-point
  "Returns a random vec of a random point inside of the specified triangle.
  See: https://blogs.sas.com/content/iml/2020/10/19/random-points-in-triangle.html"
  [vertex-a vertex-b vertex-c]
  (let [[ax ay] vertex-a
        [bx by] vertex-b
        [cx cy] vertex-c
        [tax tay] [(- bx ax) (- by ay)]
        [tbx tby] [(- cx ax) (- cy ay)]
        [u1 u2] (gen-u-vals)
        [wx wy] [(+ (* u1 tax) (* u2 tbx)) (+ (* u1 tay) (* u2 tby))]]
    [(+ ax wx) (+ ay wy)]))
