(ns artlib.geometry.barycentric)

(defn point->barycentric
  "Returns the barycentric coordinate of the point on the traingle defined
    by vertexes a, b, c."
  [vertex-a vertex-b vertex-c point]
  ;; TODO(2023-06-18): probably a better way to do this with core.matrix
  (let [[px py] point
        [ax ay] vertex-a
        [bx by] vertex-b
        [cx cy] vertex-c
        aa (+ 
            (* ax (- by cy)) 
            (* bx (- cy ay)) 
            (* cx (- ay by)))
        l1 (+
            (- (* bx cy) (* cx by))
            (* (- by cy) px)
            (* (- cx bx) py))
        l2 (+
            (- (* cx ay) (* ax cy))
            (* (- cy ay) px)
            (* (- ax cx) py))
        l3 (+ 
             (- (* ax by) (* bx ay))
             (* (- ay by) px)
             (* (- bx ax) py))]
    [(/ l1 aa) (/ l2 aa) (/ l3 aa)]))

