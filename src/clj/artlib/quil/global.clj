(ns artlib.quil.global)

(defmacro with-matrix [& body]
  `(do
     (q/push-matrix)
     (try
       ~@body
       (finally (q/pop-matrix)))))

(defmacro with-style [& body]
  `(do
     (q/push-style)
     (try
       ~@body
       (finally (q/pop-style)))))

