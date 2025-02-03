(ns artlib.testutils)

(let [epsilon 1e-6]
  (defn eq-float
    "Truthiness of equality between two numbers using maching epsilon."
    [a b]
    (< (Math/abs (- (float a) (float b))) epsilon)))

(defn eq-int
  "Truthiness of equality between two numbers using rounding to nearest int."
  [a b]
  (= 
    (Math/round (float a))
    (Math/round (float b))))

(defn eq-vecf
  "Trutiness of equality of vector by testing float equality component wise."
  [a b]
  (if (not= (count a) (count b))
    false
    (->> (map vector a b)
         (map #(apply eq-float %))
         (filter false?)
         (empty?))))

(defn eq-veci
  "Trutiness of equality of vector by testing integer equality component wise."
  [a b]
  (if (not= (count a) (count b))
    false
    (->> (map vector a b)
         (map #(apply eq-int %))
         (filter false?)
         (empty?))))
