(ns artlib.color.dictionary
  "This namespace contiains utilities for interfacing with 
  definitions from the book \"A Dictionary of Colour Combinations\" 
  compiled by Sanzo Wada."
  (:require [clojure.string :refer [lower-case]]
            [clojure.java.io :as io]
            [clojure.data.json :as json])
  (:import [java.util HashMap ArrayList]
           [java.awt Color]))

(defrecord Dictionary [file json combinations])

(defn-
  load-combinations
  "Traverses the passed in colors to load a map of color combinations.
  This uses gross mutable java types internally for performance."
  [colors]
  (let [combinations (HashMap.)]
    (doseq [color colors]
      (doseq [idx (:combinations color)]
        (let [entries (.computeIfAbsent
                        combinations
                        idx
                        (reify
                          java.util.function.Function
                          (apply [this v] (ArrayList.))))]
          (.add entries color))))
    (apply hash-map 
           (mapcat (fn [entry]
                  (let [k (keyword (str (.getKey entry)))
                        v (into [] (.getValue entry))]
                    [k v])) 
                combinations))))
(defn 
  init
  "Load the color dictionary into memory."
  (^Dictionary [] (init "color_dictionary.json"))
  (^Dictionary [^String resource-path]
   (let [url (io/resource resource-path)
         json (json/read-json (io/reader url))
         combinations (load-combinations json)]
     (map->Dictionary {:file url 
                       :json json
                       :combinations combinations}))))

(defn 
  cmyk->rgb
  "Convert a CMYK color to RGB values. 
  Defaults range for CMYK is [0, 100] and default range for RGB is [0, 256)."
  ([c m y k]
   (cmyk->rgb 100 255 c m y k))
  ([cmyk-scale rgb-scale c m y k]
   (let [k-amt (- 1 (/ k cmyk-scale))
         r (* rgb-scale (- 1 (/ c cmyk-scale)) k-amt)
         g (* rgb-scale (- 1 (/ m cmyk-scale)) k-amt)
         b (* rgb-scale (- 1 (/ y cmyk-scale)) k-amt)]
     [r g b])))

(defn
  color->rgb
  "Converts a color to RGB values. Range for RGB is [0, 256)."
  [{c :c m :m y :y k :k}]
  (cmyk->rgb c m y k))

(defn 
  color->Color
  "Convert a color to an instance of java.awt.Color."
  ^Color [color]
  (let [[r g b] (map int (color->rgb color))]
    (Color. r g b)))

(defn
  color->hsb
  "Convert a color to HSB values.
  Range for H is [0, 360) and for S, B is [0, 100)."
  [color]
  (let [[r g b] (map float (color->rgb color))
        [h s b] (Color/RGBtoHSB (float r) (float g) (float b) nil)
        h (* 360 h)
        s (* 100 s)
        b (* 100 b)]
    [h s b]))

(defn
  get-color
  [^Dictionary color-dictionary ^String color-name]
  (let [search-name (lower-case color-name)]
    (->> (:json color-dictionary)
         (filter (fn [entry] (= search-name (lower-case (:name entry))))) 
         first)))

(defn 
  get-color-rgb
  "Retrieve the RGB values of a color by name."
  [^Dictionary color-dictionary ^String color-name]
  (when-let [match (get-color color-dictionary color-name)]
    (color->rgb match)))

(defn 
  get-color-java
  "Retrieve Color by name."
  ^Color [^Dictionary color-dictionary ^String color-name]
  (when-let [match (get-color color-dictionary color-name)]
    (color->Color match)))

(defn 
  get-color-hsb
  "Retrieve the HSB values of a color by name."
  ^Color [^Dictionary color-dictionary ^String color-name]
  (when-let [match (get-color color-dictionary color-name)]
    (color->hsb match)))

(defn
  get-combination
  "Retrieve the color combination. Combinations are from :1 to :348."
  [^Dictionary color-dictionary combination]
  (get (:combinations color-dictionary) combination))

(defn
  get-combination-rgb
  "Retrieve the color combination. Combinations are from :1 to :348."
  [^Dictionary color-dictionary combination]
  (when-let [match (get-combination color-dictionary combination)]
    (map color->rgb match)))

(defn
  get-combination-java
  "Retrieve the color combination. Combinations are from :1 to :348."
  [^Dictionary color-dictionary combination]
  (when-let [match (get-combination color-dictionary combination)]
    (map color->Color match)))

(defn
  get-combination-hsb
  "Retrieve the color combination. Combinations are from :1 to :348."
  [^Dictionary color-dictionary combination]
  (when-let [match (get-combination color-dictionary combination)]
    (map color->hsb match)))
