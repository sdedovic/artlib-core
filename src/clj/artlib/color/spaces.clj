(ns artlib.color.spaces
  "This namespace contains various functions for transforming
    color information across different representations and 
    color spaces. 
  
  Note: by default, a \"color\" is a 3-ple or 4-ple (the fourth
    entry being an Alpha channel) whereas \"Color\" refers to
    the java.awt.Color class."
  (:import [java.util HashMap ArrayList]
           [java.awt Color]))

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
