(ns artlib.color.spaces
  "This namespace contains various functions for transforming
    color information across different representations and 
    color spaces. 
  
  Note: by default, a \"color\" is a 3-ple, 4-ple, or 5-ple of
    numbers (with the last entry being the Alpha channel) whereas 
    a \"Color\" refers to the java.awt.Color class."
  (:import [java.util HashMap ArrayList]
           [java.awt Color]))

; ==== java.awt.Color Transforms ==== ;
(defn
  Color->rgbi
  "Convert java.awt.Color to RGB.
    Range of RGB is [0, 256)."
  [^Color color]
  [(.getRed color) (.getGreen color) (.getBlue color)])

(defn
  Color->rgbf
  "Convert java.awt.Color to RGB.
    Range of RGB is [0.0, 1.0]."
  [^Color color]
  (let [[r g b] (Color->rgbi color)]
    [(float (/ r 255)) (float (/ g 255)) (float (/ b 255))]))

(defn
  Color->rgb
  "Convert java.awt.Color to RGB.
    Range of RGB is [0, 256).
    Alias of Color->rgbi."
  [^Color color]
  (Color->rgbi color))

(defn
  Color->rgbai
  "Convert java.awt.Color to RGBA.
    Range of RGBA is [0, 256)."
  [^Color color]
  [(.getRed color) (.getGreen color) (.getBlue color) (.getAlpha color)])

(defn
  Color->rgbaf
  "Convert java.awt.Color to RGBA.
    Range of RGBA is [0.0, 1.0]."
  [^Color color]
  (let [[r g b a] (Color->rgbai color)]
    [(float (/ r 255)) (float (/ g 255)) (float (/ b 255)) (float (/ a 255))]))

(defn
  Color->rgba
  "Convert java.awt.Color to RGBA.
    Range of RGBA is [0, 256).
    Alias of Color->rgbai."
  [^Color color]
  (Color->rgbai color))

(defn
  Color->hsb
  "Convert java.awt.Color to HSB.
    Range for H is [0, 360) and for S, B is [0, 100]."
  [^Color color]
  (let [[r g b] (Color->rgbi color)
        [h s b] (Color/RGBtoHSB r g b nil)]
    [(* 360 h) (* 100 s) (* 100 b)]))

(defn
  Color->hsba
  "Convert java.awt.Color to HSBA.
    Range for H is [0, 360).
    Range for S, B is [0, 100].
    Range for A is [0.0, 1.0)."
  [^Color color]
  (let [[r g b a] (Color->rgbai color)
        [h s b] (Color/RGBtoHSB r g b nil)]
    [(* 360 h) (* 100 s) (* 100 b) (float (/ a 255))]))

; ==== CMYK Transforms ==== ;
(defn 
  cmyk->Color
  "Convert a CMYK color to java.awt.Color. 
     Range for CMYK values is [0, 100]."
  [cmyk]
  (let [[c m y k] cmyk
        k-amt (- 1 (/ k 100))
        r (- 1 (/ c 100) k-amt)
        g (- 1 (/ m 100) k-amt)
        b (- 1 (/ y 100) k-amt)]
    (Color. (float r) (float g) (float b))))

(defn 
  cmyk->rgbi
  "Convert a CMYK color to RGB values. 
    Range for CMYK values is [0, 100].
    Range for RGB is [0, 256)."
  [cmyk]
  (->> cmyk
       cmyk->Color
       Color->rgbi))
(defn 
  cmyk->rgbf
  "Convert a CMYK color to RGB values. 
    Range for CMYK values is [0, 100].
    Range for RGB is [0.0, 1.0]."
  [cmyk]
  (->> cmyk
       cmyk->Color
       Color->rgbf))

(defn 
  cmyk->rgb
  "Convert a CMYK color to RGB values. 
    Range for CMYK values is [0, 100].
    Range for RGB is [0, 256).
    Alias of cmyk->rgbi."
  [cmyk]
  (cmyk->rgbi cmyk))

(defn 
  cmyk->rgbai
  "Convert a CMYK color to RGBA values. 
    Range for CMYK values is [0, 100].
    Range for RGBA is [0, 256)."
  [cmyk]
  (->> cmyk
       cmyk->Color
       Color->rgbai))
(defn 
  cmyk->rgbaf
  "Convert a CMYK color to RGBA values. 
    Range for CMYK values is [0, 100].
    Range for RGBA is [0.0, 1.0]."
  [cmyk]
  (->> cmyk
       cmyk->Color
       Color->rgbaf))

(defn 
  cmyk->rgba
  "Convert a CMYK color to RGBA values. 
    Range for CMYK values is [0, 100].
    Range for RGBA is [0, 256).
    Alias of cmyk->rgbai."
  [cmyk]
  (cmyk->rgbai cmyk))

(defn
  cmyk->hsb
  "Convert a CMYK color to HSB values.
    Range for CMYK values is [0, 100].
    Range for H is [0, 360) and for S, B is [0, 100]."
  [cmyk]
  (->> cmyk
       cmyk->Color
       Color->hsb))

(defn
  cmyk->hsba
  "Convert a CMYK color to HSBA values.
    Range for CMYK values is [0, 100].
    Range for H is [0, 360).
    Range for S, B is [0, 100].
    Range for A is [0.0, 1.0)."
  [cmyk]
  (->> cmyk
       cmyk->Color
       Color->hsba))

; ==== RGB[A] Transforms ==== ;
(defn
  rgbi->Color
  "Convert a RGB color to java.awt.Color.
    Range for RGB values is [0, 256)."
  [rgb]
  (let [[r g b] (map int rgb)]
   (Color. r g b)))

(defn
  rgbf->Color
  "Convert a RGB color to java.awt.Color.
    Range for RGB values is [0.0, 1.0]."
  [rgb]
  (let [[r g b] (map float rgb)]
   (Color. r g b)))

(defn
  rgb->Color
  "Convert a RGB color to java.awt.Color.
    Range for RGB values is [0, 256).
    Alias for rgbi->Color."
  [rgb]
  (rgbi->Color rgb))

(defn
  rgbai->Color
  "Convert a RGBA color to java.awt.Color.
    Range for RGBA values is [0, 256)."
  [rgba]
  (let [[r g b a] (map int rgba)]
   (Color. r g b a)))

(defn
  rgbaf->Color
  "Convert a RGBA color to java.awt.Color.
    Range for RGBA values is [0.0, 1.0]."
  [rgba]
  (let [[r g b a] (map float rgba)]
   (Color. r g b a)))

(defn
  rgba->Color
  "Convert a RGBA color to java.awt.Color.
    Range for RGBA values is [0, 256).
    Alias for rgbai->Color."
  [rgba]
  (rgbai->Color rgba))
