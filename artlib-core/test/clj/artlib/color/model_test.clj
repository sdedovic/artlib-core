
(ns artlib.color.model-test
  (:require [artlib.color.model :as m]
            [artlib.testutils :refer [eq-vecf]]
            [clojure.test :refer [is deftest testing]])
  (:import [java.awt Color]))

(deftest color-transformation-test
  (testing "Color transformations through different spaces"
           ;; Pure red color in different formats
           (let [red-cmyk [0 100 100 0]          ; Red in CMYK
                 red-rgb [255 0 0]               ; Red in RGB (8-bit)
                 red-rgbi [255 0 0]              ; Red in RGBi (8-bit)
                 red-rgbf [1.0 0.0 0.0]          ; Red in RGBf (float)
                 red-rgba [255 0 0 255]          ; Red in RGBA (8-bit)
                 red-rgbaf [1.0 0.0 0.0 1.0]     ; Red in RGBAf (float)
                 red-hsb [0 100 100]           ; Red in HSB
                 red-color (Color. 255 0 0)]     ; Red as Java Color

             ;; Test CMYK conversions
             (testing "CMYK conversions"
                      (is (eq-vecf red-hsb (m/cmyk->hsb red-cmyk)))
                      (is (eq-vecf red-rgbi (m/cmyk->rgbi red-cmyk)))
                      (is (eq-vecf red-rgba (m/cmyk->rgba red-cmyk)))
                      (is (eq-vecf red-rgbf (m/cmyk->rgbf red-cmyk)))
                      (is (instance? Color (m/cmyk->Color red-cmyk))))

             ;; Test Color object conversions
             (testing "Color object conversions"
                      (is (eq-vecf red-hsb (m/Color->hsb red-color)))
                      (is (eq-vecf red-rgb (m/Color->rgb red-color)))
                      (is (eq-vecf red-rgbi (m/Color->rgbi red-color)))
                      (is (eq-vecf red-rgba (m/Color->rgba red-color)))
                      (is (eq-vecf red-rgbf (m/Color->rgbf red-color)))
                      (is (eq-vecf red-rgbaf (m/Color->rgbaf red-color))))

             ;; Test round-trip conversions
             (testing "Round-trip conversions"
                      (let [original-color red-color
                            ;; Convert through multiple spaces and back
                            round-trip-color (-> red-cmyk
                                                 m/cmyk->rgbi
                                                 m/rgbi->Color
                                                 m/Color->hsba
                                                 m/hsba->Color)]
                        (is (= (m/Color->rgb original-color)
                               (m/Color->rgb round-trip-color))
                            "Colors should be equal after round-trip conversion")))

             ;; Test with alpha channel
             (testing "Alpha channel preservation"
                      (let [semi-transparent-rgba [255 0 0 128]
                            color-with-alpha (m/rgba->Color semi-transparent-rgba)]
                        (is (= 128 (last (m/Color->rgba color-with-alpha)))
                            "Alpha channel should be preserved")))

             ;; Test edge cases
             (testing "Edge cases"
                      ;; Black
                      (let [black-cmyk [0 0 0 100]
                            black-rgb [0 0 0]]
                        (is (= black-rgb (m/Color->rgb (m/cmyk->Color black-cmyk)))))

                      ;; White
                      (let [white-cmyk [0 0 0 0]
                            white-rgb [255 255 255]]
                        (is (= white-rgb (m/Color->rgb (m/cmyk->Color white-cmyk)))))))))

(deftest color-value-ranges-test
  (testing "Color value ranges"
           ;; Test CMYK range validation
           (is (thrown? IllegalArgumentException (m/cmyk->rgb [-1 0 0 0]))
               "CMYK values should be >= 0")
           (is (thrown? IllegalArgumentException (m/cmyk->rgb [0 101 0 0]))
               "CMYK values should be <= 100")

           ;; Test HSB range validation
           (let [valid-hsb [359 100 100]
                 invalid-saturation [0 101 100]
                 invalid-brightness [0 100 101]]
             (is (m/hsb->Color valid-hsb))
             (is (thrown? IllegalArgumentException (m/hsb->Color invalid-saturation))
                 "Saturation should be 0-100")
             (is (thrown? IllegalArgumentException (m/hsb->Color invalid-brightness))
                 "Brightness should be 0-100"))))