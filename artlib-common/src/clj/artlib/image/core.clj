(ns artlib.image.core
  (:import [java.awt.image BufferedImage DataBufferFloat WritableRaster ComponentColorModel BandedSampleModel]
           [java.awt.color ColorSpace]))

;; https://android.googlesource.com/platform/frameworks/native/+/43aa2b1cbf7a03e248e10f4d0fec0463257cd52d/awt/java/awt/image/DataBuffer.java
(defn data-type->type [in])

(defn make-gray-32f 
  "Creates a new BufferedImage with the specified width and height."
  ^BufferedImage [width height]
  (let [color-space (ColorSpace/getInstance ColorSpace/CS_GRAY)
        color-model (new ComponentColorModel color-space
                         false ;; alpha
                         false ;; premultiplied
                         ComponentColorModel/OPAQUE
                         DataBufferFloat/TYPE_FLOAT)
        sample-model (new BandedSampleModel
                          DataBufferFloat/TYPE_FLOAT
                          width
                          height
                          1) ;; number of channels
        buffer (new DataBufferFloat (* width height))
        raster (WritableRaster/createWritableRaster sample-model buffer nil)]
    (new BufferedImage color-model raster false nil)))

(defn make-rgb-32f 
  "Creates a new BufferedImage with the specified width and height."
  ^BufferedImage [width height]
  (let [color-space (ColorSpace/getInstance ColorSpace/CS_sRGB)
        color-model (new ComponentColorModel color-space
                         false ;; alpha
                         false ;; premultiplied
                         ComponentColorModel/OPAQUE
                         DataBufferFloat/TYPE_FLOAT)
        sample-model (new BandedSampleModel
                          DataBufferFloat/TYPE_FLOAT
                          width
                          height
                          3) ;; number of channels, RGB -> 3
        buffer (new DataBufferFloat (* width height 3))
        raster (WritableRaster/createWritableRaster sample-model buffer nil)]
    (new BufferedImage color-model raster false nil)))

(defn make-rgba-32f 
  "Creates a new BufferedImage with the specified width and height."
  ^BufferedImage [width height]
  (let [color-space (ColorSpace/getInstance ColorSpace/CS_sRGB)
        color-model (new ComponentColorModel color-space
                         true ;; alpha
                         false ;; premultiplied
                         ComponentColorModel/TRANSLUCENT
                         DataBufferFloat/TYPE_FLOAT)
        sample-model (new BandedSampleModel
                          DataBufferFloat/TYPE_FLOAT
                          width
                          height
                          4) ;; number of channels, RGBA -> 4
        buffer (new DataBufferFloat (* width height 4))
        raster (WritableRaster/createWritableRaster sample-model buffer nil)]
    (new BufferedImage color-model raster false nil)))
