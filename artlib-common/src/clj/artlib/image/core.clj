(ns artlib.image.core
  (:import [java.awt.image BufferedImage DataBufferFloat WritableRaster 
                           ComponentColorModel PixelInterleavedSampleModel 
                           BandedSampleModel ColorConvertOp]
           [java.awt RenderingHints]
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
        sample-model (new PixelInterleavedSampleModel
                          DataBufferFloat/TYPE_FLOAT
                          width
                          height
                          3
                          (* width 3)
                          (into-array Integer/TYPE [0 1 2]))  ;; band offsets [R=0 G=1 B=2]
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
        sample-model (new PixelInterleavedSampleModel
                          DataBufferFloat/TYPE_FLOAT
                          width
                          height
                          4
                          (* width 4)
                          (into-array Integer/TYPE [0 1 2 3]))  ;; band offsets [R=0 G=1 B=2]
        buffer (new DataBufferFloat (* width height 4))
        raster (WritableRaster/createWritableRaster sample-model buffer nil)]
    (new BufferedImage color-model raster false nil)))

(defn convert-f32-to-int-rgb
  "Convert RGB f32 BufferedImage to RGB int BufferedImage using ColorConvertOp"
  ^BufferedImage [^BufferedImage f32-img]
  (let [width (.getWidth f32-img)
        height (.getHeight f32-img)
        int-img (BufferedImage. width height BufferedImage/TYPE_INT_RGB)

        ;; Set up high-quality rendering hints
        hints (doto (RenderingHints. RenderingHints/KEY_COLOR_RENDERING 
                                     RenderingHints/VALUE_COLOR_RENDER_QUALITY)
                (.put RenderingHints/KEY_DITHERING RenderingHints/VALUE_DITHER_DISABLE)
                (.put RenderingHints/KEY_INTERPOLATION RenderingHints/VALUE_INTERPOLATION_NEAREST_NEIGHBOR))
        color-op (ColorConvertOp. hints)]
    (.filter color-op f32-img int-img)
    int-img))


