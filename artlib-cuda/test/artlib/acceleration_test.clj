(ns artlib.acceleration-test
  (:require [artlib.acceleration.core :as acceleration]
            [artlib.image.core :as img]
            [mikera.image.core :as imgz]
            [clojure.test :refer :all])
  (:import [artlib.acceleration.cuda CudaAccelerationProvider]
           [java.awt.image BufferedImage]))

(deftest cuda-acceleration-provider
  (testing "provider is registered"
    (is (not (empty? (acceleration/providers))))
    (is (= CudaAccelerationProvider (type (first (acceleration/providers))))))
  
  ;; note: the following tests only work on devices with supported hardware with 
  ;;  the CUDA toolkit installed

  (testing "provider works"
    (let [provider (first (acceleration/providers))]
      (is (some? (acceleration/create provider)))
      (is (= "CUDAAccelerationProvider" (acceleration/name provider)))))

  (testing "acceleartion works"
    (let [provider (first (acceleration/providers))
          accelerator (acceleration/create provider)]
      (is (some? (acceleration/info accelerator))))))

(deftest cuda-contour-helper
  (let [provider (first (acceleration/providers))
        accelerator (acceleration/create provider)]
    (testing "basic execution flow"
      (let [rgb-32f (img/make-rgb-32f 10 10)
            gray-32f (img/make-gray-32f 10 10)]
        (is (thrown? IllegalArgumentException (acceleration/compute-contour-lines accelerator rgb-32f 0.5)))
        (is (some? (acceleration/compute-contour-lines accelerator gray-32f 0.5)))))))
