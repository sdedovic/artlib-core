(ns artlib.acceleration-test
  (:require [artlib.acceleration.core :as acceleration]
            [clojure.test :refer :all])
  (:import (artlib.acceleration.cuda CudaAccelerationProvider)))

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
