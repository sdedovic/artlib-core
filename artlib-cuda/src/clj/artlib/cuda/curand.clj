(ns artlib.cuda.curand
  (:require [clojure.core.match :refer [match]]
            [com.stuartsierra.component :as component]
            [uncomplicate.clojurecuda.internal.utils :refer [with-check]]
            [uncomplicate.clojurecuda.internal.impl :refer [native-pointer]]
            [uncomplicate.commons.core :refer [Info info Wrappable wrap Wrapper extract Releaseable release let-release]])
  (:import [jcuda.jcurand JCurand curandRngType curandGenerator]
           [jcuda.driver CUdeviceptr]
           [uncomplicate.clojurecuda.internal.impl CULinearMemory]))

;; ==================== Release resources =======================

(deftype CURANDGenerator [ref]
  Object
  (hashCode [this]
    (hash (deref ref)))
  (equals [this other]
    (= (deref ref) (extract other)))
  (toString [this]
    (format "#CURANDGenerator[0x%s]" (Long/toHexString (native-pointer (deref ref)))))
  Wrapper
  (extract [this]
    (deref ref))
  Releaseable
  (release [this]
    (locking ref
      (when-let [d (deref ref)]
        (locking d
          (with-check (JCurand/curandDestroyGenerator d) (vreset! ref nil)))))
    true))

(extend-type curandGenerator
  Info
  (info [this]
    (info (wrap this)))
  Wrappable
  (wrap [ctx]
    (->CURANDGenerator (volatile! ctx))))

;; ====================== Psuedo-Random Number Generator ========================================

(defn- ->curandRngType
  [rng-type]
  (match rng-type
         :test                    curandRngType/CURAND_RNG_TEST
         :pseudo-default          curandRngType/CURAND_RNG_PSEUDO_DEFAULT
         :pseudo-mrg32k3a         curandRngType/CURAND_RNG_PSEUDO_MRG32K3A
         :pseudo-xorwow           curandRngType/CURAND_RNG_PSEUDO_XORWOW
         :pseudo-mtgp32           curandRngType/CURAND_RNG_PSEUDO_MTGP32
         :pseudo-mt19937          curandRngType/CURAND_RNG_PSEUDO_MT19937
         :pseudo-philox4_32_10    curandRngType/CURAND_RNG_PSEUDO_PHILOX4_32_10
         :quasi-default           curandRngType/CURAND_RNG_QUASI_DEFAULT
         :quasi-sobol32           curandRngType/CURAND_RNG_QUASI_SOBOL32
         :quasi-scrambled_sobol32 curandRngType/CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
         :quasi-sobol64           curandRngType/CURAND_RNG_QUASI_SOBOL64
         :quasi-scrambled_sobol64 curandRngType/CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
         :else (throw "Unknown curandRngType/CURAND Generator Type!")))

(defn create-generator
  "Initializes the cuRAND driver. This will create a pseudo-random number generator."
  (^CURANDGenerator [] (create-generator :pseudo-default))
  (^CURANDGenerator [rng-type]
  (let [rng-type (->curandRngType rng-type)]
    (let-release [gen (new curandGenerator)]
      (with-check (JCurand/curandCreateGenerator gen rng-type) (wrap gen))))))

(defn seed-generator
  "Sets the seed for the supplied cuRAND pseudo-random number generator. This must
  be called prior to using the generator."
  ^curandGenerator [^CURANDGenerator gen seed]
  (with-check
   (JCurand/curandSetPseudoRandomGeneratorSeed (extract gen) (long seed))
   gen))

;; ====================== Psuedo-Random Number Generation ========================================

(defn generate-uniform
  "Generate n floats on the device."
  ^CUdeviceptr [^CURANDGenerator gen ^CULinearMemory output-ptr n]
  (with-check
   (JCurand/curandGenerateUniform (extract gen) (extract output-ptr) (long n))
   output-ptr))

(defn generate-normal
  "Generate n floats on the device."
  ^CULinearMemory [^CURANDGenerator gen ^CULinearMemory output-ptr n mean std-dev]
  (with-check
   (JCurand/curandGenerateNormal (extract gen) (extract output-ptr) (long n) (float mean) (float std-dev))
   output-ptr))
