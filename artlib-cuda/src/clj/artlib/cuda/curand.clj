(ns artlib.cuda.curand
  (:require [clojure.core.match :refer [match]]
            [uncomplicate.clojurecuda.internal.utils :refer [with-check]]
            [uncomplicate.commons.core :refer [Releaseable let-release]])
  (:import [org.bytedeco.javacpp DoublePointer FloatPointer IntPointer LongPointer]
           [org.bytedeco.cuda.curand curandGenerator_st]
           [org.bytedeco.cuda.global curand]
           (uncomplicate.clojurecuda.internal.impl CUDevicePtr)))

(extend-type curandGenerator_st
  Releaseable
  (release [this]
    (locking this
      (curand/curandDestroyGenerator this)
      (.deallocate this)
      (.setNull this)
      true)))

(defn- ->curandRngType
  [rng-type]
  (match rng-type
         :test curand/CURAND_RNG_TEST
         :pseudo-default curand/CURAND_RNG_PSEUDO_DEFAULT
         :pseudo-mrg32k3a curand/CURAND_RNG_PSEUDO_MRG32K3A
         :pseudo-xorwow curand/CURAND_RNG_PSEUDO_XORWOW
         :pseudo-mtgp32 curand/CURAND_RNG_PSEUDO_MTGP32
         :pseudo-mt19937 curand/CURAND_RNG_PSEUDO_MT19937
         :pseudo-philox4-32-10 curand/CURAND_RNG_PSEUDO_PHILOX4_32_10
         :quasi-default curand/CURAND_RNG_QUASI_DEFAULT
         :quasi-sobol32 curand/CURAND_RNG_QUASI_SOBOL32
         :quasi-scrambled-sobol32 curand/CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
         :quasi-sobol64 curand/CURAND_RNG_QUASI_SOBOL64
         :quasi-scrambled-sobol64 curand/CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
         :else (throw (Exception. "Unknown rng-type: " rng-type))))

(defn create-generator
  "Create new random number generator.

  Valid values for rng-type are:
    :test
    :pseudo-default
    :pseudo-mrg32k3a
    :pseudo-xorwow
    :pseudo-mtgp32
    :pseudo-mt19937
    :pseudo-philox4-32-10
    :quasi-default
    :quasi-sobol32
    :quasi-scrambled-sobol32
    :quasi-sobol64
    :quasi-scrambled-sobol64

  Default is :psuedo-default.

  [See curandCreateGenerator](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  (^curandGenerator_st [] (create-generator :pseudo-default))
  (^curandGenerator_st [rng-type]
   (let [rng-type (->curandRngType rng-type)]
     (let-release [gen (curandGenerator_st.)]
                  (with-check (curand/curandCreateGenerator gen rng-type) gen)))))

(defn set-pseudo-random-generator-seed
  "Set the seed value of the pseudo-random number generator.

  [See curandSetPseudoRandomGeneratorSeed](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  ^curandGenerator_st [^curandGenerator_st gen seed]
  (with-check
    (curand/curandSetPseudoRandomGeneratorSeed gen (long seed))
    gen))


(defn generate
  "Use gen to generate n 32-bit results into the device memory at output-ptr.

  Results are 32-bit values with every bit random.

  [See curandGenerate](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  [^curandGenerator_st gen ^IntPointer output-ptr n]
  (with-check
    (curand/curandGenerate gen output-ptr (long n))
    nil))

(defn generate-long-long
  "Use gen to generate n 64-bit results into the device memory at output-ptr.

  Results are 64-bit values with every bit random.

  [See curandGenerateLongLong](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  [^curandGenerator_st gen ^LongPointer output-ptr n]
  (with-check
    (curand/curandGenerateLongLong gen output-ptr (long n))
    nil))

(defn generate-uniform
  "Use gen to generate n float results into the device memory at output-ptr.

  Results are 32-bit floating point values between 0.0f and 1.0f,
  excluding 0.0f and including 1.0f.

  [See curandGenerateUniform](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  [^curandGenerator_st gen ^FloatPointer output-ptr n]
  (with-check
    (curand/curandGenerateUniform gen output-ptr (long n))
    nil))

(defn generate-uniform-double
  "Use gen to generate n double results into the device memory at output-ptr.

  Results are 64-bit double precision floating point values between
  0.0 and 1.0, excluding 0.0 and including 1.0.

  [See curandGenerateUniformDouble](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  [^curandGenerator_st gen ^DoublePointer output-ptr n]
  (with-check
    (curand/curandGenerateUniformDouble gen output-ptr (long n))
    nil))

(defn generate-normal
  "Use gen to generate n float results into the device memory at output-ptr.

  Results are 32-bit floating point values with mean mean and standard standard
  deviation stddev.

  [See curandGenerateNormal](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  [^curandGenerator_st gen ^FloatPointer output-ptr n mean stddev]
  (with-check
    (curand/curandGenerateNormal gen output-ptr (long n) (float mean) (float stddev))
    nil))

(defn generate-normal-double
  "Use gen to generate n double results into the device memory at output-ptr.

  Results are 64-bit floating point values with mean mean and standard standard
  deviation stddev

  [See curandGenerateNormalDouble](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  [^curandGenerator_st gen ^DoublePointer output-ptr n mean stddev]
  (with-check
    (curand/curandGenerateNormalDouble gen output-ptr (long n) (double mean) (double stddev))
    nil))

(defn generate-log-normal
  "Use gen to generate n float results into the device memory at output-ptr.

  Results are 32-bit floating point values with log-normal distribution based on
  an associated normal distribution with mean mean and standard deviation stddev.

  [See curandGenerateLogNormal](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  [^curandGenerator_st gen ^FloatPointer output-ptr n mean stddev]
  (with-check
    (curand/curandGenerateLogNormal gen output-ptr (long n) (float mean) (float stddev))
    nil))

(defn generate-log-normal-double
  "Use gen to generate n double results into the device memory at output-ptr.

  Results are 64-bit floating point values with log-normal distribution based on
  an associated normal distribution with mean mean and standard deviation stddev.

  [See curandGenerateLogNormalDouble](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST)
  "
  [^curandGenerator_st gen ^DoublePointer output-ptr n mean stddev]
  (with-check
    (curand/curandGenerateLogNormalDouble gen output-ptr (long n) (double mean) (double stddev))
    nil))

;; TODO: curandCreatePoissonDistribution
;; TODO: curandDestroyDistribution
;; TODO: curandGeneratePoisson
;; TODO: curandGenerateSeeds
;; TODO: curandGetDirectionVectors32
;; TODO: curandGetScrambleConstants32
;; TODO: curandGetDirectionVectors64
;; TODO: curandGetScrambleConstants64