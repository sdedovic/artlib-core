(ns artlib.cuda.particles
  (:require [com.stuartsierra.component :as component]
            [uncomplicate.clojurecuda.core :refer [program compile! module launch! grid-1d parameters function in-context]]
            [uncomplicate.commons.core :refer [Releaseable release with-release let-release]])
  (:import [uncomplicate.clojurecuda.internal.impl CUModule]))

(defn init
  "Compiles and loads the Particles module, returning it."
  ^CUModule [ctx]
  (in-context
   ctx
   (let [src (str (slurp "src/cuda/artlib/particle.cu"))]
     (with-release [prog (compile! (program src))]
       (let-release [modl (module prog)] modl)))))

;; ==================== Kernel Bindings =======================

(defn move-brownian
  "Runs the moveBrownian kernel returning the particle-buffer after execution."
  [module particle-buffer random-number-buffer count-particles magnitude]
  (with-release [kernel (function module "moveBrownian")]
    (launch!
     kernel
     (grid-1d count-particles)
     (parameters particle-buffer random-number-buffer (int count-particles) (float magnitude))))
  particle-buffer)

(defn apply-force-field
  "Runs the applyForceField kernel returning the particle-buffer after execution."
  [module particle-buffer count-particles force-constant]
  (with-release [kernel (function module "applyForceField")]
    (launch!
     kernel
     (grid-1d count-particles)
     (parameters particle-buffer (int count-particles) (float force-constant))))
  particle-buffer)

(defn move-each
  "Runs the moveEach kernel returning the particle-buffer after execution."
  [module particle-buffer move-by-buffer count-particles]
  (with-release [kernel (function module "moveEach")]
    (launch!
     kernel
     (grid-1d count-particles)
     (parameters particle-buffer move-by-buffer (int count-particles))))
  particle-buffer)

(defn move-all
  "Runs the moveAll kernel returning the particle-buffer after execution."
  [module particle-buffer x-distance y-distance count-particles]
  (with-release [kernel (function module "moveAll")]
    (launch!
     kernel
     (grid-1d count-particles)
     (parameters particle-buffer (float x-distance) (float y-distance) (int count-particles))))
  particle-buffer)