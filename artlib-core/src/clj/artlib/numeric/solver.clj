(ns artlib.numeric.solver
  (:import (org.apache.commons.math3.ode FirstOrderDifferentialEquations ContinuousOutputModel)
           (org.apache.commons.math3.ode.nonstiff GraggBulirschStoerIntegrator)))

(defn ->first-order
  "Constructs an object for computing a first order differential equation,
    i.e. somethihg of the form y_i'=f_i(t, y) with t0 and y(t0)=y0 known where i
    is the index of the dimension. This constructor accepts more than one function, 
    in the case of multidimentional problems. See the test cases to understand.

  Each function will recieve a real-valued t and vector of y. Each function returns
    the real value of the i-th y_i."
  [& equations]
  (reify
    FirstOrderDifferentialEquations
    (getDimension 
      [_this] 
      (count equations))
    (computeDerivatives [_this t ys ydots]
      (doseq [i (range (count equations))]
        (let [equation (nth equations i)
              value (equation t ys)]
          (aset ydots i (double value)))))))

(defn make-gragg-bulirsch-stoer
  "Helper fn to make a gragg-bulirsch-stoer integrator. Probably don't need to
    use this function directly and instad use it from one of the other, 
    higher-level functions."
  ([]
   (make-gragg-bulirsch-stoer 1e-8 100 1e-10 1e-10))
  ([min-step max-step]
   (make-gragg-bulirsch-stoer min-step max-step 1e-10 1e-10))
  ([min-step max-step abs-error rel-error]
   (new GraggBulirschStoerIntegrator 
        (double min-step)
        (double max-step)
        (double abs-error)
        (double rel-error))))

(defn solver
  "Create a function to compute the final state of an ODE. The first parameter is
    either a first order differential equation or a fn that creates it by
    application of the subsequent parameters.
 
  The returned solver is a function of the initial time, initital state, and final 
    time. It returns a vec of the final time and with the final state.
  
  State is always the same shape as the list of fns supplied when creating the 
    differential equations, i.e. the first function computes the derivative of the 
    first entry in the state structure, and so on.

  Uses the default Gragg-Bulirsch-Stoer integrator."
  [eq-or-eqfn & args]
  (let [eq (if (fn? eq-or-eqfn) (apply eq-or-eqfn args) eq-or-eqfn)
        integrator (make-gragg-bulirsch-stoer)]
    (fn [t0 init-state t]
      (let [init-state-array (into-array Double/TYPE (map double init-state))
            output-array (make-array Double/TYPE (count init-state))
            t-actual (.integrate integrator 
                                 eq 
                                 (double t0) 
                                 init-state-array 
                                 (double t) 
                                 output-array)]
        (vec (concat [t-actual] output-array))))))

(defn evolver
  [eq-or-eqfn & args]
  (let [eq (if (fn? eq-or-eqfn) (apply eq-or-eqfn args) eq-or-eqfn)
        storage (new ContinuousOutputModel)
        integrator (doto 
                     (make-gragg-bulirsch-stoer)
                     (.addStepHandler storage))]
    (fn [t0 init-state t]
      (let [init-state-array (into-array Double/TYPE (map double init-state))
            output-array (make-array Double/TYPE (count init-state))
            t-actual (.integrate integrator 
                                 eq 
                                 (double t0) 
                                 init-state-array 
                                 (double t) 
                                 output-array)]
        (fn [n]
          (let [tn (+ t0 (* n (- t t0)))]
            (.setInterpolatedTime storage (double tn))
            (vec
              (concat
                [(.getInterpolatedTime storage)]
                (.getInterpolatedState storage)))))))))


