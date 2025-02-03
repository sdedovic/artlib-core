(defproject com.dedovic/artlib-cuda "0.0.18-SNAPSHOT"
  :description "GPU (via CUDA) accelerated utilities for making generative art"
  :dependencies [[org.clojure/clojure]
                 [org.clojure/core.match]

                 [uncomplicate/clojurecuda "0.10.0"]
                 [uncomplicate/commons "0.10.0"]
                 [org.jcuda/jcuda "10.1.0"]
                 [org.jcuda/jcurand "10.1.0"]]
  :source-paths ["src/clj", "src/cuda"]
  :java-source-paths ["src/java"]
  :resource-paths ["resources"]
  :aot :all)
