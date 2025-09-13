(defproject com.dedovic/artlib-cuda "0.0.18-SNAPSHOT"
  :description "GPU (via CUDA) accelerated utilities for making generative art"
  :monolith/inherit true
  :dependencies [[org.clojure/clojure :scope "provided"]
                 [org.clojure/core.match]

                 [com.dedovic/artlib-common]

                 [net.mikera/imagez "0.12.0"]

                 [uncomplicate/clojurecuda "0.21.0"]
                 [uncomplicate/commons "0.16.1"]
                 [org.uncomplicate/clojure-cpp "0.4.0"]]
  :profiles {:test 
             {:dependencies 
              [[org.bytedeco/cuda "12.6-9.5-1.5.11" :classifier "linux-x86_64-redist"]] }}
  :source-paths ["src/clj" "src/cuda"]
  :java-source-paths ["src/java"]
  :resource-paths ["resources"]
  :aot :all)
