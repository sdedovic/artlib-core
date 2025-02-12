(defproject com.dedovic/artlib-core "0.0.18-SNAPSHOT"
  :description "Utilities for making generative art"
  :plugins [[com.dedovic/lein-modules-new-profiles "0.3.14"]]
  :dependencies [[org.clojure/clojure]
                 [net.mikera/core.matrix]
                 [progrock]
                 [quil]

                 ; serde
                 [org.clojure/data.json]
                 [org.locationtech.jts/jts-core]

                 ; numerical integration
                 [org.apache.commons/commons-math3]]
  :source-paths ["src/clj"]
  :test-paths ["test/clj"]
  :java-source-paths ["src/java"]
  :resource-paths ["resources"]
  :aot :all)
