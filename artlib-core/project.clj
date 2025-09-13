(defproject com.dedovic/artlib-core "0.0.18"
  :description "Utilities for making generative art"
  :monolith/inherit true

  :plugins [[com.dedovic/lein-version "1.0.0"]]
  :middleware [lein-version.plugin/middleware]

  :dependencies [[org.clojure/clojure :scope "provided"]
                 [com.dedovic/artlib-common]

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
