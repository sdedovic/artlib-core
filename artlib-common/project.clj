(defproject com.dedovic/artlib-common "0.0.19-SNAPSHOT"
  :description "Utilities for making generative art"
  :monolith/inherit true

  :plugins [[com.dedovic/lein-version "1.0.0"]]
  :middleware [lein-version.plugin/middleware]

  :dependencies [[org.clojure/clojure :scope "provided"]]

  :profiles {:test
             {:dependencies
              [[net.mikera/imagez "0.12.0"]] }}

  :source-paths ["src/clj"]
  :test-paths ["test/clj"]

  :java-source-paths ["src/java"]
  :resource-paths ["resources"]
  :aot :all)
