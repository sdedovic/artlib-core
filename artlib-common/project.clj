(defproject com.dedovic/artlib-common "0.0.19-SNAPSHOT"
  :description "Utilities for making generative art"
  :monolith/inherit true

  :plugins [[com.dedovic/lein-version "1.0.0"]]
  :middleware [lein-version.plugin/middleware]

  :dependencies [[org.clojure/clojure :scope "provided"]]

  :source-paths ["src/clj"]
  :test-paths ["test/clj"]

  :java-source-paths ["src/java"]
  :resource-paths ["resources"]
  :aot :all)
