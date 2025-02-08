(defproject com.dedovic/artlib-common "0.0.18-SNAPSHOT"
  :description "Utilities for making generative art"
  :plugins [[com.dedovic/lein-modules-new-profiles "0.3.16"]]
  :dependencies [[org.clojure/clojure :scope "provided"]]
  :source-paths ["src/clj"]
  :test-paths ["test/clj"]
  :java-source-paths ["src/java"]
  :resource-paths ["resources"]
  :aot :all)
