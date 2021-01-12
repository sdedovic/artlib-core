(defproject com.dedovic/artlib-core "0.0.1-SNAPSHOT"
  :description "Utilities for making generative art"
  :url "https://github.com/sdedovic/artlib-core"
  :license {:name "MIT"
            :url "https://opensource.org/licenses/MIT"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [progrock "0.1.2"]
                 [quil "3.1.0"]]
  :source-paths ["src/clj"]
  :java-source-paths ["src/java"]
  :resource-paths ["resources"]
  :aot :all)
