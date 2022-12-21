(defproject com.dedovic/artlib-core "0.0.5-SNAPSHOT"
  :description "Utilities for making generative art"
  :url "https://github.com/sdedovic/artlib-core"
  :license {:name "MIT"
            :url "https://opensource.org/licenses/MIT"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [net.mikera/vectorz-clj "0.48.0"]
                 [progrock "0.1.2"]
                 [quil "4.0.0-SNAPSHOT-1"]
                 [org.locationtech.jts/jts-core "1.18.1"]]
  :deploy-repositories [["releases"  {:sign-releases false :url "https://clojars.org"}]
                        ["snapshots" {:sign-releases false :url "https://clojars.org"}]]
  :source-paths ["src/clj"]
  :java-source-paths ["src/java"]
  :resource-paths ["resources"]
  :aot :all)
