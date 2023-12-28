(defproject com.dedovic/artlib-core "0.0.9"
  :description "Utilities for making generative art"
  :url "https://github.com/sdedovic/artlib-core"
  :license {:name "Apache License, Version 2.0"
            :url "https://www.apache.org/licenses/LICENSE-2.0.html"}
  :dependencies [[org.clojure/clojure "1.12.0-alpha1"]
                 [net.mikera/vectorz-clj "0.48.0"]
                 [progrock "0.1.2"]
                 [quil "4.3.1323"]

                 ; serde
                 [org.clojure/data.json "2.4.0"]
                 [org.locationtech.jts/jts-core "1.18.1"]]
  :release-tasks [["vcs" "assert-committed"]
                  ["change" "version"
                   "leiningen.release/bump-version" "release"]
                  ["vcs" "commit"]
                  ["vcs" "tag" "--no-sign"] ;; TODO: start signing things
                  ["deploy"]
                  ["change" "version" "leiningen.release/bump-version"]
                  ["vcs" "commit"]
                  ["vcs" "push"]]
  :deploy-repositories [["releases"  {:sign-releases false 
                                      :url "https://clojars.org/repo"
                                      :username :env/clojars_user
                                      :password :env/clojars_token}]]
  :source-paths ["src/clj"]
  :test-paths ["test/clj"]
  :java-source-paths ["src/java"]
  :resource-paths ["resources"]
  :aot :all)
