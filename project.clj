(defproject com.dedovic/artlib-parent "0.0.18-SNAPSHOT"
  :plugins [[lein-changelog "0.3.2"]
            [lein-pprint "1.3.2"]
            [lein-monolith "1.10.3"]
            [com.dedovic/lein-version "1.0.0"]]

  :middleware [lein-version.plugin/middleware]

  :monolith {:inherit 
             [:url :license :deploy-repositories :release-tasks]

             :inherit-leaky 
             [:repositories :license :managed-dependencies]
             
             :project-dirs 
             ["artlib-core" "artlib-cuda" "artlib-common"]}

  :url "https://github.com/sdedovic/artlib-core"
  :license {:name "Apache License, Version 2.0" 
            :url  "https://www.apache.org/licenses/LICENSE-2.0.html"}

  :deploy-repositories  [["releases" {:sign-releases false
                                      :url           "https://clojars.org/repo"
                                      :username      :env/clojars_user
                                      :password      :env/clojars_token}]]

  :managed-dependencies [[org.clojure/clojure "1.12.0"]
                         [org.clojure/core.match "1.0.0"]
                         [net.mikera/core.matrix "0.63.0"]

                         ;; this
                         [com.dedovic/artlib-core :version]
                         [com.dedovic/artlib-common :version]
                         [com.dedovic/artlib-cuda :version]

                         ; progress bar
                         [progrock "0.1.2"]

                         ; graphics
                         [quil "4.3.1323"]

                         ; serde
                         [org.clojure/data.json "2.4.0"]
                         [org.locationtech.jts/jts-core "1.18.1"]

                         ; numerical integration
                         [org.apache.commons/commons-math3 "3.6.1"]]

  :release-tasks [;; 1 - tests
                  ["vcs" "assert-committed"]
                  ["monolith" "each" "test"]

                  ;; 2 - bump versions and update changelog sections
                  ["change" "version" "leiningen.release/bump-version" "release"]
                  ["monolith" "each" "change" "version" "leiningen.release/bump-version" "release"]
                  ["changelog" "release"]

                  ;; 3 - commit changes
                  ["vcs" "commit"]
                  ["vcs" "tag" "--no-sign"]                 ;; TODO: start signing things

                  ;; 4 - deploy to clojars
                  ["monolith" "each" "deploy"]

                  ;; 5 - bump version for new dev cycle and push
                  ["change" "version" "leiningen.release/bump-version"]
                  ["monolith" "each" "change" "version" "leiningen.release/bump-version"]
                  ["vcs" "commit"]
                  ["vcs" "push"]]

  :dependencies [[org.clojure/clojure :scope "provided"]])
