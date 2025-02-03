(defproject com.dedovic/artlib-parent "0.1.0"
  :plugins [[lein-changelog "0.3.2"]
            [lein-pprint "1.3.2"]
            [com.dedovic/lein-modules-new-profiles "0.3.14"]]

  :modules {:dirs ["artlib-core"] :subprocess nil}
  :profiles {;; profile applied to all modules
             :inherited
             {:url                  "https://github.com/sdedovic/artlib-core"
              :license              {:name "Apache License, Version 2.0"
                                     :url  "https://www.apache.org/licenses/LICENSE-2.0.html"}

              :deploy-repositories  [["releases" {:sign-releases false
                                                  :url           "https://clojars.org/repo"
                                                  :username      :env/clojars_user
                                                  :password      :env/clojars_token}]]

              ;; versions go here
              :managed-dependencies [[org.clojure/clojure "1.12.0-alpha1"]
                                     [org.clojure/buildtest "1.0.0"]
                                     [net.mikera/core.matrix "0.63.0"]
                                     [progrock "0.1.2"]
                                     [quil "4.3.1323"]

                                     ; serde
                                     [org.clojure/data.json "2.4.0"]
                                     [org.locationtech.jts/jts-core "1.18.1"]

                                     ; numerical integration
                                     [org.apache.commons/commons-math3 "3.6.1"]]}}

  :release-tasks [;; 1 - tests
                  ["vcs" "assert-committed"]
                  ["modules" "test"]

                  ;; 2 - bump versions and update changelog sections
                  ["change" "version" "leiningen.release/bump-version" "release"]
                  ["modules" "change" "version" "leiningen.release/bump-version" "release"]
                  ["changelog" "release"]

                  ;; 3 - commit changes
                  ["vcs" "commit"]
                  ["vcs" "tag" "--no-sign"]                 ;; TODO: start signing things

                  ;; 4 - deploy to clojars
                  ["modules" "deploy"]

                  ;; 5 - bump version for new dev cycle and push
                  ["change" "version" "leiningen.release/bump-version"]
                  ["modules" "change" "version" "leiningen.release/bump-version"]
                  ["vcs" "commit"]
                  ["vcs" "push"]])
