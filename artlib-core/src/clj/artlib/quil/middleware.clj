(ns artlib.quil.middleware
  (:require [clojure.java.shell :refer [sh]]
            [quil.core :as q]
            [artlib.quil.global :refer :all]
            [progrock.core :as pr])
  (:import (java.text SimpleDateFormat)
           (java.util Date)))

(def progress (atom nil))

(defn- now
  "Gets the current system time as a file friendly prefix."
  []
  (.format (SimpleDateFormat. "yyyy-MM-dd_HH-mm-ss_") (new Date)))

(defn- wrap-setup
  "Wraps the supplied Quil setup function with animation helpers. The
  render? flag determines if we are looping or rendering a video out."
  [setup-fn framerate render?]
  (fn []
    (q/frame-rate framerate)
    (q/color-mode :hsb 360 100 100 1.0)
    
    (q/stroke-weight 1)
    (q/no-stroke)
    (q/no-fill)
    (q/background 0 0 100)

    (when render?
      (q/stroke-cap :round)
      (q/stroke-join :round))

    (assoc (setup-fn) :frame 0)))

(defn- wrap-draw
  "Wraps the supplied Quil draw function with animation helpers. The
  render? flag determines if we are looping or rendering a video out."
  [draw-fn size dirname framerate length render?]
  (fn [state]
    (if (and (> (:frame state) length) render?)
      (let [[width height] size
            size-str (str width "x" height)
            img-folder (str "output/tmp/" dirname "/")
            vid-folder (str "output/render/" dirname "/")
            vid-name (str (now) "render.mp4")
            cmd ["ffmpeg" 
                 "-r" (str framerate)
                 "-s" size-str 
                 "-i" (str img-folder "%d.png")
                 "-vcodec" "libx264" "-crf" "25" "-pix_fmt" "yuv420p"
                 (str vid-folder vid-name)]]
        (clojure.java.io/make-parents (str vid-folder vid-name))
        (apply sh cmd)
        (println "\n\nOutput video: " vid-name)
        (println "Output location: " vid-folder)
        (q/exit))
      (do
        (draw-fn state)
        (if render?
          (let [frame (:frame state)
                folder (str "output/tmp/" dirname "/")]
            (swap! progress pr/tick 1)
            (q/save (str folder frame ".png"))
            (pr/print @progress))
          (let [frame (:frame state)
                framerate (q/current-frame-rate)
                time-elapsed (/ frame framerate)]
            (with-style
              (q/fill 0 0 100)
              (q/no-stroke)
              (q/rect 0 0 100 50)
              (q/fill 0 0 0)
              (q/stroke 0 0 0)
              (q/text (format "%-7s %9d" "Frame" frame) 5 13)
              (q/text (format "%-7s %9.1f" "FPS" framerate) 5 22.5)
              (q/text (format "%-7s %9.1f" "Time" time-elapsed) 5 35))))))))

(defn- wrap-update
  "Wraps the supplied Quil fun-mode state update function with 
  animation helpers."
  [update-fn length render?]
  (fn [old-state]
    (if render?
      (update (update-fn old-state) :frame inc)
      (if (>= (:frame old-state) length)
        (assoc (update-fn old-state) :frame 0)
        (update (update-fn old-state) :frame inc)))))


(defn animation-mode [options]
  (let [size (:size options)
        animation-opts (:animation options {:framerate 30 :render? false :length 300})
        framerate (:framerate animation-opts 30)
        length (:length animation-opts 300)
        render? (:render? animation-opts false)
        dirname (:dirname animation-opts "0001")]
    (reset! progress (pr/progress-bar length))
    (-> options
        (update :draw wrap-draw size dirname framerate length render?)
        (update :setup wrap-setup framerate render?)
        (update :update wrap-update length render?))))
