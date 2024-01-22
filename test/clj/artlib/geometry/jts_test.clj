(ns artlib.geometry.jts-test
  (:require [artlib.geometry.jts :as jts]
            [clojure.test :refer [is deftest]]))

(deftest ->Coordinate
  (let [a (jts/->Coordinate [1.2 3])
        b (jts/->Coordinate [1.2 3 4.5])
        c (jts/->Coordinate 1.2 3)
        d (jts/->Coordinate 1.2 3 4.5)]
    (is (= (.getX a) 1.2))
    (is (= (.getY a) 3.0))

    (is (= (.getX b) 1.2))
    (is (= (.getY b) 3.0))
    (is (= (.getZ b) 4.5))

    (is (= (.getX c) 1.2))
    (is (= (.getY c) 3.0))

    (is (= (.getX d) 1.2))
    (is (= (.getY d) 3.0))
    (is (= (.getZ d) 4.5))))

(deftest Coordinate->point
  (let [a (jts/->Coordinate [1.2 3])
        b (jts/->Coordinate [1.2 3 4.5])]
    (is (= (jts/Coordinate->point a) [1.2 3.0]))
    (is (= (jts/Coordinate->point b) [1.2 3.0 4.5]))))

(deftest ->polygon
  (let [points-a [[0 0] [1 0] [2 0] [2 5] [1 5] [0 5]]
        polygon-a (jts/->Polygon points-a)
        coords-a (.getCoordinates polygon-a)

        points-b [[0 0] [1 0] [2 0] [2 5] [1 5] [0 5] [0 0]]
        polygon-b (jts/->Polygon points-b)
        coords-b (.getCoordinates polygon-b)]

    ;; Polygon is closed, i.e. last point is first point
    ;;  so number of points is +1 in this case
    (is (= (.getNumPoints polygon-a) 7))
    
    (is (= (jts/Coordinate->point (aget coords-a 0)) [0.0 0.0]))
    (is (= (jts/Coordinate->point (aget coords-a 1)) [1.0 0.0]))
    (is (= (jts/Coordinate->point (aget coords-a 2)) [2.0 0.0]))
    (is (= (jts/Coordinate->point (aget coords-a 3)) [2.0 5.0]))
    (is (= (jts/Coordinate->point (aget coords-a 4)) [1.0 5.0]))
    (is (= (jts/Coordinate->point (aget coords-a 5)) [0.0 5.0]))
    (is (= (jts/Coordinate->point (aget coords-a 6)) [0.0 0.0]))
    
    (is (= (.getNumPoints polygon-b) 7))
    
    (is (= (jts/Coordinate->point (aget coords-b 0)) [0.0 0.0]))
    (is (= (jts/Coordinate->point (aget coords-b 1)) [1.0 0.0]))
    (is (= (jts/Coordinate->point (aget coords-b 2)) [2.0 0.0]))
    (is (= (jts/Coordinate->point (aget coords-b 3)) [2.0 5.0]))
    (is (= (jts/Coordinate->point (aget coords-b 4)) [1.0 5.0]))
    (is (= (jts/Coordinate->point (aget coords-b 5)) [0.0 5.0]))
    (is (= (jts/Coordinate->point (aget coords-b 6)) [0.0 0.0]))))

(deftest Geometry->points
  (let [points [[0 0] [1 0] [2 0] [2 5] [1 5] [0 5]]
        polygon (jts/->Polygon points)
        points (jts/Geometry->points polygon)
        points-open (jts/Geometry->points polygon :open)]

    ;; make sure default is :closed
    (is (= points (jts/Geometry->points polygon :closed)))

    (is (= (count points) 7))
    (is (= (nth points 0) [0.0 0.0]))
    (is (= (nth points 1) [1.0 0.0]))
    (is (= (nth points 2) [2.0 0.0]))
    (is (= (nth points 3) [2.0 5.0]))
    (is (= (nth points 4) [1.0 5.0]))
    (is (= (nth points 5) [0.0 5.0]))
    (is (= (nth points 6) [0.0 0.0]))
    
    (is (= (count points-open) 6))
    (is (= (nth points-open 0) [0.0 0.0]))
    (is (= (nth points-open 1) [1.0 0.0]))
    (is (= (nth points-open 2) [2.0 0.0]))
    (is (= (nth points-open 3) [2.0 5.0]))
    (is (= (nth points-open 4) [1.0 5.0]))
    (is (= (nth points-open 5) [0.0 5.0]))))
