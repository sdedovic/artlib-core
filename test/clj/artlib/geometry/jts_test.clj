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

(deftest buffer-poly
  (let [round-fn #(/ (Math/round (* % 1e5)) 1e5)
        poly [[-1.0 -1.0] [-1.0 1.0] [1.0 1.0] [1.0 -1.0]]
        buffered-in (jts/buffer-poly poly -0.1)
        ;; TODO: test buffering by positive numbers as idk what it does
        ;;buffered-out (map #(map round-fn %) (jts/buffer-poly poly 0.1))
        ]
    (is (= buffered-in [[-0.9 -0.9] [-0.9 0.9] [0.9 0.9] [0.9 -0.9]]))))

(deftest line-segments-intersect?
  (let [p1 [-1 0] p2 [1 0]
        p3 [0 -1] p4 [0 1]]
    (is (jts/line-segments-intersect? p1 p2 p3 p4))
    (is (jts/line-segments-intersect? [p1 p2] [p3 p4]))
    
    (is (not (jts/line-segments-intersect? p1 p2 p1 p2)))
    (is (not (jts/line-segments-intersect? p1 p2 p2 p1)))

    (is (not (jts/line-segments-intersect? p1 p2 [-1 1] [1 1])))))

(deftest line-segment-intersection
  (let [p1 [-1 0] p2 [1 0]
        p3 [0 -1] p4 [0 1]]
    (is (= (jts/line-segment-intersection p1 p2 p3 p4) [0.0 0.0]))
    (is (= (jts/line-segment-intersection [p1 p2] [p3 p4]) [0.0 0.0]))
    
    (is (nil? (jts/line-segment-intersection p1 p2 p1 p2)))
    (is (nil? (jts/line-segment-intersection p1 p2 p2 p1)))

    (is (nil? (jts/line-segment-intersection p1 p2 [-1 1] [1 1])))

    (is (= (jts/line-segment-intersection [0 7] [6 25] [0 9] [8 25]) [2.0 13.0]))))

(deftest cut-segment-by-segment
  (let [a (jts/cut-segment-by-segment [0 0] [1 0] [5 0] [-5 0])
        b (jts/cut-segment-by-segment [0 7] [6 25] [0 9] [8 25])]
    (is (= a [[[0 0] [1 0]]]))
    (is (= b [[[0 7] [2.0 13.0]] [[2.0 13.0] [6 25]]]))))
