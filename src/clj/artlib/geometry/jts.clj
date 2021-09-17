(ns artlib.geometry.jts
  (:import (org.locatio                                     ;;;;;ntech.jts.geom GeometryFactory Coordinate Polygon Geometry
             )
           (org.locationtech.jts.operation.buffer BufferParameters)))

(defn ^Polygon ->Polygon
  "Convert the supplied points to a JTS Polygon.
    The input seq must be closed, i.e. first and last points are the same.",
  [points]
  (let [shell (into-array Coordinate (map
                                       (fn [[x y]] (Coordinate. (double x) (double y)))
                                       points))
        factory (GeometryFactory.)]
    (.createPolygon factory shell)))

(defn ->points
  "Convert the supplied JTS Geometry to a point seq."
  [^Geometry geom]
  (map
    (fn [^Coordinate coord] [(.getX coord) (.getY coord)])
    (seq (.getCoordinates geom))))

(defn repair
  "Closes the points if they are not closed,
    i.e. repeat the first point at the end."
  [points]
  (if (= (last points) (first points))
    points
    (concat points [(first points)])))

(defn offset
  "Perform a polygon offsetting operation on the supplied seq of points.
    Returns an open shape if supplied, or closed shape if supplied.",
  [polygon amt]
  (let [offset-closed (fn [closed-poly]
                        (let [shell (->Polygon closed-poly)
                              new-poly (.buffer shell amt BufferParameters/CAP_FLAT)]
                          (->points new-poly)))]
    (if (= (first polygon) (last polygon))
      (offset-closed polygon)
      (let [closed-poly (concat polygon [(first polygon)])
            output (offset-closed closed-poly)]
        (drop-last output)))))
