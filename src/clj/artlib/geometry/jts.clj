(ns artlib.geometry.jts
  (:import (org.locationtech.jts.geom GeometryFactory Coordinate Polygon Geometry)
           (org.locationtech.jts.algorithm RobustLineIntersector)
           (org.locationtech.jts.operation.buffer BufferParameters)))

(defn ^Coordinate ->Coordinate
  "Converts the paremeters to a Coordinate. Calling with a single argument
    treats the value as a vector. Calling with two or more treats the
    arguments as elements."
  ([elements]
   (apply ->Coordinate elements))
  ([x y]
   (Coordinate. (double x) (double y)))
  ([x y z]
   (Coordinate. (double x) (double y) (double z))))

(defn Coordinate->point
  "Convert the Coordinate to a vec of its elements/"
  [^Coordinate coord]
  ;; .equals for ##NaN becuase Clojure will never =, == ##NaN
  (if (.equals (.getZ coord) Coordinate/NULL_ORDINATE)
    [(.getX coord) (.getY coord)]
    [(.getX coord) (.getY coord) (.getZ coord)]))

(defn- repair
  "Closes the points if they are not closed,
    i.e. repeat the first point at the end."
  [points]
  (if (= (last points) (first points))
    points
    (concat points [(first points)])))

(defn ^Polygon ->Polygon
  "Convert the supplied points to a Polygon, potentially closing
    the original points. A 'closed' Polygon has the same first and
    last point."
  [points]
  (let [coords (->> points
                    repair
                    (map ->Coordinate)
                    into-array)
        factory (GeometryFactory.)]
    (.createPolygon factory coords)))

(defn Geometry->points
  "Convert the supplied JTS Geometry to a point seq. The optional argument 'as'
    supports values of :open or :closed and will drop or keep the last point,
    respectively. The default is :closed. This is due to JTS Polygons always
    being closed."
  ([^Geometry geom]
   (Geometry->points geom :closed))
  ([^Geometry geom as]
   (let [points (map Coordinate->point (.getCoordinates geom))]
     (if (= as :open)
       (drop-last points)
       points))))


;; TODO(2024-1-21): move methods below this comment to a different 
;;  namespace as they are implmentation-agnostic and general purpose.

(defn buffer-poly
  "Perform a polygon offsetting operation on the supplied seq of points.
    Returns an open shape if supplied, or closed shape if supplied."
  [polygon amt]
  (let [buffer-fn (fn [closed-poly]
                        (let [shell (->Polygon closed-poly)
                              new-poly (.buffer shell amt BufferParameters/CAP_FLAT)]
                          (Geometry->points new-poly)))]
    (if (= (first polygon) (last polygon))
      (buffer-fn polygon)
      (let [closed-poly (concat polygon [(first polygon)])
            output (buffer-fn closed-poly)]
        (drop-last output)))))

(defn line-segments-intersect?
  "Returns true if the two line segments intersect. The lines 
    are not projected past their range."
  ([a b]
   (line-segments-intersect? (first a) (last a) (first b) (last b)))
  ([p1 p2 p3 p4]
   (let [intersector (RobustLineIntersector.)]
     (.computeIntersection intersector
                           (->Coordinate p1)
                           (->Coordinate p2)
                           (->Coordinate p3)
                           (->Coordinate p4))
     (.isProper intersector))))

(defn line-segment-intersection
  "Computes the proper intersection of the two line segments. The lines are 
    not projected past their range. Returns nil if they do not intersect or
    if they are colinear."
  ([a b]
   (line-segment-intersection (first a) (last a) (first b) (last b)))
  ([p1 p2 p3 p4]
   (let [intersector (RobustLineIntersector.)]
     (.computeIntersection intersector
                           (->Coordinate p1)
                           (->Coordinate p2)
                           (->Coordinate p3)
                           (->Coordinate p4))
     (if (.isProper intersector)
       (Coordinate->point (.getIntersection intersector 0))
       nil))))

(defn cut-segment-by-segment
  "Cuts the first line segment using the second line segment, if they intersect.
    Returns a list of line segments. When the line segments do not intersect, 
    returns a list of the first line segment."
  ([a b]
   (cut-segment-by-segment (first a) (last a) (first b) (last b)))
  ([p1 p2 p3 p4]
   (if-let [point (line-segment-intersection p1 p2 p3 p4)]
     [[p1 point] [point p2]]
     [[p1 p2]])))

(defn cut-segment-by-collection
  "Same as cut-segment-by-segment, but the second argument is a collection of line
    segments, to be evaluated recursively against the seg."
  [seg col]
  (if (and (seq col) (seq seg))
    (let [cut-by (first col)
          new-segments (cut-segment-by-segment seg cut-by)
          cut-next (rest col)]
      (mapcat #(cut-segment-by-collection % cut-next) new-segments))
    [seg]))
