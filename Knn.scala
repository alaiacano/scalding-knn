import com.twitter.scalding._
import cascading.flow.FlowDef
import cascading.pipe.Pipe
import cascading.tuple.Fields

class KnnExampleJob(args: Args) extends Job(args) {
  val k = args.getOrElse("k", "15").toInt

  // load up the iris dataset
  val iris = Tsv("iris.tsv", ('id, 'class, 'sepalLength, 'sepalWidth, 'petalLength, 'petalWidth))
    .read
    // Just use 2 of the features so we can visualize them easier. Need to convert them to Points
    .map(('sepalLength, 'sepalWidth) -> 'features) {tup: (Double, Double) => Point(tup._1, tup._2)}
    .project('id, 'class, 'features)

  // use 2/3 of the data as a training set
  val irisTrain = iris.filter('id){id: Int => (id % 3) != 0}

  // build the test set as the inverse of the training set, and discard the 'class Field
  val irisTest = iris
    .filter('id){id: Int => (id % 3) ==0}
    .discard('class)

  // prepare the model
  val model = Knn.fit(irisTrain, 'features, 'class)
  
  // apply the model
  val predictions = Knn.classify(irisTest, model, 'features, 'id, k)(Distance.euclidean)

  // figure out how well we did
  val output = iris
    .leftJoinWithTiny('id -> 'id2, predictions.rename(('id, 'class) -> ('id2, 'classPred)))
    .discard('id2)
    .map('classPred -> 'classPred) {x: String => Option(x).getOrElse("")}
    .map('features -> ('sepalLength, 'sepalWidth)) {x: Point => (x.coord(0), x.coord(1))}
    .project('id, 'class, 'classPred, 'sepalLength, 'sepalWidth)
    .write(Tsv("iris_pred.tsv"))

}



object Knn {
  import TDsl._
  import Dsl._

  /**
   * "Trains" (more like "transforms") a model by converting the input features
   * to a Point object.
   *
   * {{{
   *   val model = Knn.fit(trainingSet, ('feature1, 'feature2), 'label)
   * }}}
   */
  def fit(trainingSet: Pipe, features: Fields, className: Fields)(implicit fd: FlowDef) : TypedPipe[(Point, String)] = {
    trainingSet
      .rename((features, className) -> ('__modelPoint, '__modelClass))
      .toTypedPipe[(Point, String)]('__modelPoint, '__modelClass)
  }

  /**
   * Uses the model to classify the input data pipe.
   *
   * @param data A pipe containing a field of `Point`s and a field of id's
   * @param model The pipe returned from the `fit` method.
   * @param featureField A field containing your features (`Point`s).
   * @param idFields A field containing a unique ID for each data point.
   * @param k Number of neighbors to use in the classification (the "k" in "kNN")
   * @return A pipe with three fields: whatever you called `idFields`, `class` and `classCount`.
   */
  def classify(data: Pipe, model: TypedPipe[(Point, String)], featureField: Fields, idFields: Fields, k: Int)(distfn : (Point,Point) => Double )(implicit fd: FlowDef) = {
    val predictions = data

      // convert these features into a Point
      .rename(featureField -> '__dataPoint)
      .project(Fields.join(idFields, '__dataPoint))
      
      // !!!!!!!!!!! DANGER !!!!!!!!!!!
      .crossWithTiny(model.toPipe('__modelPoint, '__modelClass))

      // calculate distance
      .map(('__dataPoint, '__modelPoint) -> 'distance) {tup: (Point, Point) => distfn(tup._1, tup._2)}

      // get the K nearest neighbors to each point
      .groupBy(idFields) {
        _.sortWithTake(('distance, '__modelClass) -> 'knn, k) {
          (t0 :(Double, String), t1:(Double, String)) => t0._1 < t1._1
        }
      }
      .flatten[(Double, String)]('knn -> ('distance, '__modelClass))
      .project(idFields, '__modelClass)

      // do a majority rule vote to pick the class
      .groupBy(idFields, '__modelClass) {_.size('classCount)}
      .groupBy(idFields) {
        _.sortWithTake(('classCount, '__modelClass) -> 'nn, 1) {
          (t0 :(Double, String), t1:(Double, String)) => t0._1 > t1._1
        }
      }
      .flatten[(Int, String)]('nn -> ('classCount, 'class))
      .project(idFields, 'class, 'classCount)
    predictions
  }

}


case class Point(coord: Double*) {
  override def toString = coord.mkString(",")
}

object Distance {
  /**
   * Calculates the euclidean distance between two `Point`s.
   *
   * {{{
   *   Distance.euclid(Point(0,0), Point(1,1))      // 1.414...
   *   Distance.euclid(Point(0,0,0), Point(1,1,1))  // 1.732...
   * }}}
   *
   * @param pt1 Point with an arbitrary number of dimensions.
   * @param pt2 Point with the same number of dimensions as `pt1`.
   * @return Double with the euclidean distance between the two points.
   */
  def euclidean(pt1: Point, pt2: Point) = {
    // TODO: throw an exception here.
    require(pt1.coord.size == pt2.coord.size)
    math.sqrt(pt1.coord.zip(pt2.coord).map(i=>math.pow(i._1-i._2,2)).sum)
  }

  /**
   * Calculates the manhattan distance between two `Point`s.
   *
   * {{{
   *   Distance.manhattan(Point(0,0), Point(1,1))       // 2
   *   Distance.manhattan(Point(0,0,0), Point(1,1,1))   // 3
   * }}}
   *
   * @param pt1 Point with an arbitrary number of dimensions.
   * @param pt2 Point with the same number of dimensions as `pt1`.
   * @return Double with the manhattan distance between the two points.
   */
  def manhattan(pt1: Point, pt2: Point) = {
    require(pt1.coord.size == pt2.coord.size)
    (pt1.coord.zip(pt2.coord).map(i=>math.abs(i._1-i._2))).sum
  }
}
