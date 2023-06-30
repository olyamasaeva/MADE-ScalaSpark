package org.apache.spark.ml.made

import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, Vector, Vectors, Matrices}
import org.apache.spark.ml.param.{DoubleParam, ParamMap}
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader, MLWritable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, StructField, DoubleType, StructType}
import org.apache.spark.ml.util._
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib
import org.apache.spark.sql.types._

class LinearRegression(override val uid: String) extends Regressor[Dataset[_], LinearRegression, LinearRegressionModel] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))
  
  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)


  val maxIter: DoubleParam = new DoubleParam(this, "maxIter", "Maximum number of iterations")

  def getMaxIter: Double = $(maxIter)

  def setMaxIter(value: Double): this.type = set(maxIter, value)

  setDefault(maxIter -> 100)

  override def train(dataset: Dataset[_]): LinearRegressionModel = {
    val features = dataset.drop("label")
    val label = dataset.select(col("label")).rdd.map(row => row.getAs[DenseVector](0)).collect()

    var values: Array[Array[Double]] = features.rdd.map(_.toSeq.toArray.map(_.asInstanceOf[Double])).collect()
    val array1D: Array[Double] = values.flatten
    val numRows: Int = values.length
    val numCols: Int = values.headOption.map(_.length).getOrElse(0)
    val X: DenseMatrix = new DenseMatrix(numRows, numCols, array1D)
    val Y: DenseVector = new DenseVector(label.flatMap(_.toArray))

    var weights = new DenseVector(Array.fill(numCols)(1.0)) //DenseVector.zeros[Double](numFeatures)
    var bias = 1.0
    val learningRate = 0.01
    val maxIter = getMaxIter.toInt

    for (_ <- 1 to maxIter) {

      var weightGradients = new DenseVector(Array.fill(numCols)(0.0))
      var biasGradient = 0.0

      for (i <- 0 until numRows) {
        val featureVector = new DenseVector(X.toArray.slice(i * numCols, (i + 1) * numCols))
        val prediction = featureVector.dot(weights) + bias
        val error = prediction - Y(i)
        val add = new DenseVector(featureVector.values.map(_ * error))
        weightGradients = new DenseVector((add.toArray, weightGradients.toArray).zipped.map(_ + _))
        biasGradient += error
      }
      weights = new DenseVector(weights.toArray.zip(weightGradients.toArray).map { case (x, y) => x - y * learningRate })
      bias = bias - (biasGradient * learningRate)
    }

    copyValues(new LinearRegressionModel(uid, weights, bias)).setParent(this)
  }

  private def dotProduct(weights: DenseVector, bias: Double, features: DenseMatrix): DenseVector = {
    val res = features.multiply(weights)
    new DenseVector(res.values.map(_ + bias))
  }

  override def transformSchema(schema: StructType): StructType = schema
}

class LinearRegressionModel private[made](override val uid: String, val weights: DenseVector, val bias: Double) extends RegressionModel[Dataset[_], LinearRegressionModel] with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(uid, weights, bias))

  override def predict(dataset: Dataset[_]): Double = {
    val features = dataset.drop("label")

    var values: Array[Array[Double]] = features.rdd.map(_.toSeq.toArray.map(_.asInstanceOf[Double])).collect()
    val array1D: Array[Double] = values.flatten
    val numRows: Int = values.length
    val numCols: Int = values.headOption.map(_.length).getOrElse(0)
    val X: DenseMatrix = new DenseMatrix(numRows, numCols, array1D)

    val res = dotProduct(weights, bias, X).values(0)
    res
  }

  private def dotProduct(weights: DenseVector, bias: Double, features: DenseMatrix): DenseVector = {
    val res = features.multiply(weights)
    new DenseVector(res.values.map(_ + bias))
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val data = Seq(Row(weights, bias))
      val schema = StructType(Seq(
        StructField("weights", DataTypes.createArrayType(DataTypes.DoubleType)),
        StructField("bias", DoubleType)))
      val df = sparkSession.createDataFrame(sparkSession.sparkContext.parallelize(data), schema)
      df.write.parquet(path + "/weights")
    }
  }
}
object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
    override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val weights = sqlContext.read.parquet(path + "/weights").select(col("weights")).head().getAs[DenseVector](0)
      val bias = sqlContext.read.parquet(path + "/weights").select(col("bias")).head().getAs[Double](0)
      val model = new LinearRegressionModel(weights, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}


