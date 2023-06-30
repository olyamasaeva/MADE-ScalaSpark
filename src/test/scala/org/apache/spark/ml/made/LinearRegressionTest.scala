package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
//import org.apache.spark.ml.regression.LinearRegressionModel

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  def createTestDataset(spark: SparkSession): Dataset[_] = {
    import spark.implicits._
    val data = Seq(
      (1.0, new DenseVector(Array(1.0, 2.0, 3.0))),
      (2.0, new DenseVector(Array(4.0, 5.0, 6.0))),
      (3.0, new DenseVector(Array(7.0, 8.0, 9.0)))
    )
    data.toDF("label", "features")
  }

  "MyLinearRegression" should "train a linear regression model" in {
    val spark = SparkSession.builder().master("local").appName("test").getOrCreate()
    val dataset = createTestDataset(spark)

    val linearRegression = new LinearRegression()
    linearRegression.setMaxIter(10)

    val model = linearRegression.fit(dataset)

    assert(model.weights.size == 3)
    assert(model.bias != 0.0)
  }

  it should "predict values using the trained model" in {
    val spark = SparkSession.builder().master("local").appName("test").getOrCreate()
    val dataset = createTestDataset(spark)

    val linearRegression = new LinearRegression()
    linearRegression.setMaxIter(10)

    val model = linearRegression.fit(dataset)

    val predictions = model.transform(dataset)

    assert(predictions.columns.contains("prediction"))
    assert(predictions.count() == dataset.count())
  }

  it should "copy and set parameters correctly" in {
    val linearRegression = new LinearRegression()
    val copied = linearRegression.copy(ParamMap(linearRegression.maxIter -> 20))

    assert(copied.getMaxIter == 20)
  }
}

class LinearRegressionModelSpec extends AnyFlatSpec {

  "LinearRegressionModel" should "predict values using the trained model" in {
    val spark = SparkSession.builder().master("local").appName("test").getOrCreate()
    //val columns = Seq("label","feature_1", "feature_2")
    val data = Seq((0.1,0.2,0.3), (100,0.2,5.0), (3.0,0.0,1.4))
    val rdd = spark.sparkContext.parallelize(data)
    val dataset = spark.createDataFrame(rdd).toDF("label","feature_1", "feature_2")

   // val dataset = createTestDataset(spark)

    val weights = new DenseVector(Array(1.0, 2.0, 3.0))
    val bias = 0.5

    val model = new LinearRegressionModel(weights, bias)

    val prediction = model.predict(dataset)

    assert(prediction.isInstanceOf[Double])
  }

  it should "copy and set parameters correctly" in {
    val weights = new DenseVector(Array(1.0, 2.0, 3.0))
    val bias = 0.5

    val model = new LinearRegressionModel(weights, bias)
    val copied = model.copy(ParamMap.empty)

    assert(copied.weights == weights)
    assert(copied.bias == bias)
  }
}
