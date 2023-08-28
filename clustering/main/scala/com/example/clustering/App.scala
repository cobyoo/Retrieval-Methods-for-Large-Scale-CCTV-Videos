package com.example.clustering

/** @author
  *   ${user.name}
  */
import org.apache.spark.rdd.RDD // RDD 형식 파일
import org.apache.spark._
import org.apache.spark.sql.{Row, SparkSession, SQLContext, DataFrame, Encoders}
import org.apache.spark.sql.functions.lit
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.types.{
  DataTypes,
  StructType,
  StructField,
  StringType,
  IntegerType,
  FloatType,
  MapType,
  DoubleType,
  ArrayType
}
import org.apache.spark.sql.functions.{
  col,
  from_json,
  split,
  explode,
  abs,
  exp,
  pow,
  sqrt,
  broadcast
}
import scala.util.Random
import org.apache.spark.sql.functions.{expr, max}
import org.apache.spark.sql.functions
import scala.collection.mutable.Map
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions._
import java.util.concurrent.TimeUnit.NANOSECONDS
import scala.collection.immutable._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import java.util.ArrayList
import scala.collection.mutable
import scala.util.control.Breaks._
import java.math.MathContext
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions._
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.mllib.recommendation.Rating
import scala.collection.mutable.ArrayBuffer
import java.net.InetAddress
import org.apache.spark.storage.StorageLevel
import java.text.SimpleDateFormat
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.ignite.configuration.IgniteConfiguration
import org.apache.ignite.Ignition
import org.apache.ignite.spark.{IgniteContext, IgniteRDD}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.LongType
import scala.math.Ordering
import java.net.InetSocketAddress
import org.apache.ignite.spi.discovery.tcp.TcpDiscoverySpi
import org.apache.ignite.spi.discovery.tcp.ipfinder.kubernetes.TcpDiscoveryKubernetesIpFinder
import org.apache.ignite.kubernetes.configuration.KubernetesConnectionConfiguration
import org.apache.ignite.configuration.CacheConfiguration
import org.apache.ignite.cache.query.SqlFieldsQuery
import org.apache.ignite.configuration.CacheConfiguration
import org.apache.ignite.spark.IgniteDataFrameSettings
import org.apache.ignite.spark.{IgniteContext, IgniteRDD}
import org.apache.ignite.configuration.{CacheConfiguration, IgniteConfiguration}
import com.univocity.parsers.csv.CsvParserSettings
import com.univocity.parsers.csv.CsvParser

// mvn clean compile assembly:single

case class ImageFeature(
    video_Id: Long,
    approxFeatures: Array[Double],
    origFeatures: Vector,
    no_img: Double,
    scene_graph: String
)

object App {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("VideoQueryProcessor")
      .setMaster("yarn")
      .set("spark.hadoop.validateOutputSpecs", "false") 
      .set("spark.executor.extraClassPath", "/path/to/hadoop/lib/*.jar")

    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    val ssc = new StreamingContext(spark.sparkContext, Seconds(10))

    val rawData: DataFrame =
      spark.read.parquet("/input/data.parquet")

    val defaultImgFeatureRDD = rawData
      .as[(String, Long, Array[Double], String)]
      .map { case (video_id, no_img, orig_feature, scene_graph) =>
        val approxFeatures = ax_feature(Vectors.dense(orig_feature), 5)
        val origFeatures = Vectors.dense(orig_feature)
        ImageFeature(
          video_id.toLong,
          Array(approxFeatures),
          origFeatures,
          no_img,
          scene_graph
        )
      }
      .rdd

    val igniteConfig = new IgniteConfiguration()
    igniteConfig.setCacheConfiguration(new CacheConfiguration())
    Ignition.setClientMode(true)
    val igniteContext =
      new IgniteContext(spark.sparkContext, () => igniteConfig, false)
    val igniteRDD = igniteContext.fromCache[Long, ImageFeature]("imagefeatures")

    igniteRDD.savePairs(
      defaultImgFeatureRDD.map(imgFeature => (imgFeature.video_Id, imgFeature))
    )

    val queryStream = ssc.socketTextStream("localhost", 9000)

    queryStream.foreachRDD { rdd =>
      rdd.foreach { queryString =>
        val queryFeatures: Vector = processQuery(queryString)
        val k = 5
        val windowSize = 10

        val (matchedGraphs, events) = searchFV(
          clips = igniteRDD.values.toArray,
          querySubgraph = queryFeatures,
          k = k,
          windowSize = windowSize
        )

        matchedGraphs.zip(events).foreach {
          case (graph, (eventStart, eventEnd, dist)) =>
            println(s"Matched Graph: ${graph.map(_.toString).mkString(", ")}")
            println(
              s"Event Start: $eventStart, Event End: $eventEnd, Distance: $dist"
            )
        }
      }
    }

    ssc.start()
    ssc.awaitTermination()

    igniteContext.close(true)
    spark.stop()
  }

  def ax_feature(features: Vector, numBuckets: Int): Vector = {
    val bucketSize = 1.0 / numBuckets
    val approxFeature = Vectors.dense(features.toArray.map { value =>
      val bucketIndex = math.min((value / bucketSize).toInt, numBuckets - 1)
      if (value >= 0.0 && value <= 1.0) {
        bucketIndex.toDouble
      } else {
        value
      }
    })
    approxFeature
  }

  def dtwDistance(x: Vector, y: Vector): Double = {
    val m = x.size
    val n = y.size
    val dtw = Array.fill(m + 1, n + 1)(Double.PositiveInfinity)
    dtw(0)(0) = 0

    for (i <- 1 to m) {
      for (j <- 1 to n) {
        val cost = math.abs(x(i - 1) - y(j - 1))
        dtw(i)(j) = cost + math.min(
          math.min(dtw(i - 1)(j), dtw(i)(j - 1)),
          dtw(i - 1)(j - 1)
        )
      }
    }
    dtw(m)(n)
  }

  def searchFV(
      clips: Array[ImageFeature],
      querySubgraph: Vector,
      k: Int,
      windowSize: Int
  ): (Array[Array[ImageFeature]], Array[(Long, Long, Double)]) = {

    val matchedGraphs = new ArrayBuffer[Array[ImageFeature]]()
    val distances = new ArrayBuffer[Double]()

    clips.sliding(windowSize).foreach { clipGraph =>
      val clipFeatures =
        clipGraph.flatMap(_.approxFeatures).map(ax_feature(_, 5))
      val dist = clipFeatures.zipWithIndex.map { case (features, index) =>
        dtwDistance(querySubgraph, features)
      }.sum

      if (matchedGraphs.length < k) {
        matchedGraphs.append(clipGraph.toArray)
        distances.append(dist)
      } else {
        val maxIndex = distances.indices.maxBy(distances)
        if (dist < distances(maxIndex)) {
          matchedGraphs(maxIndex) = clipGraph.toArray
          distances(maxIndex) = dist
        }
      }
    }

    val sortedGraphs = matchedGraphs.sortBy(_.head.no_img)

    val events = sortedGraphs.map { graph =>
      val dist = distances(sortedGraphs.indexOf(graph))
      val intervalDuration = 1
      val eventStart = graph.head.no_img * intervalDuration
      val eventEnd = (graph.head.no_img + windowSize - 1) * intervalDuration
      (eventStart, eventEnd, dist)
    }

    (sortedGraphs.toArray, events.toArray)
  }

  // Process the query and create a feature vector
  def processQuery(queryString: String): Vector = {
    val queryFeaturesArray = queryString.toCharArray.map(_.toDouble)
    val queryFeatures = Vectors.dense(queryFeaturesArray)
    queryFeatures
  }
}
