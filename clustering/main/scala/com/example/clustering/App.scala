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

// object App {
//   case class ImageFeature(
//       video_Id: Long,
//       approxFeatures: Array[Vector],
//       origFeatures: Vector,
//       no_img: Long
//   )

//   def ax_feature(features: Vector, numBuckets: Int): Vector = {
//     val bucketSize = 1.0 / numBuckets
//     val approxFeature = Vectors.dense(features.toArray.map { value =>
//       val bucketIndex = math.min((value / bucketSize).toInt, numBuckets - 1)
//       if (value >= 0.0 && value <= 1.0) {
//         bucketIndex.toDouble
//       } else {
//         value
//       }
//     })
//     approxFeature
//   }

//   def compareFeatures(imageFeatures: Vector, queryFeatures: Vector): Double = {
//     val gamma = 0.5
//     val sum = imageFeatures.toArray
//       .zip(queryFeatures.toArray)
//       .map { case (p1, p2) =>
//         val d = p2 - p1
//         math.pow(d, 2)
//       }
//       .sum
//     math.exp(-gamma / sum)
//   }

//   def dtwDistance(x: Vector, y: Vector): Double = {
//     val m = x.size
//     val n = y.size
//     val dtw = Array.fill(m + 1, n + 1)(Double.PositiveInfinity)

//     dtw(0)(0) = 0

//     for (i <- 1 to m) {
//       for (j <- 1 to n) {
//         val cost = math.abs(x(i - 1) - y(j - 1))
//         dtw(i)(j) = cost + math.min(
//           math.min(dtw(i - 1)(j), dtw(i)(j - 1)),
//           dtw(i - 1)(j - 1)
//         )
//       }
//     }
//     dtw(m)(n)
//   }

//   def searchFV(
//       clips: Array[ImageFeature],
//       querySubgraph: Vector,
//       k: Int
//   ): Array[Array[ImageFeature]] = {
//     val matchedGraphs = new ArrayBuffer[Array[ImageFeature]]()
//     var minDist = Double.PositiveInfinity

//     clips.sliding(k).foreach { clipGraph =>
//       val clipFeatures =
//         clipGraph.flatMap(_.approxFeatures).map(ax_feature(_, 5))
//       val dist = clipFeatures.zipWithIndex.map { case (features, index) =>
//         dtwDistance(querySubgraph, features)
//       }.sum

//       if (dist < minDist) {
//         minDist = dist
//         matchedGraphs.clear()
//         val matchedClipGraph = clipGraph.toArray
//         matchedGraphs.append(matchedClipGraph)
//       } else if (dist == minDist) {
//         val matchedClipGraph = clipGraph.toArray
//         matchedGraphs.append(matchedClipGraph)
//       }
//     }

//     // Apply DTW to matchedGraphs
//     val dtwMatchedGraphs = matchedGraphs.map { graph =>
//       val dtwGraph = graph.map { imageFeature =>
//         val dtwFeatures = imageFeature.approxFeatures.map(ax_feature(_, 5))
//         imageFeature.copy(approxFeatures = dtwFeatures)
//       }
//       dtwGraph
//     }

//     // Sort the image graph sequences
//     val sortedGraphs = dtwMatchedGraphs.sortBy(_.head.no_img)

//     sortedGraphs.toArray
//   }

//   def main(args: Array[String]): Unit = {
//     val spark = SparkSession
//       .builder()
//       .appName("SimilaritySearch")
//       .getOrCreate()

//     val rawData: DataFrame =
//       spark.read.parquet("file:///home/dblab/ysh/zata_file/data.parquet")

//     // rawData.show()
// //     root
// //  |-- video_id: string (nullable = true)
// //  |-- no_img: long (nullable = true)
// //  |-- orig_feature: array (nullable = true)
// //  |    |-- element: double (containsNull = true)

//     import spark.implicits._

//     val defaultImgFeatureRDD = rawData
//       .as[(String, Long, Array[Double])]
//       .map { case (video_id, no_img, orig_feature) =>
//         val approxFeatures = ax_feature(Vectors.dense(orig_feature), 5)
//         val origFeatures = Vectors.dense(orig_feature)
//         val imageFeature = ImageFeature(
//           video_id.toLong,
//           Array(approxFeatures),
//           origFeatures,
//           no_img
//         )
//         imageFeature
//       }
//       .rdd

//     defaultImgFeatureRDD.cache()

//     // val queryFeatures =
//     //   Vectors.dense(0.24528900, 0.24145714, 0.69088054, 0.32330546, 0.19808596,
//     //     0.84528900, 0.94145714, 0.89088054, 0.32330546, 0.19808596, 0.84528900,
//     //     0.24145714, 0.09088054, 0.92330546, 0.19808596, 0.84528900, 0.24145714,
//     //     0.69088054, 0.02330546, 0.49808596, 0.84528900, 0.24145714, 0.69088054,
//     //     0.32330546, 0.59808596, 0.64528900, 0.24145714, 0.69088054, 0.32330546,
//     //     0.19808596, 0.0)
//     import scala.util.Random

//     val random = new Random()

//     val queryFeatures = Vectors.dense(Array.fill(30)(math.random()))

//     val k = 3 // 근접 이웃 개수

//     var start = System.nanoTime() // 시작 시간 측정
//     val similarities = defaultImgFeatureRDD
//       .flatMap { imageFeature =>
//         val videoId = imageFeature.video_Id
//         val origFeatures = imageFeature.origFeatures
//         val approxFeatures = imageFeature.approxFeatures.map(ax_feature(_, 5))
//         val similarities = approxFeatures.map(approxFeature =>
//           compareFeatures(approxFeature, queryFeatures)
//         )
//         val rows = approxFeatures.zip(similarities).map {
//           case (approxFeature, similarity) =>
//             (
//               videoId,
//               similarity,
//               origFeatures,
//               imageFeature.no_img,
//               approxFeature
//             )
//         }
//         rows
//       }
//       .toDF("video_Id", "similarity", "origFeatures", "no_img", "approxFeature")

//     var end = System.nanoTime() // 시작 시간 측정

// // 유사도 기준 상위 k개의 결과 선택
//     val topKImages = similarities.orderBy($"similarity".desc).take(k)

//     val topKImagesRDD = spark.sparkContext.parallelize(topKImages)

//     val schema = StructType(
//       Seq(
//         StructField("video_Id", LongType, nullable = false),
//         StructField("similarity", DoubleType, nullable = false)
//       )
//     )
//     val topKImagesDF = spark.createDataFrame(topKImagesRDD, schema)

//     val defaultImgFeatureDF = defaultImgFeatureRDD
//       .flatMap { imageFeature =>
//         val videoId = imageFeature.video_Id
//         val noImg = imageFeature.no_img
//         val origFeatures = imageFeature.origFeatures
//         val numBuckets = 5
//         val approxFeatures =
//           imageFeature.approxFeatures.map(ax_feature(_, numBuckets))
//         val queryFeature = queryFeatures
//         val similarities = approxFeatures.map(approxFeature =>
//           compareFeatures(approxFeature, queryFeature)
//         )
//         val rows = approxFeatures.zip(similarities).map {
//           case (approxFeature, similarity) =>
//             (videoId, similarity, origFeatures, noImg, approxFeature)
//         }
//         rows
//       }
//       .toDF("video_Id", "similarity", "origFeatures", "no_img", "approxFeature")

//     val vectorToArray = udf((vector: Vector) => vector.toArray)

//     val resultDF = similarities
//       .orderBy(col("similarity").desc, col("no_img"))
//       .select(
//         col("video_Id"),
//         col("similarity"),
//         col("origFeatures"),
//         col("no_img"),
//         col("approxFeature")
//       )

//     val sortedResultDF = resultDF
//       .orderBy(
//         col("similarity").desc,
//         col("origFeatures"),
//         col("no_img")
//       )

//     println(
//       "=========================================================================================="
//     )
//     println(
//       "                                    Similarity Result                                     "
//     )
//     println(
//       "=========================================================================================="
//     )
//     println("\n\n")
//     sortedResultDF.show(k)

//     val clipsArray = defaultImgFeatureRDD.collect()
//     val videoGraphs = searchFV(clipsArray, queryFeatures, k)

// // Print similar video graphs
//     videoGraphs.foreach { graph =>
//       println(
//         "=========================================================================================="
//       )
//       println(
//         "                             Video Clips Similarity Result                                "
//       )
//       println(
//         "=========================================================================================="
//       )

//       val dtwDistances = graph.map { imageFeature =>
//         val dtwDist = dtwDistance(queryFeatures, imageFeature.approxFeatures(0))
//         (imageFeature, dtwDist)
//       }

//       val sortedGraph = dtwDistances.sortBy(_._2)

//       sortedGraph.foreach { case (imageFeature, dtwDist) =>
//         println(s"Video ID: ${imageFeature.video_Id}, DTW Distance: $dtwDist")
//       }

//       println("\n\n")
//     }

//     println(
//       "=========================================================================================="
//     )
//     println(
//       "                                        Time Taken                                        "
//     )
//     println(
//       "=========================================================================================="
//     )
//     println(
//       s"Time Taken: ${java.util.concurrent.TimeUnit.NANOSECONDS.toNanos(end - start)}ns"
//     )
//     println("\n\n")

//     defaultImgFeatureRDD.unpersist()

//     spark.stop()
//   }

// }

// object App {
//   case class ImageFeature(
//       video_Id: Long,
//       approxFeatures: Array[Vector],
//       origFeatures: Vector,
//       no_img: Long
//   )

//   def ax_feature(features: Vector, numBuckets: Int): Vector = {
//     val bucketSize = 1.0 / numBuckets
//     val approxFeature = Vectors.dense(features.toArray.map { value =>
//       val bucketIndex = math.min((value / bucketSize).toInt, numBuckets - 1)
//       if (value >= 0.0 && value <= 1.0) {
//         bucketIndex.toDouble
//       } else {
//         value
//       }
//     })
//     approxFeature
//   }

//   def compareFeatures(imageFeatures: Vector, queryFeatures: Vector): Double = {
//     val gamma = 0.5
//     val sum = imageFeatures.toArray
//       .zip(queryFeatures.toArray)
//       .map { case (p1, p2) =>
//         val d = p2 - p1
//         math.pow(d, 2)
//       }
//       .sum
//     math.exp(-gamma / sum)
//   }

//   def dtwDistance(x: Vector, y: Vector): Double = {
//     val m = x.size
//     val n = y.size
//     val dtw = Array.fill(m + 1, n + 1)(Double.PositiveInfinity)

//     dtw(0)(0) = 0

//     for (i <- 1 to m) {
//       for (j <- 1 to n) {
//         val cost = math.abs(x(i - 1) - y(j - 1))
//         dtw(i)(j) = cost + math.min(
//           math.min(dtw(i - 1)(j), dtw(i)(j - 1)),
//           dtw(i - 1)(j - 1)
//         )
//       }
//     }
//     dtw(m)(n)
//   }

//   def searchFV(
//       clips: Array[ImageFeature],
//       querySubgraph: Vector,
//       k: Int,
//       windowSize: Int
//   ): (Array[Array[ImageFeature]], Option[(Long, Long, Double)]) = {
//     val matchedGraphs = new ArrayBuffer[Array[ImageFeature]]()
//     var minDist = Double.PositiveInfinity
//     var event: Option[(Long, Long, Double)] = None

//     clips.sliding(k).foreach { clipGraph =>
//       val clipFeatures =
//         clipGraph.flatMap(_.approxFeatures).map(ax_feature(_, 5))
//       val dist = clipFeatures.zipWithIndex.map { case (features, index) =>
//         dtwDistance(querySubgraph, features)
//       }.sum

//       if (dist < minDist) {
//         minDist = dist
//         matchedGraphs.clear()
//         val matchedClipGraph = clipGraph.toArray
//         matchedGraphs.append(matchedClipGraph)
//         // 이벤트 정보 추출 (시작, 끝, 거리)
//         val intervalDuration = 1 // 각 비디오 클립의 실제 지속 시간(초)으로 대체해주세요.

//         val eventStart = matchedClipGraph.head.no_img * intervalDuration
//         val eventEnd =
//           (matchedClipGraph.head.no_img + windowSize - 1) * intervalDuration
//         event = Some((eventStart, eventEnd, dist))
//       } else if (dist == minDist) {
//         val matchedClipGraph = clipGraph.toArray
//         matchedGraphs.append(matchedClipGraph)
//       }
//     }
//     val deduplicatedGraphs = matchedGraphs.distinct

//     // matchedGraphs에 DTW 적용하기
//     val dtwMatchedGraphs = deduplicatedGraphs.map { graph =>
//       val dtwGraph = graph.map { imageFeature =>
//         val dtwFeatures = imageFeature.approxFeatures.map(ax_feature(_, 5))
//         imageFeature.copy(approxFeatures = dtwFeatures)
//       }
//       dtwGraph
//     }

//     val sortedGraphs = dtwMatchedGraphs.sortBy(_.head.no_img)

//     (sortedGraphs.toArray, event)
//   }
//   def main(args: Array[String]): Unit = {
//     val spark = SparkSession
//       .builder()
//       .master("yarn")
//       .appName("SimilaritySearch")
//       .getOrCreate()

//     val rawData: DataFrame =
//       spark.read.parquet("/input/data.parquet")

//     // rawData.show()
//     //     root
//     //  |-- video_id: string (nullable = true)
//     //  |-- no_img: long (nullable = true)
//     //  |-- orig_feature: array (nullable = true)
//     //  |    |-- element: double (containsNull = true)

//     import spark.implicits._

//     val random = new Random()
//     val defaultImgFeatureRDD = rawData
//       .as[(String, Long, Array[Double])]
//       .map { case (video_id, no_img, orig_feature) =>
//         val approxFeatures = ax_feature(Vectors.dense(orig_feature), 5)
//         val origFeatures = Vectors.dense(orig_feature)
//         val imageFeature = ImageFeature(
//           video_id.toLong,
//           Array(approxFeatures),
//           origFeatures,
//           no_img
//         )
//         imageFeature
//       }
//       .rdd

//     defaultImgFeatureRDD.cache()
//     val queryFeatures = Vectors.dense(Array.fill(64)(math.random()))

//     val k = 3 // 근접 이웃 개수

//     var start = System.nanoTime() // 시작 시간 측정
//     val similarities = defaultImgFeatureRDD
//       .flatMap { imageFeature =>
//         val videoId = imageFeature.video_Id
//         val origFeatures = imageFeature.origFeatures
//         val approxFeatures = imageFeature.approxFeatures.map(ax_feature(_, 5))
//         val similarities = approxFeatures.map(approxFeature =>
//           compareFeatures(approxFeature, queryFeatures)
//         )
//         val rows = approxFeatures.zip(similarities).map {
//           case (approxFeature, similarity) =>
//             (
//               videoId,
//               similarity,
//               origFeatures,
//               imageFeature.no_img,
//               approxFeature
//             )
//         }
//         rows
//       }
//       .toDF("video_Id", "similarity", "origFeatures", "no_img", "approxFeature")

//     var end = System.nanoTime() // 시작 시간 측정

//     // 유사도 기준 상위 k개의 결과 선택
//     val topKImages = similarities.orderBy($"similarity".desc).take(k)

//     val topKImagesRDD = spark.sparkContext.parallelize(topKImages)

//     val schema = StructType(
//       Seq(
//         StructField("video_Id", LongType, nullable = false),
//         StructField("similarity", DoubleType, nullable = false)
//       )
//     )
//     val topKImagesDF = spark.createDataFrame(topKImagesRDD, schema)

//     val defaultImgFeatureDF = defaultImgFeatureRDD
//       .flatMap { imageFeature =>
//         val videoId = imageFeature.video_Id
//         val noImg = imageFeature.no_img
//         val origFeatures = imageFeature.origFeatures
//         val numBuckets = 5
//         val approxFeatures =
//           imageFeature.approxFeatures.map(ax_feature(_, numBuckets))
//         val queryFeature = queryFeatures
//         val similarities = approxFeatures.map(approxFeature =>
//           compareFeatures(approxFeature, queryFeature)
//         )
//         val rows = approxFeatures.zip(similarities).map {
//           case (approxFeature, similarity) =>
//             (videoId, similarity, origFeatures, noImg, approxFeature)
//         }
//         rows
//       }
//       .toDF("video_Id", "similarity", "origFeatures", "no_img", "approxFeature")

//     val vectorToArray = udf((vector: Vector) => vector.toArray)

//     val resultDF = similarities
//       .orderBy(col("similarity").desc, col("no_img"))
//       .select(
//         col("video_Id"),
//         col("similarity"),
//         col("origFeatures"),
//         col("no_img"),
//         col("approxFeature")
//       )

//     val sortedResultDF = resultDF
//       .orderBy(
//         col("similarity").desc,
//         col("origFeatures"),
//         col("no_img")
//       )

//     println(
//       "=========================================================================================="
//     )
//     println(
//       "                                    Similarity Result                                     "
//     )
//     println(
//       "=========================================================================================="
//     )
//     sortedResultDF.show(k)
//     println("\n\n")

//     // Video Clips Similarity Result 및 Event 정보 출력
//     val clipsArray = defaultImgFeatureRDD.collect()
//     val windowSize = 3
//     val (videoGraphs, event) =
//       searchFV(clipsArray, queryFeatures, k, windowSize)
//     val top3Graphs = videoGraphs.take(3)

//     top3Graphs.foreach { graph =>
//       println(
//         "=========================================================================================="
//       )
//       println(
//         "                             Video Clips Similarity Result                                "
//       )
//       println(
//         "=========================================================================================="
//       )

//       val dtwDistances = graph.map { imageFeature =>
//         val dtwDist = dtwDistance(queryFeatures, imageFeature.approxFeatures(0))
//         (imageFeature.video_Id, imageFeature.no_img, dtwDist)
//       }

//       val sortedGraph = dtwDistances.sortBy(_._2)
//       val frameNumberGroups = sortedGraph
//         .map { case (videoId, frameNumber, dtwDist) =>
//           (videoId, frameNumber, dtwDist)
//         }
//         .sliding(windowSize)
//         .toList

//       frameNumberGroups.foreach { frameNumbers =>
//         val frameNumberString = frameNumbers.map(_._2).mkString(", ")
//         val videoId = frameNumbers.head._1
//         val dtwDist = frameNumbers.head._3
//         println(
//           s"Video ID: $videoId, Frame Numbers: $frameNumberString, DTW Distance: $dtwDist"
//         )
//       }

//       println("\n\n")
//     }
//     println(
//       "=========================================================================================="
//     )
//     println(
//       "                                        Time Taken                                        "
//     )
//     println(
//       "=========================================================================================="
//     )
//     println(
//       s"Time Taken: ${java.util.concurrent.TimeUnit.NANOSECONDS.toNanos(end - start)}ns"
//     )
//     println("\n\n")

//     defaultImgFeatureRDD.unpersist()

//     spark.stop()
//   }
// }

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
