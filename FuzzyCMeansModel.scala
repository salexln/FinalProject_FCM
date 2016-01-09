/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import scala.collection.JavaConverters._
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.util.MLUtils

/**
 * A clustering model for Fuzzy C - Means
 * @param clusterCenters centers of the clusters
 */
class FuzzyCMeansModel(val clusterCenters: Array[Vector])
  extends Saveable with Serializable with PMMLExportable {


  override protected def formatVersion: String = "1.0"

  override def save(sc: SparkContext, path: String): Unit = {
     FuzzyCMeansModel.SaveLoadV1_0.save(sc, this, path)
  }
  /**
   * A Java-friendly constructor that takes an Iterable of Vectors.
   */
  def this(centers: java.lang.Iterable[Vector]) = this(centers.asScala.toArray)

  /**
   *
   * @return Total number of clusters.
   */
  def clusterCentersNum() : Int = clusterCenters.length

  /**
   *
   * @return The cluster centers
   */
  def centers(): Array[Vector] = {
    clusterCenters
  }

  /**
   *
   * @param data_point The data point you want to get the membership vector for
   * @return Membership vector for the input point: defines the membership
   *         of the input point to each cluster
   */
  def getMembershipVector(data_point: VectorWithNorm): Array[Double] = {
    val ui = Array.fill[Double](clusterCentersNum())(0)    
    
    val clusterCenters = centers()

    val distance_from_center = Array.fill[Double](clusterCentersNum())(0)

    // compute distances:
    var total_distance = 0.0
    for (j <- 0 until clusterCentersNum()) {
      val center_with_norm = new VectorWithNorm(clusterCenters(j))
      // val dist = KMeans.fastSquaredDistance(center_with_norm, data_point)
      val dist = 1
      // val dist = MLUtils.fastSquaredDistance(clusterCenters(j), 2.0, data_point.vector, 2.0)

      // distance_from_center(j) = math.pow(dist,
      //                           1/( FuzzyCMeans.getFuzzynessCoefficient - 1))
      distance_from_center(j) = 1.0

      total_distance += (1 / distance_from_center(j))
    }

    // compute the u_i_j:
    for (j <- 0 until clusterCentersNum()) {
      // val u_i_j: Double = math.pow(distance_from_center(j) * total_distance,
      //   -1)      
      // ui(j) = u_i_j      
      ui(j) = 1.0
    }
    ui
  }

  // def getMembershipMatrix(data: RDD[Vector]) : Array[Array[Double]] = {
  //   val norms = data.map(Vectors.norm(_, 2.0))
  //   norms.persist()
  //   val zippedData = data.zip(norms).map { case (v, norm) =>
  //     new VectorWithNorm(v, norm)
  //   }
  //   // val membershipMatrix = Array
  //   // val membershipMatrix = Array.ofDim[Double](zippedData.size, clusterCentersNum())
  //   var membershipMatrixBuffer = new ArrayBuffer[Array[Double]]()

  //   zippedData.foreach { data_point =>      
  //     // val memVector = getMembershipVector(data_point)
  //     val temp = Array.fill[Double](clusterCentersNum())(0)
  //     membershipMatrixBuffer += temp
  //   }
  //   membershipMatrixBuffer.toArray
  // }

  // def getMembershipMatrix222(data: RDD[Vector]) : Array[Array[Double]] = {
  //   val mem = new ArrayBuffer[Array[Double]]()
    
  //   val norms = data.map(Vectors.norm(_, 2.0))
  //   norms.persist()
  //   val zippedData = data.zip(norms).map { case (v, norm) =>
  //     new VectorWithNorm(v, norm)
  //   }

  //   def mergeMatrixes(a: ArrayBuffer[Array[Double]], b: ArrayBuffer[Array[Double]]): ArrayBuffer[Array[Double]] = {
  //     // add matrix b to matrix a by adding each line:
  //     b.foreach{ line =>
  //       a += line }      
  //     a
  //   }      

  //   val new_center_candidates = zippedData.mapPartitions { data_ponts =>

  //       var membershipMatrixBuffer = new ArrayBuffer[Array[Double]]()
  //       data_ponts.foreach { data_point =>
  //         // val memVector = getMembershipVector(data_point)  
  //         val memVector = Array.fill[Double](clusterCentersNum())(0)    

  //         val point_distance = Array.fill[Double](clusterCentersNum)(0)

  //         var arr:Array[Double] = data_point.vector.toArray
  //         val temp_point = new VectorWithNorm(arr.clone)          
  //         var total_distance = 0.0

  //         // computation of the distance of data_point from each cluster:
  //         for (j <- 0 until clusterCentersNum) {
  //           // the distance of data_point from cluster j:
  //           // val cluster_to_point_distance =
  //           //   KMeans.fastSquaredDistance(clusterCenters(j), temp_point)
  //           val cluster_to_point_distance = MLUtils.fastSquaredDistance(clusterCenters(j), 2.0, temp_point.vector, 2.0)
  //           point_distance(j) = math.pow(cluster_to_point_distance, 1/( FuzzyCMeans.getFuzzynessCoefficient - 1))

  //           // update the total_distance:
  //           total_distance += (1 / point_distance(j))
  //         }

  //         for (j <- 0 until clusterCentersNum) {
  //           // calculation of (u_ij)^m:
  //           val u_i_j_m: Double = math.pow(point_distance(j) * total_distance, -FuzzyCMeans.getFuzzynessCoefficient)
  //           // val u_i_j_m: Double = 1.0
  //           memVector(j) = u_i_j_m
  //         }
          
  //         membershipMatrixBuffer += memVector        
  //       }

  //       // val out_tuple = for (j <- 0 until 2) yield {
  //       //   (j, (1, 1))        
  //       // }
  //       // out_tuple.iterator
  //       // val tuple = for(j <-0 until 1) yield {
  //       //   (membershipMatrixBuffer)
  //       // }
  //       // tuple.iterator


  //   }.reduce(mergeMatrixes)//.collectAsMap
  //    // .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2)).collectAsMap()
  //   //.reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2)).collectAsMap()

  //   // zippedData.mapPartitions { data_points =

  //   //   data_points.foreach { data_point =>
  //   //     val ui = Array.fill[Double](4)(0)  
  //   //     mem += ui
  //   //   }
  //   // }

  //   // data.foreach { data_point =>     
  //   //   val ui = Array.fill[Double](4)(0)  
  //   //   mem += ui
  //   // }
  //   mem.toArray
  // }

  // def getMatrix(data: RDD[Vector]) : RDD[Vector] = {
  //   val norms = data.map(Vectors.norm(_, 2.0))
  //   norms.persist()
  //   val zippedData = data.zip(norms).map { case (v, norm) =>
  //     new VectorWithNorm(v, norm)
  //   }

  //   def mergeMatrixes(a: Array[Array[Double]], b: Array[Array[Double]]): Array[Array[Double]] = {
  //     var tempMemMtrix = ArrayBuffer[Array[Double]]()
  //     a.foreach{ point =>
  //       tempMemMtrix += point 
  //     }
  //     b.foreach{ point =>
  //       tempMemMtrix += point
  //     }
  //     tempMemMtrix.toArray      
  //   }
  //   data //for compolation reasons
  // }

  
  def predictCenters(data: RDD[Vector]): RDD[Int] = {
    data.map(p => findClosestCenter(p) )
  }

  def findClosestCenter(point: Vector) : Int = {
    var min:Double = 1000
    var min_idx:Int = 0
    for (i <- 0 until clusterCentersNum() ) {
      val dist = MLUtils.fastSquaredDistance(point, 2.0, clusterCenters(i), 2.0)
      if(dist < min) {
        min = dist
        min_idx = i
      }
    }
    min_idx
  }

  def predictValues(data: RDD[Vector]): RDD[Double] = {
    data.map(p => findClosestCenterValue(p) )
  }

  def findClosestCenterValue(point: Vector) : Double = {
    var min:Double = 1000
    var min_idx:Int = 0
    for (i <- 0 until clusterCentersNum() ) {
      val dist = MLUtils.fastSquaredDistance(point, 2.0, clusterCenters(i), 2.0)
      if(dist < min) {
        min = dist
        min_idx = i
      }
    }
    min
  }


  def predictAll(data: RDD[Vector]) : RDD[Array[Double]] = {
    data.map(p => findAll(p))
  }

  def findAll(point: Vector): Array[Double] = {
    
    val membershipVec = Array.fill[Double](clusterCentersNum)(0.0)
    for (i <- 0 until clusterCentersNum() ) {
      val dist = MLUtils.fastSquaredDistance(point, 2.0, clusterCenters(i), 2.0)
      membershipVec(i) = dist
    }
    membershipVec
  }


  def predictMatrix(data: RDD[Vector]) : RDD[Array[Double]] = {
    data.map(p => findVector(p))
  }

  def findVector(point: Vector) : Array[Double] = {
    val membershipVec = Array.fill[Double](clusterCentersNum)(0.0)
    val point_distance = Array.fill[Double](clusterCentersNum)(1.0)
    var total_distance = 0.0

    for(i <- 0 until clusterCentersNum) {
      val dist = MLUtils.fastSquaredDistance(point, 2.0, clusterCenters(i), 2.0) + 0.001    
      point_distance(i) = math.pow(dist, 1/( FuzzyCMeans.getFuzzyness - 1))          
      total_distance += (1 / point_distance(i))

    }
    
    for(i <- 0 until clusterCentersNum) {      
      val u_i_j_m = math.pow(point_distance(i) * total_distance , -FuzzyCMeans.getFuzzyness)    
      membershipVec(i) = u_i_j_m      
    }
    membershipVec
  }


  // def getFuzzySets(data: RDD[Vector]) = {
  //   val norms = data.map(Vectors.norm(_, 2.0))
  //   norms.persist()
  //   val zippedData = data.zip(norms).map { case (v, norm) =>
  //     new VectorWithNorm(v, norm)
  //   }
    
  //   def mergeMatrixes(a: Array[Array[Double]], b: Array[Array[Double]]): Array[Array[Double]] = {
  //     var tempMemMtrix = ArrayBuffer[Array[Double]]()
  //     a.foreach{ point =>
  //       tempMemMtrix += point 
  //     }
  //     b.foreach{ point =>
  //       tempMemMtrix += point
  //     }
  //     tempMemMtrix.toArray      
  //   }

  //   val mapper = zippedData.mapPartitions { points =>
  //     val pointsCopy = points.duplicate
  //     val nPoints = pointsCopy._1.length
  //     val membershipMatrix = Array.ofDim[Double](nPoints, clusterCentersNum)        

  //     val singleDist = Array.fill[Double](clusterCentersNum)(0)
  //     val numDist = Array.fill[Double](clusterCentersNum)(0)
  //     var i = 0

  //     pointsCopy.foreach { point =>
  //       var denom = 0.0
  //       for (j <- 0 until clusterCentersNum) {
  //         singleDist(j) = (MLUtils.fastSquaredDistance(point.vector, 2.0, clusterCenters(j), 2.0))            
  //         numDist(j) = math.pow(singleDist(j), (1 / (FuzzyCMeans.getFuzzynessCoefficient - 1)))
  //         denom += (1 / numDist(j))
  //       }
  //       for (j <- 0 until clusterCentersNum) {
  //         val u = (numDist(j) * denom) //uij^m  
  //         membershipMatrix(i)(j) = (1 / u)
  //       }      
  //       i += 1
  //     }
  //   }.collectAsMap(mergeMatrixes)

  // }



  /**
   *
   * @param data_point The data point you want to get the highest associated center for
   * @return The index of the most associated center
   */
  def findMostAssociatedCenter(data_point: VectorWithNorm) : Int = {
    val u_i = getMembershipVector(data_point)

    // find the max value in the u_i array:
    u_i.indexOf(u_i.max)
  }
}



object FuzzyCMeansModel  extends Loader[FuzzyCMeansModel] {

  override def load(sc: SparkContext, path: String): FuzzyCMeansModel = {
    FuzzyCMeansModel.SaveLoadV1_0.load(sc, path)
  }

  private case class Cluster(id: Int, point: Vector)

  private object Cluster {
    def apply(r: Row): Cluster = {
      Cluster(r.getInt(0), r.getAs[Vector](1))
    }
  }

  private[clustering]
  object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    private[clustering]
    val thisClassName = "org.apache.spark.mllib.clustering.FuzzyCMeansModel"

    def save(sc: SparkContext, model: FuzzyCMeansModel, path: String): Unit = {
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      val metadata = compact(render(
        ("class" -> thisClassName) ~ ("version" -> thisFormatVersion)
                                   ~ ("c" -> model.clusterCentersNum() )))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))
      val dataRDD = sc.parallelize(model.clusterCenters.zipWithIndex).map { case (point, id) =>
        Cluster(id, point)
      }.toDF()
      dataRDD.write.parquet(Loader.dataPath(path))
    }

    def load(sc: SparkContext, path: String): FuzzyCMeansModel = {
      implicit val formats = DefaultFormats
      val sqlContext = new SQLContext(sc)
      val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path)
      assert(className == thisClassName)
      assert(formatVersion == thisFormatVersion)
      val c = (metadata \ "c").extract[Int]
      val centroids = sqlContext.read.parquet(Loader.dataPath(path))
      Loader.checkSchema[Cluster](centroids.schema)
      val localCentroids = centroids.map(Cluster.apply).collect()
      assert(c == localCentroids.length)
      new FuzzyCMeansModel(localCentroids.sortBy(_.id).map(_.point))
    }
  }
}






