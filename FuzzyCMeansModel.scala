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
   * @param data INput data
   * @return Returs the membership matrix
   */
  def getMembershipMatrix(data: RDD[Vector]) : RDD[Array[Double]] = {
    data.map(p => getMembershipVector(p))
  }

  def getMembershipVector(point: Vector) : Array[Double] = {
    val membershipVec = Array.fill[Double](clusterCentersNum)(0.0)
    val point_distance = Array.fill[Double](clusterCentersNum)(0.0)
    var total_distance = 0.0

    for(i <- 0 until clusterCentersNum) {
      val dist = MLUtils.fastSquaredDistance(point, 2.0, clusterCenters(i), 2.0) + 0.001
      point_distance(i) = math.pow(dist, 1/( FuzzyCMeans.getFuzzynessCoefficient - 1))      
      total_distance += (1 / point_distance(i))

    }

    for(i <- 0 until clusterCentersNum) {
      // val u_i_j_m = math.pow(point_distance(i) * total_distance, -FuzzyCMeans.getFuzzynessCoefficient)
      // val u_i_j = point_distance(i) * total_distance
      val u_i_j = math.pow(point_distance(i) * total_distance, -1)
      membershipVec(i) = u_i_j
    }
    membershipVec
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
      // val localCentroids = centroids.map(Cluster.apply).collect()
      val localCentroids = centroids.rdd.map(Cluster.apply).collect()
      assert(c == localCentroids.length)
      new FuzzyCMeansModel(localCentroids.sortBy(_.id).map(_.point))
    }
  }
}






