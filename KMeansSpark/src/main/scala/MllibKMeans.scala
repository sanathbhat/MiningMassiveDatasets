import breeze.linalg.SparseVector
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.IDF

import org.apache.spark.mllib.clustering.{KMeans => KM}
import org.apache.spark.mllib.linalg.Vectors

import java.io._

object MllibKMeans {
	def main(args: Array[String]) {
	val k = args(0).toInt
	val iterations = args(1).toInt
	val tolerance = args(2).toDouble
	val documentFile = args(3)

	val conf = new SparkConf().setAppName("MllibKMeans")
	val sc = new SparkContext(conf)

	val documents = sc.textFile(documentFile)
		.map(line=>{val docTuple = line.split(" ");
		if(docTuple.length==3) (docTuple(0).toInt, (docTuple(1).toInt, docTuple(2).toInt)) else (0, (0, 0))})	//to skip first 3 lines, modify them
			.filter(_._1>0)						//then filter those three lines
				.groupByKey()					//group words of the same document together
					.map({case (k, v)=> v.foldLeft(List[Int]()){ case (document, (word, count)) =>  List.fill(count)(word):::document} })	//map them as a document with words repeated as many times as its count

	//tf
	val hashingTF = new HashingTF()
	val tf: RDD[Vector] = hashingTF.transform(documents)
	//idf
	
	val idf = new IDF().fit(tf)
	val tfidf: RDD[Vector] = idf.transform(tf)
	tfidf.cache()
	//use refection to access the private method toBreeze(). For some strange reason, this useful method is private! I found no other easy way of converting a Vector to a breeze SparseVector which is useful for algebraic manipulation of sparse vectors.
	//val sparseVectorTFIDF = tfidf.map(vector=>vector.getClass.getMethod("toBreeze").invoke(vector).asInstanceOf[SparseVector[Double]])
	//sparseVectorTFIDF.cache
	//Ready to start k-means	
	val clusters = KM.train(tfidf, k, iterations)

	println("Centroids: ")
	clusters.clusterCenters.foreach(println)
	

	//********For Testing*******************************************
	
	//oldAndNewCentroids.saveAsTextFile("OP")
	//oldAndNewCentroids.foreach(line=>println(line))

	//println("No Of old centroids: "+oldCentroids.value.size)
	//println("No Of centroids: "+oldAndNewCentroids.count)
	
  }
}

