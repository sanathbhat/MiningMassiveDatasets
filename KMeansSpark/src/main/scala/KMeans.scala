import breeze.linalg.SparseVector
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.IDF

import java.io._

object KMeans {
	def main(args: Array[String]) {
	//val vocabFile = "data/vocab.nips.txt"
	val k = args(0).toInt
	val iterations = args(1).toInt
	val tolerance = args(2).toDouble
	val documentFile = args(3)

	val conf = new SparkConf().setAppName("KMeans")
	val sc = new SparkContext(conf)

	val documents = sc.textFile(documentFile)
		.map(line=>{val docTuple = line.split(" ");
		if(docTuple.length==3) (docTuple(0).toInt, (docTuple(1).toInt, docTuple(2).toInt)) else (0, (0, 0))})	//to skip first 3 lines, modify them
			.filter(_._1>0)						//then filter those three lines
				.groupByKey()					//group words of the same document together
					.map({case (k, v)=> v.foldLeft(List[Int]()){ case (document, (word, count)) =>  List.fill(count)(word):::document} })	//map them as a document with words repeated as many times as its count
	documents.cache
	//tf
	val hashingTF = new HashingTF()
	val tf: RDD[Vector] = hashingTF.transform(documents)
	//idf
	tf.cache()	
	val idf = new IDF().fit(tf)
	val tfidf: RDD[Vector] = idf.transform(tf)
	tfidf.cache
	//use refection to access the private method toBreeze(). For some strange reason, this useful method is private! I found no other easy way of converting a Vector to a breeze SparseVector which is useful for algebraic manipulation of sparse vectors.
	val sparseVectorTFIDF = tfidf.map(vector=>vector.getClass.getMethod("toBreeze").invoke(vector).asInstanceOf[SparseVector[Double]])
	sparseVectorTFIDF.cache
	//Ready to start k-means	
	//take k samples for centroids and index them. The indices are needed for grouping the documents by the centroids as the centroid SparkVectors couldn't be determined to be equal to one another and groupByKey wouldn't work as expected
	var oldCentroids = sc.broadcast(sparseVectorTFIDF.takeSample(false, k)
				.zipWithIndex				//index each vector of the RDD to get a new RDD
					.map(vector=>vector.swap)	//switch the positions of the index and the vector (v, i)=>(i, v)
						.toMap)			//get a immutable map
	var converge = 0.0
	var currentIteration = 0
	
	do {
		//e-step and m-step!
		var indexAndNewCentroids = sparseVectorTFIDF.map(d=>(oldCentroids.value.map({ case (i,c) => (i, (d-c).dot(d-c) ) }).minBy(_._2)._1, d))	//map to (i, d) for each document d where i is the index of the nearest centroid to d
			.groupByKey()									//group documents by the indices of their nearest centroid
				.map({case (i, ds)=> (i, ds.reduceLeft(_+_):/ds.size.toDouble)})	//average the document vectors to get new centroids
					.collect.toMap	
		
		//convergence update
		converge = indexAndNewCentroids.foldLeft(0.0)({case (conv, (i, c)) => conv+(c-oldCentroids.value(i)).dot(c-oldCentroids.value(i))}) 	//compute convergence as the difference between corresponding old and new centroids
		
		println("Iteration "+currentIteration + ": Convergence = " + converge)
		
		oldCentroids = sc.broadcast(indexAndNewCentroids)
		
		currentIteration+=1
	}while(currentIteration<iterations && converge>tolerance)
	
	println("Iterations to converge ="+ currentIteration)
	println("Centroid residual ="+converge)
	println("Centroids: ")
	oldCentroids.value
		.map({case (i, v) => v})
			.foreach(centroid=>println(centroid))
	

	//********For Testing*******************************************
	
	//oldAndNewCentroids.saveAsTextFile("OP")
	//oldAndNewCentroids.foreach(line=>println(line))

	//println("No Of old centroids: "+oldCentroids.value.size)
	//println("No Of centroids: "+oldAndNewCentroids.count)
	
  }
}

