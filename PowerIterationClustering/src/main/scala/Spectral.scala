import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.{ PowerIterationClustering, PowerIterationClusteringModel }
import breeze.plot._

object Spectral {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("PIClustering")//.setMaster("local[*]")
        val sc = new SparkContext(conf)

        val k = args(0).toInt //number of clusters
        val x = args(1).toInt //max iterations
        val g = sc.broadcast(args(2).toDouble) //gamma
        val dataFile = args(3) //data file

        val indexedDataPoints = sc.textFile(dataFile)
            .map(line => line.split("\t"))
            .zipWithIndex()
            .map(x => x.swap).cache

        val affinityMatrix = indexedDataPoints.cartesian(indexedDataPoints)
            .filter(x => x._1._1 < x._2._1)
            .map({
                case (((indXi), (xi)), ((indXj), (xj))) => (indXi, indXj,
                    Math.exp(-g.value * (Math.pow(xj(0).toDouble - xi(0).toDouble, 2) + Math.pow(xj(1).toDouble - xi(1).toDouble, 2))))
            }).cache

        //ready to start power iteration
        val pic = new PowerIterationClustering().setK(k).setMaxIterations(x)
        val model = pic.run(affinityMatrix)
        
        val colors = Array[String]("RED", "BLUE", "GREEN", "BLACK")
        
        //map the points out as x, y, cluster color
        val clusteredCoordinates = model.assignments.map(x => (x.id, x.cluster))
            .join(indexedDataPoints)
            .map(point => (point._2._1, point._2._2(0), point._2._2(1)))
        
        //group by cluster
        val groupedClusters = clusteredCoordinates.groupBy({case(c, x, y) => c})
        
        val f = Figure();
        val p = f.subplot(0)
        p.xlabel = "x"
        p.ylabel = "y"

        val series = sc.broadcast(new Array[Series](k));
        
        
        groupedClusters.foreach({ case(cluster, points) =>
            val xCoords = points.map({case(cluster, x, y) => x.toDouble}).toSeq;
            val yCoords = points.map({case(cluster, x, y) => y.toDouble}).toSeq;
            //for each group plot using a different color
            series.value(cluster) = plot(xCoords, yCoords, '.', colors(cluster));
        })

        series.value.foreach { x => p+=x }
        f.saveas("scatter.png")
	println("Done. Saved output to scatter.png")
    }
}
