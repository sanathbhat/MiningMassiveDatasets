import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Matrix, DenseMatrix => DM, Vectors}
import org.apache.spark.mllib.linalg.distributed._
import scala.util.Random

object DSGDMF {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("DSGD-MF").setMaster("local[*]"))
    sc.setLogLevel("ERROR")
    val dataFile = "data/nf_subsample.csv"
//    val dataFile = "data/nf_tiny.csv"
//    val spark = SparkSession.builder().appName("DSGD-MF").master("local[*]").getOrCreate()
//    import spark.implicits._
//
//    val df_raw = spark.read.csv(dataFile)
//
//    val df = df_raw.map{case a=>(a.getString(0).toInt, a.getString(1).toInt, a.getString(2).toInt)}.toDF("userId", "movieId", "rating")
//
//    val maxIds = df.agg(max("userId").as("maxU"), max("movieId").as("maxM")).head()

    //get input values
    val I = args(0).toInt         //number of iterations
    val B = args(1).toInt         //number of parallel blocks at a time = number of workers
    val F = args(2).toInt         //number of factors i.e. number of cols in W and number of rows in H
    val beta = args(3).toDouble      //decay constant for step size
    val lambda = args(4).toDouble    //regularization constant

    //set up V matrix
    val v = new CoordinateMatrix(
      sc.textFile(dataFile)
        .map(_.split(",").map(_.toInt))
          .map(a => new MatrixEntry(a(0)-1, a(1)-1, a(2)))
      )

    val startTime = System.nanoTime()

    val m = v.numRows().toInt       //number of users
    val n = v.numCols().toInt       //number of movies
    val V = sc.broadcast(v.toBlockMatrix(Math.ceil(m/B.toDouble).toInt, Math.ceil(n/B.toDouble).toInt))

    //create W and H' matrices as maps of a block matrix
    val wRowsPerBlock = sc.broadcast(Math.ceil(m/B.toDouble).toInt)
    val hColsPerBlock = sc.broadcast(Math.ceil(n/B.toDouble).toInt)

    var W = sc.broadcast(new IndexedRowMatrix(
      sc.parallelize(0 to m-1)
        .map(i => new IndexedRow(i, Vectors.dense(Array.fill(F)(Random.nextDouble))))
    ).toBlockMatrix(wRowsPerBlock.value, F).blocks.collectAsMap())

    var H = sc.broadcast(new IndexedRowMatrix(
      sc.parallelize(0 to n-1)
        .map(i => new IndexedRow(i, Vectors.dense(Array.fill(F)(Random.nextDouble))))
    ).toBlockMatrix(hColsPerBlock.value, F).transpose.blocks.collectAsMap())

    //ready to start DSGD
    var totalSGDUpdatesYet = 0
    for(i <- 0 to I) {      //iterations
      val s = Random.nextInt(B)   //for a random stratum
      val updatedWH = V.value.blocks
        .filter{case((bi, bj), _) => bj == (bi + s) % B}      //filter blocks of current stratum
        .repartition(B)
        .map{case((bi, bj), vb) => ((bi,bj), sgdMfOnePass(vb, W.value(bi, 0), H.value(0, bj), totalSGDUpdatesYet, beta, lambda))}   //map each block to corresponding W and H updates
        .cache()

      val updatedWBlocks = updatedWH.map{case((wr, _), (wb, _, _)) => ((wr, 0), wb)}
      val updatedHBlocks = updatedWH.map{case((_, hc), (_, hb, _)) => ((0, hc), hb)}
      totalSGDUpdatesYet += updatedWH.map(_._2._3).reduce(_+_)

      W = sc.broadcast(new BlockMatrix(updatedWBlocks, wRowsPerBlock.value, F).blocks.collectAsMap())
      H = sc.broadcast(new BlockMatrix(updatedHBlocks, F, hColsPerBlock.value).blocks.collectAsMap())

      //compute L2 loss
      val Wtemp = new BlockMatrix(sc.makeRDD(W.value.toSeq), wRowsPerBlock.value, F).toLocalMatrix()
      val Htemp = new BlockMatrix(sc.makeRDD(H.value.toSeq), F, hColsPerBlock.value).transpose.toLocalMatrix()

      val WB = new BDM[Double](Wtemp.numRows, Wtemp.numCols, Wtemp.toArray)
      val HB = new BDM[Double](Htemp.numRows, Htemp.numCols, Htemp.toArray)

      val lnzsl = v.entries
        .map(me => (me.i.toInt, me.j.toInt, me.value))
        .map{case(i, j, v) => Math.pow(v - WB(i, ::)*HB(j, ::).t, 2)}
        .reduce(_+_)

      val l2 = lnzsl + lambda*(sum(WB^:^2.0) + sum(HB^:^2.0))

      println("Iteration: " + i + " => L2 loss: " + l2)// + " | Total SGD updates = " + totalSGDUpdatesYet)

    }

//    println(W.value)
//    println(H.value)
    println("Total time taken = " + (System.nanoTime()-startTime)/1e9d + " secs")
  }

  /**
    * * Function that performs one pass of a standard sgd update for matrix factorization
    * @param v  The mxn matrix under consideration
    * @param w  The mxr factor(W) matrix
    * @param h  The rxn factor(H) matrix
    * @param tSGDUpdates  The total number of SGD updates done so far, equal to the number of entries in V matrix times the number of iterations so far
    * @param beta Decay constant for step size
    * @param lambda Regularization constant for loss function
    * @return   Updated W and H matrices with the total number of SGD updates made during the pass
    */
  def sgdMfOnePass(v:Matrix, w:Matrix, h:Matrix, tSGDUpdates:Int, beta:Double, lambda:Double) : (Matrix, Matrix, Int) = {
    val N = v.numActives

    val W = new BDM[Double](w.numRows, w.numCols, w.toArray)
    val H = new BDM[Double](h.numRows, h.numCols, h.toArray)
    var nPrime = 0

    //sgd update for each point in v
    v.asML.foreachActive{
      case(i, j, vVal) => {
        val eps = Math.pow(1000000 + nPrime + tSGDUpdates, -beta)
        val Wi = new BDV(W(i, ::).t.toArray).t      //a copy of row i that is independent of changes to the original DenseMatrix
        val Hj = new BDV(H(::, j).toArray)

        val Ni = Wi.t.size                                            //IDE issue here, works perfectly fine on REPL and executes here as well
        val Nj = Hj.size

        val WiHj = (Wi * Hj)

        val dwi = -2.0*(vVal - WiHj)*Hj   + (2.0*lambda/Ni)*Wi.t     //IDE issue here, works perfectly fine on REPL and executes here as well
        val dhj = -2.0*(vVal - WiHj)*Wi.t + (2.0*lambda/Nj)*Hj       //IDE issue here, works perfectly fine on REPL and executes here as well

//        println("Epsilon="+eps)

        W(i, ::) := (Wi.t - eps*N*dwi).t              //IDE issue here, works perfectly fine on REPL and executes here as well
        H(::, j) := Hj - eps*N*dhj                    //IDE issue here, works perfectly fine on REPL and executes here as well

        nPrime += 1
      }
    }
    (new DM(W.rows, W.cols, W.data, W.isTranspose), new DM(H.rows, H.cols, H.data, H.isTranspose), N)
  }
}