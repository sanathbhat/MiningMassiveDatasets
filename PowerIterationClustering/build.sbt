name := "PIClustering"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.5.1" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.5.1" % "provided"

libraryDependencies ++= Seq(

"org.scalanlp" %% "breeze" % "0.11.2",

	"org.scalanlp" %% "breeze-natives" % "0.11.2",

	"org.scalanlp" %% "breeze-viz" % "0.11.2"
)

