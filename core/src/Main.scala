import java.nio.file.Paths

import Matrix2DSyntax._
import cats.effect.{ContextShift, IO}
import fs2.{Chunk, Sink, io}

import scala.math.{E, log, pow}
import scala.concurrent.ExecutionContext.Implicits.global

object Main extends App {

  println("hello")

  implicit val ioContextShift: ContextShift[IO] = IO.contextShift(global)

  // read MNIST dataset
  val trainLabelFileName = "/Users/xiayunsun/Downloads/train-labels-idx1-ubyte"
  val trainImgFileName   = "/Users/xiayunsun/Downloads/train-images-idx3-ubyte"

  val trainLabels: fs2.Stream[IO, Int] = // todo: must we use IO?
    io.file
      .readAll[IO](path = Paths.get(trainLabelFileName), global, chunkSize = 1024)
      .drop(8) // skip magic number and size
      .map(_.toInt)

  private val imgDimension = 28
  val trainImages: fs2.Stream[IO, Matrix2D[Int]] =
    io.file
      .readAll[IO](path = Paths.get(trainImgFileName), global, chunkSize = 1024) // each img is 784 bytes
      .drop(16) // 8 more bytes for number of rows and number of columns
      .map(java.lang.Byte.toUnsignedInt) // java is big endian
      .chunkN(imgDimension * imgDimension, allowFewer = false)
      .map(_.toVector)
      .map(_.toMatrix2D(rows = imgDimension, cols = imgDimension))

//  val trainData: fs2.Stream[IO, (Int, Matrix2D[Int])] = trainLabels.zip(trainImages)

  // optional: sanity check
//  Preprocessor.sanityCheck(trainLabels, trainImages)

  // proprocess: x /= 255. y: binary classifier on digit 0
  val trainImagesPreprocessed: fs2.Stream[IO, Matrix2D[Double]] = trainImages.map(_.mmap(_ / 255.0))
  val trainLabelsPreprocessed: fs2.Stream[IO, Int]              = trainLabels.map(i => if (i == 0) 1 else 0)

  val trainData:fs2.Stream[IO, (Int, Matrix2D[Double])] = trainLabelsPreprocessed.zip(trainImagesPreprocessed)

  // shuffle training data: todo

  // single neuron: W = 784x1, with sigmoid as activation function
  // todo: can we get rid of var?
  val initialWeights = Vector.tabulate(784)(_ => 0.01)
  val initialBias:Double = 0

  type Param = (Vector[Double], Double)

  // forward propagation
  def forwardProp(weights: Vector[Double], bias:Double, image:Matrix2D[Double]):Double = {
    val z = weights.zip(image.flatten).map{case (w,x) => w*x}.sum + bias
    1 / (1 + pow(E, -z))
  }

  def backProp(yHat:Double, y:Int, image:Matrix2D[Double]):(Vector[Double], Double) = {

    val z = -log(1/yHat - 1)

    // dLoss/dZ
    val a = (1-2*y)*z
    val d1 = 1 / (1 + pow(E, a)) * pow(E, a) * (1-2*y)
    // dZ/dyHat
    val d2 = 1 / (yHat * (1-yHat))
    // dYHat/dz
    val d3 = yHat * (1-yHat)

    // total
    val d = d1 * d2 * d3

    val weightsGradient = image.flatten.map( x => d * x)

    val biasGradient = d

    (weightsGradient, biasGradient)
  }

  def loss(yHat:Double, y:Int):Double = {
    val z = -log(1/yHat - 1)

    log(1 + pow(E, (1-2*y)*z))
  }

  val learningRate = 1
  val batchSize = 1000


  val trainSink:(Param, Chunk[(Int, Matrix2D[Double])]) => Param = { case ((weights, bias), chk) =>

      val ys = chk.map(_._1)
      val yHats = chk.map{case (y, img) => forwardProp(weights, bias, img)}
      val losses = yHats.zip(ys).map{case (yHat, y) => loss(yHat, y)}

      val avgLoss = losses.toVector.sum / chk.size
      println(s"avg loss for chunk: $avgLoss")

      val gradients:Chunk[(Vector[Double], Double)] = yHats.zip(chk).map{case (yHat, (y, img)) => backProp(yHat, y, img)}

      val avgBiasGradient = gradients.map(_._2).toVector.sum / chk.size

      val sumWeightsGradient = gradients.map(_._1).toVector
        .reduce[Vector[Double]]{case (v1,v2) => v1.zip(v2).map(t => t._1 + t._2)}

      val avgWeightsGradient = sumWeightsGradient.map(_ / chk.size)

      (
        weights.zip(avgWeightsGradient).map{case (w, g) => w - learningRate * g},
        bias - learningRate * avgBiasGradient
      )

  }

  val (trainedWeights, trainedBias) =
  trainData.chunkN(batchSize).fold((initialWeights, initialBias))(trainSink).compile.toVector.unsafeRunSync()
    .head

  // training error
  val prediction =
    trainData.map{case (y, img) => {
      val yHat = forwardProp(trainedWeights, trainedBias, img)
      val prediction = if (yHat > 0.5) 1 else 0
      y == prediction
    }}.compile.toVector.unsafeRunSync()

  println(s"correct prediction: ${prediction.count(t => t)/prediction.size.toDouble}")


}
