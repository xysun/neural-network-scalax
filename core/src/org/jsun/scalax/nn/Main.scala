package org.jsun.scalax.nn

import java.nio.file.Paths

import cats.effect.{ContextShift, IO}
import fs2.{Chunk, io}
import org.jsun.scalax.nn.models.{LogisticRegression, NeuralNetwork}
import org.jsun.scalax.nn.graph.Tensor._

import scala.concurrent.ExecutionContext.Implicits.global

object Main extends App {

  println("hello")

  val start = System.currentTimeMillis()

  implicit val ioContextShift: ContextShift[IO] = IO.contextShift(global)

  // read MNIST dataset
  val trainLabelFileName = "/Users/xiayunsun/Downloads/train-labels-idx1-ubyte"
  val trainImgFileName = "/Users/xiayunsun/Downloads/train-images-idx3-ubyte"

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
  val trainLabelsPreprocessed: fs2.Stream[IO, Int] = trainLabels.map(i => if (i == 0) 1 else 0)

  val trainData: fs2.Stream[IO, (Int, Matrix2D[Double])] = trainLabelsPreprocessed.zip(trainImagesPreprocessed)

  // shuffle training data: todo

  // single neuron: W = 784x1, with sigmoid as activation function
  // todo: can we get rid of var?
  val initialWeights = Vector.tabulate(784)(_ => 0.01)
  val initialBias: Double = 0

  type Param = (Vector[Double], Double)


  val network: NeuralNetwork = new LogisticRegression

  val learningRate = 1
  val batchSize = 1000


  val trainSink: (Param, Chunk[(Int, Matrix2D[Double])]) => Param = {
    case ((weights, bias), chk) =>

      val ys = chk.map(_._1)
      val yHats = chk.map { case (y, img) => network.forwardProp(weights, bias, img) }
      val losses = yHats.zip(ys).map { case (yHat, y) => network.loss(yHat, y) }

      val avgLoss = losses.toVector.sum / chk.size
      println(s"avg loss for chunk: $avgLoss")

      val gradients: Chunk[(Vector[Double], Double)] = yHats.zip(chk).map { case (yHat, (y, img)) => network.backProp(yHat, y, img) }

      val avgBiasGradient = gradients.map(_._2).toVector.sum / chk.size

      val sumWeightsGradient = gradients.map(_._1).toVector
        .reduce[Vector[Double]] { case (v1, v2) => v1.zip(v2).map(t => t._1 + t._2) }

      val avgWeightsGradient = sumWeightsGradient.map(_ / chk.size)

      (
        weights.zip(avgWeightsGradient).map { case (w, g) => w - learningRate * g },
        bias - learningRate * avgBiasGradient
      )

  }

  val graph = for {
    _ <- BinaryStateNode(MatMul)
    _ <- BinaryStateNode(Add)
    ans <- BinaryStateNode(CrossEntropyLoss)
  } yield ans

  val trainSinkWithGraph: (Param, Chunk[(Int, Matrix2D[Double])]) => Param = {
    case ((weights, bias), chk) =>

      val w = Node("w", Matrix(Vector(weights)), Ident)
      val b = Node("b", Scalar(bias), Ident)

      val a =
        chk.map { case (_y, img) =>
          val x = Node("x", Matrix(img.flatten.map(Vector(_))), Ident)
          val y = Node("y", Scalar(_y), Ident)
          val args = List(w, x, b, y)

          val init = (args, emptyGraph(args))

          val ((nodes, g), loss) = graph.run(init).value
          require(nodes.size == 1)
          val gradients = g.backProp(List(w, b), nodes.head)

          (gradients("w").asInstanceOf[Matrix].m, gradients("b").asInstanceOf[Scalar].v, loss.asInstanceOf[Scalar].v)
        }

      val wGradients = a.map(_._1)
      val bGradients = a.map(_._2)
      val losses = a.map(_._3)

      val avgLoss = losses.toVector.sum / chk.size
      println(s"avg loss: $avgLoss")

      val avgBiasGradient = bGradients.toVector.sum / chk.size

      val sumWeightsGradient = wGradients.toVector
          .map(_.head) // we know it's 1x784
        .reduce[Vector[Double]] { case (v1, v2) => v1.zip(v2).map(t => t._1 + t._2) }

      val avgWeightsGradient = sumWeightsGradient.map(_ / chk.size)

      (
        weights.zip(avgWeightsGradient).map { case (w, g) => w - learningRate * g },
        bias - learningRate * avgBiasGradient
      )
  }

  val (trainedWeights, trainedBias) =
    trainData.chunkN(batchSize).fold((initialWeights, initialBias))(trainSinkWithGraph).compile.toVector.unsafeRunSync()
      .head

  // training error
  val prediction =
    trainData.map { case (y, img) => {
      val yHat = network.forwardProp(trainedWeights, trainedBias, img)
      val prediction = if (yHat > 0.5) 1 else 0
      y == prediction
    }
    }.compile.toVector.unsafeRunSync()

  println(s"correct prediction: ${prediction.count(t => t) / prediction.size.toDouble}")

  println(s"elapsed: ${(System.currentTimeMillis() - start) / 1000.0} seconds")


}
