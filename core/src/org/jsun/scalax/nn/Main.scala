package org.jsun.scalax.nn

import java.nio.file.Paths

import cats.effect.{ ContextShift, IO }
import fs2.{ Chunk, io }
import org.jsun.scalax.nn.models.{ LogisticRegression, NeuralNetwork }
import org.jsun.scalax.nn.datatypes._
import org.jsun.scalax.nn.graph.{ BinaryStateNode, Node, SingleStateNode }
import org.jsun.scalax.nn.graph.operations.Ident
import org.jsun.scalax.nn.graph.emptyGraph
import org.jsun.scalax.nn.graph.operations.binary._
import org.jsun.scalax.nn.graph.operations.single._

import scala.concurrent.ExecutionContext.Implicits.global

object Main extends App {

  println("hello")

  val start = System.currentTimeMillis()

  implicit val ioContextShift: ContextShift[IO] = IO.contextShift(global)

  // read MNIST dataset
  val trainLabelFileName = "/Users/xiayunsun/Downloads/train-labels-idx1-ubyte"
  val trainImgFileName   = "/Users/xiayunsun/Downloads/train-images-idx3-ubyte"

  val testImgFileName   = "/Users/xiayunsun/Downloads/t10k-images-idx3-ubyte"
  val testLabelFileName = "/Users/xiayunsun/Downloads/t10k-labels-idx1-ubyte"

  def prepTrainData(labelFileName: String,
                    imgFileName: String): fs2.Stream[IO, (Int, Matrix)] = {
    val trainLabels: fs2.Stream[IO, Int] = // todo: must we use IO?
      io.file
        .readAll[IO](path = Paths.get(labelFileName), global, chunkSize = 1024)
        .drop(8) // skip magic number and size
        .map(_.toInt)

    val imgDimension = 28
    val trainImages: fs2.Stream[IO, Matrix] =
      io.file
        .readAll[IO](path = Paths.get(imgFileName), global, chunkSize = 1024) // each img is 784 bytes
        .drop(16) // 8 more bytes for number of rows and number of columns
        .map(java.lang.Byte.toUnsignedInt) // java is big endian
        .chunkN(imgDimension * imgDimension, allowFewer = false)
        .map(_.toVector)
        .map(v => Matrix.fromVector(v.map(_.toDouble), imgDimension, imgDimension))

    //  val trainData: fs2.Stream[IO, (Int, Matrix2D[Int])] = trainLabels.zip(trainImages)

    // optional: sanity check
    //  Preprocessor.sanityCheck(trainLabels, trainImages)

    // proprocess: x /= 255. y: binary classifier on digit 0
    val trainImagesPreprocessed: fs2.Stream[IO, Matrix] =
      trainImages.map(matrix => Matrix(matrix.m.map(_.map(_ / 255.0))))
    val trainLabelsPreprocessed: fs2.Stream[IO, Int] = trainLabels.map(i => if (i == 0) 1 else 0)

    trainLabelsPreprocessed.zip(trainImagesPreprocessed)
  }

  val trainData = prepTrainData(trainLabelFileName, trainImgFileName)
  val testData  = prepTrainData(testLabelFileName, testImgFileName)

  // shuffle training data: todo

  // single neuron: W = 784x1, with sigmoid as activation function
  // todo: can we get rid of var?
  val initialWeights      = Vector.tabulate(784)(_ => 0.01)
  val initialBias: Double = 0

  case class Parameters(
      w1: Matrix,
      b1: Matrix,
      w2: Matrix,
      b2: Scalar
  )

  val initParameters = Parameters(
    w1 = Matrix(Vector.tabulate(64, 784)((x, y) => 0.01)),
    b1 = Matrix(Vector.tabulate(64, 1)((x, y) => 0)),
    w2 = Matrix(Vector.tabulate(1, 64)((x, y) => 0.01)),
    b2 = Scalar(0)
  )

  type Param = (Vector[Double], Double)

  val network: NeuralNetwork = new LogisticRegression

  val learningRate = 1
  val batchSize    = 1000

  val trainSink: (Param, Chunk[(Int, Matrix)]) => Param = {
    case ((weights, bias), chk) =>
      val ys     = chk.map(_._1)
      val yHats  = chk.map { case (y, img) => network.forwardProp(weights, bias, img) }
      val losses = yHats.zip(ys).map { case (yHat, y) => network.loss(yHat, y) }

      val avgLoss = losses.toVector.sum / chk.size
      println(s"avg loss for chunk: $avgLoss")

      val gradients: Chunk[(Vector[Double], Double)] =
        yHats.zip(chk).map { case (yHat, (y, img)) => network.backProp(yHat, y, img) }

      val avgBiasGradient = gradients.map(_._2).toVector.sum / chk.size

      val sumWeightsGradient = gradients
        .map(_._1)
        .toVector
        .reduce[Vector[Double]] { case (v1, v2) => v1.zip(v2).map(t => t._1 + t._2) }

      val avgWeightsGradient = sumWeightsGradient.map(_ / chk.size)

      (
        weights.zip(avgWeightsGradient).map { case (w, g) => w - learningRate * g },
        bias - learningRate * avgBiasGradient
      )

  }

  val graph = for {
    _   <- BinaryStateNode(MatMul)
    _   <- BinaryStateNode(Add)
    ans <- BinaryStateNode(CrossEntropy)
  } yield ans

  val graphWithHiddenLayer = for {
    _   <- BinaryStateNode(MatMul)
    _   <- BinaryStateNode(Add)
    _   <- SingleStateNode(Sigmoid)
    _   <- BinaryStateNode(RevMatMul)
    _   <- BinaryStateNode(Add)
    ans <- BinaryStateNode(CrossEntropy)
  } yield ans

  val trainSinkWithGraph: (Param, Chunk[(Int, Matrix)]) => Param = {
    case ((weights, bias), chk) =>
      val w = Node("w", Matrix(Vector(weights)), Ident)
      val b = Node("b", Scalar(bias), Ident)

      val a =
        chk.map {
          case (_y, img) =>
            val x    = Node("x", Matrix(img.m.flatten.map(Vector(_))), Ident)
            val y    = Node("y", Scalar(_y), Ident)
            val args = List(w, x, b, y)

            val init = (args, emptyGraph(args))

            val ((nodes, g), loss) = graph.run(init).value
            require(nodes.size == 1)
            val gradients = g.backProp(List(w, b), nodes.head)

            (gradients("w").asInstanceOf[Matrix].m,
             gradients("b").asInstanceOf[Scalar].v,
             loss.asInstanceOf[Scalar].v)
        }

      val wGradients = a.map(_._1)
      val bGradients = a.map(_._2)
      val losses     = a.map(_._3)

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

  val trainSinkWithHiddenLayer: (Parameters, Chunk[(Int, Matrix)]) => Parameters = {
    case (parameters, chk) =>
      val w1 = Node("w1", parameters.w1, Ident) // 64x784
      val b1 = Node("b1", parameters.b1, Ident) // 64x1
      val w2 = Node("w2", parameters.w2, Ident) // 1x64
      val b2 = Node("b2", parameters.b2, Ident) // 1

      val a =
        chk.map {
          case (_y, img) =>
            val x    = Node("x", Matrix(img.m.flatten.map(Vector(_))), Ident)
            val y    = Node("y", Scalar(_y), Ident)
            val args = List(w1, x, b1, w2, b2, y)

            val init = (args, emptyGraph(args))

            val ((nodes, g), loss) = graphWithHiddenLayer.run(init).value
            require(nodes.size == 1)
            val gradients = g.backProp(List(w1, b1, w2, b2), nodes.head)

            (gradients("w1").asInstanceOf[Matrix].m,
             gradients("b1").asInstanceOf[Matrix].m,
             gradients("w2").asInstanceOf[Matrix].m,
             gradients("b2").asInstanceOf[Scalar].v,
             loss.asInstanceOf[Scalar].v)
        }

      val w1Gradients: Chunk[Vector[Vector[Double]]] = a.map(_._1) // 64x784
      val b1Gradients: Chunk[Vector[Vector[Double]]] = a.map(_._2) // 64x1
      val w2Gradients: Chunk[Vector[Vector[Double]]] = a.map(_._3) // 1x64
      val b2Gradients: Chunk[Double]                 = a.map(_._4)
      val losses: Chunk[Double]                      = a.map(_._5)

      val avgLoss = losses.toVector.sum / chk.size
      println(s"avg loss: $avgLoss")

      val avgB2Gradient = b2Gradients.toVector.sum / chk.size

      val sumW2Gradients: Vector[Vector[Double]] =
        w2Gradients.toVector.reduce[Vector[Vector[Double]]] {
          case (v1, v2) => Matrix(v1).add(Matrix(v2)).m
        }
      val avgW2Gradient = sumW2Gradients.map(_.map(_ / chk.size * learningRate))

      val sumW1Gradients: Vector[Vector[Double]] =
        w1Gradients.toVector.reduce[Vector[Vector[Double]]] {
          case (v1, v2) => Matrix(v1).add(Matrix(v2)).m
        }
      val avgW1Gradient = sumW1Gradients.map(_.map(_ / chk.size * learningRate))

      val sumB1Gradients: Vector[Vector[Double]] =
        b1Gradients.toVector.reduce[Vector[Vector[Double]]] {
          case (v1, v2) => Matrix(v1).add(Matrix(v2)).m
        }
      val avgB1Gradient = sumB1Gradients.map(_.map(_ / chk.size * learningRate))

      Parameters(
        w1 = parameters.w1.minus(Matrix(avgW1Gradient)),
        b1 = parameters.b1.minus(Matrix(avgB1Gradient)),
        w2 = parameters.w2.minus(Matrix(avgW2Gradient)),
        b2 = Scalar(parameters.b2.v - learningRate * avgB2Gradient)
      )
  }

  val (trainedWeights, trainedBias) =
    trainData
      .chunkN(batchSize)
      .fold((initialWeights, initialBias))(trainSinkWithGraph)
      .compile
      .toVector
      .unsafeRunSync()
      .head

//  val trainedParameters:Parameters =
//    trainData.take(10000).chunkN(batchSize).fold(initParameters)(trainSinkWithHiddenLayer).compile.toVector.unsafeRunSync()
//      .head

  // training error for logistic regression
  print("check prediction...")

  val prediction =
    testData
      .map {
        case (y, img) => {
          val yHat       = network.forwardProp(trainedWeights, trainedBias, img)
          val prediction = if (yHat > 0.5) 1 else 0
          y == prediction
        }
      }
      .compile
      .toVector
      .unsafeRunSync()

  println(s"correct prediction: ${prediction.count(t => t) / prediction.size.toDouble}")

  // prediction for hidden layer
//  val predictionGraph = for {
//    _ <- BinaryStateNode(MatMul)
//    _ <- BinaryStateNode(Add)
//    _ <- SingleStateNode(Sigmoid)
//    _ <- BinaryStateNode(RevMatMul)
//    _ <- BinaryStateNode(Add)
//    ans <- SingleStateNode(Sigmoid)
//  } yield ans
//
//  println("now prediction...")
//  val prediction = trainData.map{case (_y, img) => {
//    val w1 = Node("w1", trainedParameters.w1, Ident) // 64x784
//    val b1 = Node("b1", trainedParameters.b1, Ident) // 64x1
//    val w2 = Node("w2", trainedParameters.w2, Ident) // 1x64
//    val b2 = Node("b2", trainedParameters.b2, Ident) // 1
//
//    val x = Node("x", Matrix(img.flatten.map(Vector(_))), Ident)
//    val y = Node("y", Scalar(_y), Ident)
//    val args = List(w1, x, b1, w2, b2, y)
//
//    val init = (args, emptyGraph(args))
//
//    val (_, yHat) = predictionGraph.run(init).value
//
//    val pred = if (yHat.asInstanceOf[Scalar].v > 0.5) 1 else 0
//    _y == pred
//
//  }}.compile.toVector.unsafeRunSync()
//
//  println(s"correct prediction: ${prediction.count(t => t) / prediction.size.toDouble}")

  println(s"elapsed: ${(System.currentTimeMillis() - start) / 1000.0} seconds")

}
