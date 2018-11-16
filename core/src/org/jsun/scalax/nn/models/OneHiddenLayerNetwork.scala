package org.jsun.scalax.nn.models

import fs2.Chunk
import org.jsun.scalax.nn.datatypes.{Matrix, Scalar}
import org.jsun.scalax.nn.graph.operations.Ident
import org.jsun.scalax.nn.graph.{BinaryStep, Node, Unarystep, emptyGraph}
import org.jsun.scalax.nn.graph.operations.binary.{Add, CrossEntropy, MatMul, RevMatMul}
import org.jsun.scalax.nn.graph.operations.single.Sigmoid

class OneHiddenLayerNetwork extends Model {

  private val learningRate = 1

  case class Parameters(
                         w1: Matrix,
                         b1: Matrix,
                         w2: Matrix,
                         b2: Scalar
                       )

  override type Param = Parameters

  override val initialWeights: Parameters = Parameters(
    w1 = Matrix(Vector.tabulate(64, 784)((x, y) => 0.01)),
    b1 = Matrix(Vector.tabulate(64, 1)((x, y) => 0)),
    w2 = Matrix(Vector.tabulate(1, 64)((x, y) => 0.01)),
    b2 = Scalar(0)
  )

  private val trainGraph = for {
    _   <- BinaryStep(MatMul)
    _   <- BinaryStep(Add)
    _   <- Unarystep(Sigmoid)
    _   <- BinaryStep(RevMatMul)
    _   <- BinaryStep(Add)
    ans <- BinaryStep(CrossEntropy)
  } yield ans

  private val predictionGraph = for {
    _ <- BinaryStep(MatMul)
    _ <- BinaryStep(Add)
    _ <- Unarystep(Sigmoid)
    _ <- BinaryStep(RevMatMul)
    _ <- BinaryStep(Add)
    ans <- Unarystep(Sigmoid)
  } yield ans

  override def trainBatch: (Parameters, Chunk[(Int, Matrix)]) => Parameters = {
    case (parameters, _chk) =>

      // stupid SGD
      val chk = _chk.take(_chk.size / 10)

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

            val ((nodes, g), loss) = trainGraph.run(init).value
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

  override def predict(trained: Parameters, image: Matrix): Double = {
    val w1 = Node("w1", trained.w1, Ident) // 64x784
    val b1 = Node("b1", trained.b1, Ident) // 64x1
    val w2 = Node("w2", trained.w2, Ident) // 1x64
    val b2 = Node("b2", trained.b2, Ident) // 1

    val x = Node("x", Matrix(image.m.flatten.map(Vector(_))), Ident)
    val args = List(w1, x, b1, w2, b2)

    val init = (args, emptyGraph(args))

    val (_, loss) = predictionGraph.run(init).value

    loss.asInstanceOf[Scalar].v
  }

}
