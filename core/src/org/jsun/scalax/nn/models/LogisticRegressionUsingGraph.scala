package org.jsun.scalax.nn.models
import fs2.Chunk
import org.jsun.scalax.nn.datatypes.{ Matrix, Scalar }
import org.jsun.scalax.nn.graph.operations.Ident
import org.jsun.scalax.nn.graph.operations.binary.{ Add, CrossEntropy, MatMul }
import org.jsun.scalax.nn.graph.operations.single.Sigmoid
import org.jsun.scalax.nn.graph.{ BinaryStep, Node, Unarystep, emptyGraph }

class LogisticRegressionUsingGraph extends Model {

  override type Param = (Vector[Double], Double)

  override val initialWeights: (Vector[Double], Double) = (
    Vector.tabulate(784)(_ => 0.01),
    0.0
  )

  private val learningRate = 1

  private val graph = for {
    _   <- BinaryStep(MatMul)
    _   <- BinaryStep(Add)
    ans <- BinaryStep(CrossEntropy)
  } yield ans

  private val predictGraph = for {
    _   <- BinaryStep(MatMul)
    _   <- BinaryStep(Add)
    ans <- Unarystep(Sigmoid)
  } yield ans

  override def trainBatch
    : ((Vector[Double], Double), Chunk[(Int, Matrix)]) => (Vector[Double], Double) = {
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

  override def predict(trained: (Vector[Double], Double), image: Matrix): Double = {
    // todo: think we can cache these nodes
    val w    = Node("w", Matrix(Vector(trained._1)), Ident)
    val b    = Node("b", Scalar(trained._2), Ident)
    val x    = Node("x", Matrix(image.m.flatten.map(Vector(_))), Ident)
    val args = List(w, x, b)
    val init = (args, emptyGraph(args))

    val (_, loss) = predictGraph.run(init).value
    loss.asInstanceOf[Scalar].v
  }

}
