package org.jsun.scalax.nn.models

import fs2.Chunk
import org.jsun.scalax.nn.datatypes.Matrix

import scala.math.{ E, log, pow }

class LogisticRegression extends Model {

  /*
  parameters: weights, bias
   */

  override type Param = (Vector[Double], Double)

  override val initialWeights: (Vector[Double], Double) = (
    Vector.tabulate(784)(_ => 0.01),
    0.0
  )

  private val learningRate = 1

  override def trainBatch
    : ((Vector[Double], Double), Chunk[(Int, Matrix)]) => (Vector[Double], Double) = {
    case ((weights, bias), chk) =>
      val ys     = chk.map(_._1)
      val yHats  = chk.map { case (y, img) => forwardProp(weights, bias, img) }
      val losses = yHats.zip(ys).map { case (yHat, y) => loss(yHat, y) }

      val avgLoss = losses.toVector.sum / chk.size
      println(s"avg loss for chunk: $avgLoss")

      val gradients: Chunk[(Vector[Double], Double)] =
        yHats.zip(chk).map { case (yHat, (y, img)) => backProp(yHat, y, img) }

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

  override def predict(trained: (Vector[Double], Double), image: Matrix): Double =
    forwardProp(trained._1, trained._2, image)

  private def forwardProp(weights: Vector[Double], bias: Double, image: Matrix): Double = {

    val z = weights.zip(image.m.flatten).map { case (w, x) => w * x }.sum + bias
    1 / (1 + pow(E, -z))

  }

  private def backProp(yHat: Double, y: Int, image: Matrix): (Vector[Double], Double) = {

    val z = -log(1 / yHat - 1)

    // dLoss/dZ
    val a  = (1 - 2 * y) * z
    val d1 = 1 / (1 + pow(E, a)) * pow(E, a) * (1 - 2 * y)

    // dZ/dyHat
    val d2 = 1 / (yHat * (1 - yHat))

    // dYHat/dz
    val d3 = yHat * (1 - yHat)

    // total
    val d = d1 * d2 * d3

    val weightsGradient = image.m.flatten.map(x => d * x)

    val biasGradient = d

    (weightsGradient, biasGradient)
  }

  private def loss(yHat: Double, y: Int): Double = {
    val z = -log(1 / yHat - 1)

    log(1 + pow(E, (1 - 2 * y) * z))
  }

}
