package org.jsun.scalax.nn.models

import org.jsun.scalax.nn.datatypes.Matrix

import scala.math.{ E, log, pow }

class LogisticRegression extends NeuralNetwork {

  def forwardProp(weights: Vector[Double], bias: Double, image: Matrix): Double = {

    val z = weights.zip(image.m.flatten).map { case (w, x) => w * x }.sum + bias
    1 / (1 + pow(E, -z))

  }

  def backProp(yHat: Double, y: Int, image: Matrix): (Vector[Double], Double) = {

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

  def loss(yHat: Double, y: Int): Double = {
    val z = -log(1 / yHat - 1)

    log(1 + pow(E, (1 - 2 * y) * z))
  }

}
