package org.jsun.scalax.nn.models

import org.jsun.scalax.nn.datatypes.Matrix

trait NeuralNetwork {

  def forwardProp(weights: Vector[Double], bias: Double, image: Matrix): Double

  def backProp(yHat: Double, y: Int, image: Matrix): (Vector[Double], Double)

  def loss(yHat: Double, y: Int): Double

}
