package org.jsun.scalax.nn.models

import org.jsun.scalax.nn.Matrix2D

trait NeuralNetwork {

  def forwardProp(weights: Vector[Double], bias:Double, image:Matrix2D[Double]):Double

  def backProp(yHat:Double, y:Int, image:Matrix2D[Double]):(Vector[Double], Double)

  def loss(yHat:Double, y:Int):Double

}
