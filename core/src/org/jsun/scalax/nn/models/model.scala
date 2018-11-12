package org.jsun.scalax.nn.models

import fs2.Chunk
import org.jsun.scalax.nn.datatypes.Matrix

trait Model {
  type Param
  val initialWeights: Param
  def trainBatch: (Param, Chunk[(Int, Matrix)]) => Param
  def predict(trained: Param, image: Matrix): Double
}
