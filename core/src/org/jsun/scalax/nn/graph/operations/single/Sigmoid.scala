package org.jsun.scalax.nn.graph.operations.single

import org.jsun.scalax.nn.datatypes.{ Matrix, Scalar, Tensor }
import org.jsun.scalax.nn.graph.Node
import org.jsun.scalax.nn.graph.operations.SingleOp

case object Sigmoid extends SingleOp {

  private def forward(v: Double) = 1 / (1 + math.pow(math.E, -1 * v))
  private def backward(v: Double) = {
    val y = forward(v)
    y * (1 - y)
  }

  override def f(n: Node): Tensor = n.v match {
    case Scalar(v) => Scalar(forward(v))
    case Matrix(m) => Matrix(m.map(_.map(forward)))
  }

  override def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor =
    (f(x), g) match {
      case (Scalar(y), Scalar(v)) => Scalar(v * y * (1 - y))
      case (m1: Matrix, m2: Matrix) => {
        require(m1.shape == m2.shape)
        val res = Vector.tabulate(m1.shape._1, m1.shape._2) {
          case (i, j) => {
            val y = m1.m(i)(j)
            val v = m2.m(i)(j)
            v * y * (1 - y)
          }
        }
        Matrix(res)
      }
    }
}
