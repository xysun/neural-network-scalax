package org.jsun.scalax.nn.graph.operations.binary

import org.jsun.scalax.nn.datatypes.{ Matrix, Scalar, Tensor }
import org.jsun.scalax.nn.graph.Node
import org.jsun.scalax.nn.graph.operations.BinaryOp

case object MatMul extends BinaryOp {
  override def f(n1: Node, n2: Node): Tensor = (n1.v, n2.v) match {
    case (m1: Matrix, m2: Matrix) => m1.multiply(m2)
  }

  override def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor = {
    require(inputs.size == 2)
    val other = inputs.filter(_.name != x.name)
    require(other.size == 1)

    (g, other.head.v) match {
      case (Scalar(v), m: Matrix) => {
        // other must be a vector, then reverse it
        if (m.shape._1 == 1) { // other is 1xn, so we want nx1
          val res = Vector.tabulate(m.shape._2, 1) { case (i, j) => v * m.m(j)(i) }
          Matrix(res)
        } else if (m.shape._2 == 1) { // other is nx1, so we want 1xn
          val res = Vector.tabulate(1, m.shape._1) { case (i, j) => v * m.m(j)(i) }
          Matrix(res)
        } else {
          throw new Exception(s"the other matrix must be a 1d vector! ${m.shape}")
        }
      }

      case (m1: Matrix, m2: Matrix) => {
        require(m1.shape._2 == 1 && m2.shape._2 == 1)
        val res = Vector.tabulate(m1.shape._1, m2.shape._1) {
          case (i, j) => {
            m1.m(i)(0) * m2.m(j)(0)
          }
        }
        Matrix(res)
      }
    }
  }
}
