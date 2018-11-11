package org.jsun.scalax.nn.graph.operations.binary

import org.jsun.scalax.nn.datatypes.{ Matrix, Scalar, Tensor }
import org.jsun.scalax.nn.graph.Node
import org.jsun.scalax.nn.graph.operations.BinaryOp

case object Add extends BinaryOp {
  override def bprop(inputs: List[Node], x: Node, g: Tensor) = g

  override def f(n1: Node, n2: Node): Tensor = (n1.v, n2.v) match {
    case (Scalar(v1), Scalar(v2)) => Scalar(v1 + v2)
    case (m1: Matrix, m2: Matrix) => m1.add(m2)
  }
}
