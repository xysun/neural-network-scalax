package org.jsun.scalax.nn.graph.operations.binary

import org.jsun.scalax.nn.datatypes.{ Scalar, Tensor }
import org.jsun.scalax.nn.graph.Node
import org.jsun.scalax.nn.graph.operations.BinaryOp

case object ScalarMulti extends BinaryOp {
  override def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor = {
    require(inputs.size == 2)
    val other = inputs.filter(_.name != x.name)
    require(other.size == 1)

    (g, other.head.v) match {
      case (Scalar(v1), Scalar(v2)) => Scalar(v1 * v2)
    }
  }

  override def f(n1: Node, n2: Node): Tensor = (n1.v, n2.v) match {
    case (Scalar(v1), Scalar(v2)) => Scalar(v1 * v2)
  }
}
