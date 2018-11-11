package org.jsun.scalax.nn.graph.operations.binary

import org.jsun.scalax.nn.datatypes.{ Scalar, Tensor }
import org.jsun.scalax.nn.graph.Node
import org.jsun.scalax.nn.graph.operations.BinaryOp

case object CrossEntropy extends BinaryOp {
  override def f(n1: Node, n2: Node): Tensor = (n1.v, n2.v) match { // order is important
    case (Scalar(z), Scalar(y)) => Scalar(math.log(1 + math.exp(z * (1 - 2 * y))))
  }

  override def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor = {
    // only dL/dz
    require(inputs.size == 2)
    require(inputs.filterNot(_.name == x.name).size == 1)
    val _y = inputs.filterNot(_.name == x.name).head

    (_y.v, x.v) match {
      case (Scalar(y), Scalar(z)) => {
        val h = z * (1 - 2 * y)
        Scalar(1 / (1 + math.exp(h)) * math.exp(h) * (1 - 2 * y))
      }
    }

  }
}
