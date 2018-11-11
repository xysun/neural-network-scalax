package org.jsun.scalax.nn.graph.operations.binary

import org.jsun.scalax.nn.datatypes.Tensor
import org.jsun.scalax.nn.graph.Node
import org.jsun.scalax.nn.graph.operations.BinaryOp

case object RevMatMul extends BinaryOp {

  // hack because we require an ordered stack of input nodes
  override def f(n1: Node, n2: Node): Tensor = MatMul.f(n2, n1)

  override def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor =
    MatMul.bprop(inputs, x, g)
}
