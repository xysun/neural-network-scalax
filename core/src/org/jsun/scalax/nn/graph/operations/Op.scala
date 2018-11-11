package org.jsun.scalax.nn.graph.operations

import org.jsun.scalax.nn.datatypes.Tensor
import org.jsun.scalax.nn.graph.Node

sealed trait Op {
  def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor
}

trait BinaryOp extends Op {
  def f(n1: Node, n2: Node): Tensor
}

trait SingleOp extends Op {
  def f(n: Node): Tensor
}

case object Ident extends Op {
  override def bprop(inputs: List[Node], x: Node, g: Tensor) = g
}
