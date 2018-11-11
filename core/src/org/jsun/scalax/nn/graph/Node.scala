package org.jsun.scalax.nn.graph

import org.jsun.scalax.nn.datatypes.Tensor
import org.jsun.scalax.nn.graph.operations.Op

case class Node(
    name: String,
    v: Tensor,
    op: Op
)
