package org.jsun.scalax.nn

import java.util.UUID

import cats.data.State
import org.jsun.scalax.nn.datatypes.Tensor
import org.jsun.scalax.nn.graph.operations.{ BinaryOp, SingleOp }

package object graph {
  // todo: can we get rid of BinaryState and SingleState?
  def BinaryStep(op: BinaryOp) = State[(List[Node], Graph), Tensor] {
    case (n1 :: n2 :: tail, g) => {
      val ans      = op.f(n1, n2)
      val nodeName = UUID.randomUUID().toString
      val newNode  = Node(nodeName, ans, op)

      val newGraph = new Graph {
        override val nodes: List[Node] = newNode :: g.nodes
        override val consumersMap: Map[String, List[Node]] = {
          g.consumersMap + ((n1.name, newNode :: g.consumersMap.getOrElse(n1.name, Nil))) +
          ((n2.name, newNode :: g.consumersMap.getOrElse(n2.name, Nil)))
        }
        override val inputsMap: Map[String, List[Node]] = g.inputsMap + ((newNode.name,
                                                                          List(n1, n2)))
      }

      ((newNode :: tail, newGraph), ans)
    }
  }

  def Unarystep(op: SingleOp) = State[(List[Node], Graph), Tensor] {
    case (n1 :: tail, g) => {
      val ans      = op.f(n1)
      val nodeName = UUID.randomUUID().toString
      val newNode  = Node(nodeName, ans, op)

      val newGraph = new Graph {
        override val nodes: List[Node] = newNode :: g.nodes
        override val consumersMap: Map[String, List[Node]] = {
          g.consumersMap + ((n1.name, newNode :: g.consumersMap.getOrElse(n1.name, Nil)))
        }
        override val inputsMap: Map[String, List[Node]] = g.inputsMap + ((newNode.name, List(n1)))
      }

      ((newNode :: tail, newGraph), ans)
    }
  }

  def emptyGraph(initNodes: List[Node]) = new Graph {
    override val nodes: List[Node]                     = initNodes
    override val consumersMap: Map[String, List[Node]] = Map.empty
    override val inputsMap: Map[String, List[Node]]    = Map.empty
  }
}
