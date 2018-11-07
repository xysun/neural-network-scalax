package org.jsun.scalax.nn.graph

import java.util.UUID

import cats.data.State

object Tensor {

  trait Tensor

  case class Scalar(v: Double) extends Tensor

  case class TVector(v: Vector[Double]) extends Tensor

  object Tensor {
    def add(t1:Tensor, t2:Tensor):Tensor = (t1, t2) match {
      case (Scalar(v1), Scalar(v2)) => Scalar(v1 + v2)
      case (TVector(v1), TVector(v2)) => TVector(v1.zip(v2).map(t => t._1 + t._2))
    }
  }

  trait Op {
    def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor
  }

  trait BinaryOp extends Op {
    def f(n1:Node, n2:Node):Tensor
  }

  trait SingleOp extends Op {
    def f(n:Node):Tensor
  }

  //  case class CrossProduct extends Op

  case object Add extends BinaryOp {
    override def bprop(inputs: List[Node], x: Node, g: Tensor) = g

    override def f(n1: Node, n2: Node): Tensor = (n1.v, n2.v) match {
      case (Scalar(v1), Scalar(v2)) => Scalar(v1+v2)
      case (TVector(v1), TVector(v2)) => TVector(v1.zip(v2).map(t => t._1 + t._2))
    }
  }

  case object ScalarMultiOp extends BinaryOp {
    override def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor = {
      require(inputs.size == 2)
      val other = inputs.filter(_.name != x.name)
      require(other.size == 1)

      (g, other.head.v) match {
        case (Scalar(v1), Scalar(v2)) => Scalar(v1 * v2)
      }
    }

    override def f(n1: Node, n2: Node): Tensor = (n1.v, n2.v) match {
      case (Scalar(v1), Scalar(v2)) => Scalar(v1*v2)
    }
  }

    case object Sigmoid extends SingleOp{
      override def f(n: Node): Scalar = n.v match {
        case Scalar(v) => Scalar(1 / (1 + math.pow(math.E, -1 * v)))
      }

      override def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor =  g match {
        case Scalar(v) =>
          val y = f(x)
          Scalar(v * y.v * (1-y.v))
      }
    }

  //  case class CrossEntropyLoss extends Op

  case object Ident extends Op {
    override def bprop(inputs: List[Node], x: Node, g: Tensor) = g
  }

  trait Graph { // this will be updated in forwardProp via state monad (state = graph)
    // i think a complete graph can also be automatically constructed during forward prop
    /*
    so a for {
    _ <- CrossProduct
    _ <- ScalarAdd
    ans <- Sigmoid
    } yield ans
    construct a graph of 5 nodes (state), the loss is the computed value
     */
    val nodes: List[Node]
    val consumersMap: Map[String, List[Node]]
    val inputsMap: Map[String, List[Node]]

    def backProp(targets: List[Node], z: Node) = {

      def buildGrad(v: Node, gradTable: Map[String, Tensor]): Map[String, Tensor] = {
        if (gradTable contains v.name) gradTable
        else {
          val g = consumersMap(v.name).map(c => {
            val d = buildGrad(c, gradTable)(c.name)
            c.op.bprop(inputsMap(c.name), v, d)
          }).reduce((t1, t2) => Tensor.add(t1, t2))

          gradTable + ((v.name, g))
        }
      }

      targets
        .foldRight(Map[String, Tensor](z.name -> Scalar(1))){
          case (t, gradTable) => buildGrad(t, gradTable)
        }
        .filterKeys(targets.map(_.name) contains _)

    }
  }

  def emptyGraph(initNodes:List[Node]) = new Graph {
    override val nodes: List[Node] = initNodes
    override val consumersMap: Map[String, List[Node]] = Map.empty
    override val inputsMap: Map[String, List[Node]] = Map.empty
  }

  def BinaryStateNode(op:BinaryOp) = State[(List[Node], Graph), Tensor]{
    case (n1 :: n2 :: tail, g) => {
      val ans = op.f(n1, n2)
      val nodeName = UUID.randomUUID().toString
      val newNode = Node(nodeName, ans, op)

      val newGraph = new Graph {
        override val nodes: List[Node] = newNode :: g.nodes
        override val consumersMap: Map[String, List[Node]] = {
          g.consumersMap + ((n1.name, newNode :: g.consumersMap.getOrElse(n1.name, Nil))) +
            ((n2.name, newNode :: g.consumersMap.getOrElse(n2.name, Nil)))
        }
        override val inputsMap: Map[String, List[Node]] = g.inputsMap + ((newNode.name, List(n1, n2)))
      }

      ((newNode :: tail, newGraph), ans)
    }
  }

  def SingleStateNode(op:SingleOp) = State[(List[Node], Graph), Tensor]{
    case (n1 :: tail, g) => {
      val ans = op.f(n1)
      val nodeName = UUID.randomUUID().toString
      val newNode = Node(nodeName, ans, op)

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

  case class Node(
                   name: String,
                   v: Tensor,
                   op: Op
                 )




}
