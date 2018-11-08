package org.jsun.scalax.nn.graph

import java.util.UUID

import cats.data.State

object Tensor {

  trait Tensor

  case class Scalar(v: Double) extends Tensor

  // row x column
  case class Matrix(m: Vector[Vector[Double]]) extends Tensor {
    val shape:(Int, Int) = (m.size, m.head.size)

    def add(other:Matrix):Matrix = {
      require(shape == other.shape)

      val res = Vector.tabulate(shape._1, shape._2){case (i,j) =>
        m(i)(j) + other.m(i)(j)
      }
      Matrix(res)
    }

    def multiply(other:Matrix):Tensor = {
      require(shape._2 == other.shape._1)
      val dimension = (shape._1, other.shape._2)

      val res = Vector.tabulate(dimension._1, dimension._2){case (i,j) => {
        // ith row * jth column
        val rowi = m(i)
        val colj = other.m.map(_.apply(j))
        rowi.zip(colj).map(t => t._1 * t._2).sum
      }}

      if (dimension == (1,1)) Scalar(res(0)(0))
      else Matrix(res)

    }

  }

  object Tensor {
    def add(t1:Tensor, t2:Tensor):Tensor = (t1, t2) match {
      case (Scalar(v1), Scalar(v2)) => Scalar(v1 + v2)
      case (m1:Matrix, m2:Matrix) => m1.add(m2)
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

  case object MatMul extends BinaryOp {
    override def f(n1: Node, n2: Node): Tensor = (n1.v, n2.v) match {
      case (m1:Matrix, m2:Matrix) => m1.multiply(m2)
    }

    override def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor = {
      require(inputs.size == 2)
      val other = inputs.filter(_.name != x.name)
      require(other.size == 1)

      (g, other.head.v) match {
        case (Scalar(v), m:Matrix) => {
          // other must be a vector, then reverse it
          if (m.shape._1 == 1){ // other is 1xn, so we want nx1
            val res = Vector.tabulate(m.shape._2, 1){case (i,j) => v * m.m(j)(i)}
            Matrix(res)
          } else if (m.shape._2 == 1){ // other is nx1, so we want 1xn
            val res = Vector.tabulate(1, m.shape._1){case (i,j) => v * m.m(j)(i)}
            Matrix(res)
          } else {
            throw new Exception(s"the other matrix must be a 1d vector! ${m.shape}")
          }
        }

        case (m1:Matrix, m2:Matrix) => {
          require(m1.shape._2 == 1 && m2.shape._2 == 1)
          val res = Vector.tabulate(m1.shape._1, m2.shape._1) {case (i,j) => {
            m1.m(i)(0) * m2.m(j)(0)
          }}
          Matrix(res)
      }
      }
    }
  }

  case object Add extends BinaryOp {
    override def bprop(inputs: List[Node], x: Node, g: Tensor) = g

    override def f(n1: Node, n2: Node): Tensor = (n1.v, n2.v) match {
      case (Scalar(v1), Scalar(v2)) => Scalar(v1+v2)
      case (m1:Matrix, m2:Matrix) => m1.add(m2)
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

      private def forward(v:Double) = 1 / (1 + math.pow(math.E, -1*v))
      private def backward(v:Double) = {
        val y = forward(v)
        y * (1-y)
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
            val res = Vector.tabulate(m1.shape._1, m1.shape._2) { case (i, j) => {
              val y = m1.m(i)(j)
              val v = m2.m(i)(j)
              v * y * (1 - y)
            }
            }
            Matrix(res)
          }
        }
      }

    case object CrossEntropyLoss extends BinaryOp{
      override def f(n1: Node, n2: Node): Tensor = (n1.v, n2.v) match { // order is important
        case (Scalar(z), Scalar(y)) => Scalar(math.log(1 + math.exp(z * (1-2*y))))
      }

      override def bprop(inputs: List[Node], x: Node, g: Tensor): Tensor = {
        // only dL/dz
        require(inputs.size == 2)
        require(inputs.filterNot(_.name == x.name).size == 1)
        val _y = inputs.filterNot(_.name == x.name).head

        (_y.v, x.v) match {
          case (Scalar(y), Scalar(z)) => {
            val h = z * (1- 2 * y)
            Scalar(1 / (1 + math.exp(h)) * math.exp(h) * (1 - 2 * y))
          }
        }

      }
    }

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
