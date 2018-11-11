package org.jsun.scalax.nn.graph

import org.jsun.scalax.nn.datatypes._
import org.jsun.scalax.nn.graph.operations.Ident
import org.jsun.scalax.nn.graph.operations.binary.{Add, ScalarMulti}
import org.scalatest.FunSuite

class GraphTest extends FunSuite{

  test("build graph in forward propagation"){

    val n1 = Node("n1", Scalar(1), Ident)
    val n2 = Node("n2", Scalar(2), Ident)

    val args = List(n1, n2)

    val initState = (args, emptyGraph(args))

    val g = for {
      ans <- BinaryStateNode(Add)
    } yield ans

    val ((finalNodes, finalGraph), ans) = g.run(initState).value

    assert(ans == Scalar(3))
    assert(finalGraph.nodes.size == 3)
    assert(finalGraph.consumersMap.keySet == Set("n1", "n2"))
    assert(finalGraph.inputsMap.size == 1)
    assert(finalGraph.inputsMap.head._2.map(_.name).toSet == Set("n1", "n2"))

    val gradients = finalGraph.backProp(args, finalNodes.head)
    assert(gradients == Map("n1" -> Scalar(1.0), "n2" -> Scalar(1.0)))

  }

  test("bprop2") {
    val n1 = Node("n1", Scalar(1), Ident)
    val n2 = Node("n2", Scalar(2), Ident)
    val n3 = Node("n3", Scalar(4), Ident)

    val args = List(n1,n2,n3)

    val init = (args, emptyGraph(args))

    val compute = for {
      _ <- BinaryStateNode(Add)
      ans <- BinaryStateNode(ScalarMulti)
    } yield ans

    val ((finalNodes, finalGraph), ans) = compute.run(init).value

    val gradients = finalGraph.backProp(List(n1, n2, n3), finalNodes.head)
    assert(gradients == Map("n1" -> Scalar(4.0), "n2" -> Scalar(4.0), "n3" -> Scalar(3.0)))
  }
}
