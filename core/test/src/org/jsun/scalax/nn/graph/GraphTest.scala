package org.jsun.scalax.nn.graph

import org.jsun.scalax.nn.graph.Tensor._
import org.scalatest.FunSuite

class GraphTest extends FunSuite{

  test("meh"){
    val graph = for {
      _ <- Matmul // 11
      _ <- Addition // 16
      ans <- Logit
    } yield ans

    val args = List(
      MeVector(Vector(1,2)),
      MeVector(Vector(3,4)),
      MeDouble(5)
    )

    println(graph.run(args).value)
  }

  test("build graph in forward propagation"){

    val n1 = Node("n1", 1, Ident)
    val n2 = Node("n2", 2, Ident)

    val initState = (List(n1,n2), emptyGraph(List(n1, n2)))

    val g = for {
      ans <- BinaryStateNode(ScalarAddOp)
    } yield ans

    val ((finalNodes, finalGraph), ans) = g.run(initState).value

    assert(ans == 3)
    assert(finalGraph.nodes.size == 3)
    assert(finalGraph.consumersMap.keySet == Set("n1", "n2"))
    assert(finalGraph.inputsMap.size == 1)
    assert(finalGraph.inputsMap.head._2.map(_.name).toSet == Set("n1", "n2"))

    val gradients = backProp(List(n1, n2), finalGraph, finalNodes.head)
    assert(gradients == Map("n1" -> 1.0, "n2" -> 1.0))

  }

  test("bprop2") {
    val n1 = Node("n1", 1, Ident)
    val n2 = Node("n2", 2, Ident)
    val n3 = Node("n3", 4, Ident)

    val init = (List(n1,n2,n3), emptyGraph(List(n1,n2,n3)))

    val compute = for {
      _ <- BinaryStateNode(ScalarAddOp)
      ans <- BinaryStateNode(ScalarMultiOp)
    } yield ans

    val ((finalNodes, finalGraph), ans) = compute.run(init).value

    val gradients = backProp(List(n1, n2, n3), finalGraph, finalNodes.head)
    assert(gradients == Map("n1" -> 4.0, "n2" -> 4.0, "n3" -> 3.0))
  }
}
