package org.jsun.scalax.nn.graph

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

}
