package org.jsun.scalax.nn

import cats.data.State

package object graph {

  sealed trait Op

  case class Node(op: Op, parent: Node, inputs:Seq[Double])

  type Graph = State[Node, Double]

  sealed trait Numeric // union type in dotty :/

  case class MeDouble(v:Double) extends Numeric
  case class MeVector(v:Vector[Double]) extends Numeric

  val Addition = State[List[Numeric], Double]{
    case MeDouble(a) :: MeDouble(b) :: tail =>
      val ans = a+b
      (MeDouble(ans) :: tail, ans)
  }

  val Logit = State[List[Numeric], Double]{
    case MeDouble(h)::t =>
      val ans = 1 / (1 + math.pow(math.E, -h))
      (MeDouble(ans)::t, ans)
  }

  val Matmul = State[List[Numeric], Double] {
    case MeVector(a) :: MeVector(b) :: tail =>
      val ans = a.zip(b).map{case (x,y) => x*y}.sum
      (MeDouble(ans)::tail, ans)
  }



}
