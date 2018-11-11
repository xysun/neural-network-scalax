package org.jsun.scalax.nn.graph

import org.jsun.scalax.nn.datatypes.{ Scalar, Tensor }

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

    def buildGrad(v: Node, gradTable: Map[String, Tensor]): Map[String, Tensor] =
      if (gradTable contains v.name) gradTable
      else {
        val g = consumersMap(v.name)
          .map(c => {
            val d = buildGrad(c, gradTable)(c.name)
            c.op.bprop(inputsMap(c.name), v, d)
          })
          .reduce((t1, t2) => Tensor.add(t1, t2))

        gradTable + ((v.name, g))
      }

    targets
      .foldRight(Map[String, Tensor](z.name -> Scalar(1))) {
        case (t, gradTable) => buildGrad(t, gradTable)
      }
      .filterKeys(targets.map(_.name) contains _)

  }

}
