package org.jsun.scalax.nn.datatypes

sealed trait Tensor

case class Scalar(v: Double) extends Tensor

// row x column
case class Matrix(m: Vector[Vector[Double]]) extends Tensor {
  val shape: (Int, Int) = (m.size, m.head.size)

  def add(other: Matrix): Matrix = {
    require(shape == other.shape)

    val res = Vector.tabulate(shape._1, shape._2) {
      case (i, j) =>
        m(i)(j) + other.m(i)(j)
    }
    Matrix(res)
  }

  def minus(other: Matrix): Matrix = {
    require(shape == other.shape)

    val res = Vector.tabulate(shape._1, shape._2) {
      case (i, j) =>
        m(i)(j) - other.m(i)(j)
    }
    Matrix(res)
  }

  def multiply(other: Matrix): Tensor = {
    require(shape._2 == other.shape._1)
    val dimension = (shape._1, other.shape._2)

    val res = Vector.tabulate(dimension._1, dimension._2) {
      case (i, j) => {
        // ith row * jth column
        val rowi = m(i)
        val colj = other.m.map(_.apply(j))
        rowi.zip(colj).map(t => t._1 * t._2).sum
      }
    }

    if (dimension == (1, 1)) Scalar(res(0)(0))
    else Matrix(res)

  }

}

object Matrix {
  def fromVector(v: Vector[Double], rows: Int, cols: Int): Matrix =
    Matrix(Vector.tabulate(rows, cols) { (i, j) =>
      v(cols * i + j)
    })
}

object Tensor {
  def add(t1: Tensor, t2: Tensor): Tensor = (t1, t2) match {
    case (Scalar(v1), Scalar(v2)) => Scalar(v1 + v2)
    case (m1: Matrix, m2: Matrix) => m1.add(m2)
    case _                        => throw new Exception(s"Incompatible operands: $t1, $t2")
  }
}
