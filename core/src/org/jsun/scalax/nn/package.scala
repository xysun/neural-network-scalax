package org.jsun.scalax

package object nn {

  type Matrix2D[T] = Vector[Vector[T]] // row-oriented

  implicit class EnrichedMatrix2D[T](v: Matrix2D[T]) {
    def mmap[B](f: T => B): Matrix2D[B] =
      v.map(_.map(f))
  }

  implicit class EnrichedVector[T](v: Vector[T]) {
    def toMatrix2D(rows: Int, cols: Int): Matrix2D[T] =
      Vector.tabulate(rows, cols) { (i, j) =>
      {
        v(cols * i + j)
      }
      }
  }

  implicit class EnrichedVectorVector(v:Vector[Vector[Double]]) {
    def minus(other:Vector[Vector[Double]]): Vector[Vector[Double]] = {
      require(v.size == other.size && v.head.size == other.head.size)
      val dimension = (v.size, v.head.size)
      Vector.tabulate(dimension._1, dimension._2){case (i,j) => {
        v(i)(j) - other(i)(j)
      }}
    }
  }

}
