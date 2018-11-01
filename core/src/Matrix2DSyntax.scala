object Matrix2DSyntax {

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

}
