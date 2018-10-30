object SquareMatrix {

  type Matrix2D = Vector[Vector[Int]] // row-oriented

  implicit class EnrichedVector(v:Vector[Int]) {
    def toMatrix2D(rows:Int, cols:Int):Matrix2D =
      Vector.tabulate(rows, cols){(i,j) => {
        v(cols*i + j)
      }}
  }

}
