import java.io.{ File, PrintWriter }

import Matrix2DSyntax.Matrix2D
import cats.effect.IO

object Preprocessor {

  def sanityCheck(labels: fs2.Stream[IO, Int], images: fs2.Stream[IO, Matrix2D]): Unit = {

    println("[sanity check] checking both train labels and images have length 60k...")
    assert(labels.compile.toVector.map(_.size).unsafeRunSync() == 60000)
    assert(images.compile.toVector.map(_.size).unsafeRunSync() == 60000)

    println("[sanity check] checking all labels and pixels are sane...")
    labels.compile.toVector.unsafeRunSync().forall(i => i >= 0 && i <= 10)
    images.compile.toVector
      .map(_.flatten)
      .unsafeRunSync()
      .flatten
      .forall(_ >= 0)

    println("[sanity check] export first img so we can view in jupyter...")
    val img: Matrix2D = images.take(1).compile.toVector.unsafeRunSync()(0)
    val pw            = new PrintWriter(new File("img1.csv"))
    pw.write(img.flatten.mkString("\n"))
    pw.close()
  }

}
