import java.nio.file.Paths

import Matrix2DSyntax._
import cats.effect.{ ContextShift, IO }
import fs2.io

import scala.concurrent.ExecutionContext.Implicits.global

object Main extends App {

  println("hello")

  implicit val ioContextShift: ContextShift[IO] = IO.contextShift(global)

  // read MNIST dataset
  val trainLabelFileName = "/Users/xiayunsun/Downloads/train-labels-idx1-ubyte"
  val trainImgFileName   = "/Users/xiayunsun/Downloads/train-images-idx3-ubyte"

  val trainLabels: fs2.Stream[IO, Int] = // todo: must we use IO?
    io.file
      .readAll[IO](path = Paths.get(trainLabelFileName), global, chunkSize = 1024)
      .drop(8) // skip magic number and size
      .map(_.toInt)

  private val imgDimension = 28
  val trainImages: fs2.Stream[IO, Matrix2D] =
    io.file
      .readAll[IO](path = Paths.get(trainImgFileName), global, chunkSize = 1024) // each img is 784 bytes
      .drop(16) // 8 more bytes for number of rows and number of columns
      .map(java.lang.Byte.toUnsignedInt) // java is big endian
      .chunkN(imgDimension * imgDimension, allowFewer = false)
      .map(_.toVector)
      .map(_.toMatrix2D(rows = imgDimension, cols = imgDimension))

  val trainData: fs2.Stream[IO, (Int, Matrix2D)] = trainLabels.zip(trainImages)

  // optional: sanity check
//  Preprocessor.sanityCheck(trainLabels, trainImages)

}
