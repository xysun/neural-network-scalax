import java.io.{File, PrintWriter}
import java.nio.ByteOrder
import java.nio.file.{Files, Paths}

import cats.effect.{ContextShift, ExitCode, IO}
import fs2.{Pipe, Pull, io, text}

import scala.concurrent.ExecutionContext.Implicits.global
import Matrix2DSyntax._

import scala.io.Source

object Main extends App {

  println("hello")

  // read MNIST dataset

  implicit val ioContextShift: ContextShift[IO] = IO.contextShift(global)

  val trainLabelFileName = "/Users/xiayunsun/Downloads/train-labels-idx1-ubyte"
  val trainImgFileName = "/Users/xiayunsun/Downloads/train-images-idx3-ubyte"

  val trainLabels:fs2.Stream[IO, Int] = // todo: must we use IO?
    io.file.readAll[IO](path = Paths.get(trainLabelFileName), global, chunkSize = 1024)
    .drop(8) // skip magic number and size
    .map(_.toInt)

  // confirm train labels has length 60k,

  val l = trainLabels.compile.toVector.map(_.size).unsafeRunSync()
  println(s"total labels: $l") // must be 60k

  // print out first 10 labels
  trainLabels
//    .take(10)
    .compile.toVector
    .unsafeRunSync()
    .forall(i => i >= 0 && i <= 10)

  // load image

  // chunkSize: each image is 28x28=784 bytes
  private val imgDimension = 28
  val trainImages:fs2.Stream[IO, Matrix2D] =
    io.file.readAll[IO](path = Paths.get(trainImgFileName), global, chunkSize = 1024)
    .drop(16) // 8 more bytes for number of rows and number of columns
    .map(java.lang.Byte.toUnsignedInt) // java is big endian
    .chunkN(imgDimension * imgDimension, allowFewer = false)
    .map(_.toVector)
    .map(_.toMatrix2D(rows = imgDimension, cols = imgDimension))

//  val imgCount = trainImages.compile.toVector.map(_.size).unsafeRunSync()
//  println(s"total images: $imgCount") // must be 60k

  // export first img then display in jupyter
  val img:Matrix2D = trainImages.take(1).compile.toVector.unsafeRunSync()(0)
  assert(img.flatten.forall(_ >= 0))

//  val pw = new PrintWriter(new File("img1.csv"))
//  pw.write(img.flatten.mkString("\n"))
//  pw.close()

  val trainData:fs2.Stream[IO, (Int, Matrix2D)] = trainLabels.zip(trainImages)

}
