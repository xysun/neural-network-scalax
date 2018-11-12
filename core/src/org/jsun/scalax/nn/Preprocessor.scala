package org.jsun.scalax.nn

import java.io.{File, PrintWriter}
import java.nio.file.Paths

import cats.effect.{IO, ContextShift}
import fs2.io
import org.jsun.scalax.nn.datatypes.Matrix

import scala.concurrent.ExecutionContext.Implicits.global

object Preprocessor {

  implicit val ioContextShift: ContextShift[IO] = IO.contextShift(global)

  def prepTrainData(labelFileName: String, imgFileName: String): fs2.Stream[IO, (Int, Matrix)] = {
    val labels: fs2.Stream[IO, Int] = // todo: must we use IO? well we do need a Sync type
      io.file
        .readAll[IO](path = Paths.get(labelFileName), global, chunkSize = 1024)
        .drop(8) // skip magic number and size
        .map(_.toInt)

    val imgDimension = 28
    val images: fs2.Stream[IO, Matrix] =
      io.file
        .readAll[IO](path = Paths.get(imgFileName), global, chunkSize = 1024) // each img is 784 bytes
        .drop(16) // 8 more bytes for number of rows and number of columns
        .map(java.lang.Byte.toUnsignedInt) // java is big endian
        .chunkN(imgDimension * imgDimension, allowFewer = false)
        .map(_.toVector)
        .map(v => Matrix.fromVector(v.map(_.toDouble), imgDimension, imgDimension))

    // optional: sanity check
    //  sanityCheck(trainLabels, trainImages)

    // proprocess: x /= 255. y: binary classifier on digit 0
    val imagesPreprocessed: fs2.Stream[IO, Matrix] =
      images.map(matrix => Matrix(matrix.m.map(_.map(_ / 255.0))))
    val labelsPreprocessed: fs2.Stream[IO, Int] = labels.map(i => if (i == 0) 1 else 0)

    labelsPreprocessed.zip(imagesPreprocessed)
  }

  def sanityCheck(labels: fs2.Stream[IO, Int], images: fs2.Stream[IO, Matrix]): Unit = {

    println("[sanity check] checking both train labels and images have length 60k...")
    assert(labels.compile.toVector.map(_.size).unsafeRunSync() == 60000)
    assert(images.compile.toVector.map(_.size).unsafeRunSync() == 60000)

    println("[sanity check] checking all labels and pixels are sane...")
    labels.compile.toVector.unsafeRunSync().forall(i => i >= 0 && i <= 10)
    images.compile.toVector
      .unsafeRunSync()
      .map(_.m.flatten)
      .flatten
      .forall(_ >= 0)

    println("[sanity check] export first img so we can view in jupyter...")
    val img: Matrix = images.take(1).compile.toVector.unsafeRunSync()(0)
    val pw          = new PrintWriter(new File("img1.csv"))
    pw.write(img.m.flatten.mkString("\n"))
    pw.close()
  }

}
