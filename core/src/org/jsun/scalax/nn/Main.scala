package org.jsun.scalax.nn

import org.jsun.scalax.nn.models.{LogisticRegression, LogisticRegressionUsingGraph, Model, OneHiddenLayerNetwork}

object Main extends App {

  /*
  mill core.run model=$model
  model:
  - logicticregression
  - logisticregression-graph
  - onehiddenlayer
   */
  val modelName = args.headOption.map(_.split("=").apply(1)).getOrElse("logisticregression")

  require(
    Set(
      "logisticregression",
      "logisticregression-graph",
      "onehiddenlayer"
    ) contains modelName
  )

  println(s"model = $modelName")

  val start = System.currentTimeMillis()

  // read MNIST dataset
  val trainLabelFile = "/Users/xiayunsun/Downloads/train-labels-idx1-ubyte"
  val trainImgFile   = "/Users/xiayunsun/Downloads/train-images-idx3-ubyte"

  val testImgFile   = "/Users/xiayunsun/Downloads/t10k-images-idx3-ubyte"
  val testLabelFile = "/Users/xiayunsun/Downloads/t10k-labels-idx1-ubyte"

  val trainData = Preprocessor.prepTrainData(trainLabelFile, trainImgFile)
  val testData  = Preprocessor.prepTrainData(testLabelFile, testImgFile)

  // shuffle training data: todo
  val batchSize    = 1000

  val model: Model = modelName match {
    case "logisticregression"       => new LogisticRegression
    case "logisticregression-graph" => new LogisticRegressionUsingGraph
    case "onehiddenlayer" => new OneHiddenLayerNetwork
  }

  val trainedWeights =
    trainData
      .chunkN(batchSize)
      .fold(model.initialWeights)(model.trainChunk)
      .compile
      .toVector
      .unsafeRunSync()
      .head

  print("now predict test dataset...")

  val prediction =
    testData
      .map {
        case (y, img) =>
          val yHat       = model.predict(trainedWeights, img)
          val prediction = if (yHat > 0.5) 1 else 0
          y == prediction
      }
      .compile
      .toVector
      .unsafeRunSync()

  println(s"correct prediction: ${prediction.count(t => t) / prediction.size.toDouble}")

  println(s"elapsed: ${(System.currentTimeMillis() - start) / 1000.0} seconds")

}
