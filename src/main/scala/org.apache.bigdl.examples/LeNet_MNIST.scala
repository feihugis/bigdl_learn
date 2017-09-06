package org.apache.bigdl.examples

import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet, Sample}
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils._
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch, GreyImgToSample}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.models.lenet.LeNet5
import org.apache.log4j.{Level, Logger}
import com.intel.analytics.bigdl.models.lenet.Utils._
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}

import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer

/**
  * Created by Fei Hu on 8/7/17.
  */
object LeNet_MNIST {
  def load(featureFile: String, labelFile: String): Array[ByteRecord] = {

    val featureBuffer = if (featureFile.startsWith("hdfs:")) {
      ByteBuffer.wrap(File.readHdfsByte(featureFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }
    val labelBuffer = if (featureFile.startsWith("hdfs:")) {
      ByteBuffer.wrap(File.readHdfsByte(labelFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    }
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f)
      i += 1
    }

    result
  }

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setMaster("local[4]").setAppName("BigDL-MNIST")
    val sc = new SparkContext(conf)
    Engine.init

    val classNum = 10
    val batchSize = 6*12
    val maxEpoch = 1

    val trainMean = 0.13066047740239506
    val trainStd = 0.3081078

    val testMean = 0.13251460696903547
    val testStd = 0.31048024

    val filePrefix = "hdfs:///user/fei/mnist/" ///home/fei/BigDL/mnist_data/" //"/Users/fei.hu1@ibm.com/Documents/GitHub/bigdl_learn/src/main/resources/MNIST/"
    val trainData = filePrefix + "train-images-idx3-ubyte"
    val trainLabel = filePrefix + "train-labels-idx1-ubyte"
    val validationData = filePrefix + "t10k-images-idx3-ubyte"
    val validationLabel = filePrefix + "t10k-labels-idx1-ubyte"

    val model = Sequential().add(Reshape(Array(1, 28, 28)))
                            .add(SpatialConvolution(1, 6, 5, 5))
                            .add(Tanh())
                            .add(SpatialMaxPooling(2, 2, 2, 2))
                            .add(Tanh())
                            .add(SpatialConvolution(6, 12, 5, 5))
                            .add(SpatialMaxPooling(2, 2, 2, 2))
                            .add(Reshape(Array(12 * 4 * 4)))
                            .add(Linear(12 * 4 * 4, 100))
                            .add(Tanh())
                            .add(Linear(100, classNum))
                            .add(LogSoftMax())

    val trainSet = DataSet.array(load(trainData, trainLabel), sc) -> BytesToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(batchSize)
    val validationSet = DataSet.array(load(validationData, validationLabel), sc) -> BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(batchSize)
    val train_size = trainSet.size()
    val optimizer = Optimizer(
      model = model,
      dataset = trainSet,
      criterion = ClassNLLCriterion[Float]())

    optimizer.setValidation(trigger = Trigger.everyEpoch, dataset = validationSet, vMethods = Array(new Top1Accuracy))
      .setOptimMethod(new Adagrad(learningRate=0.01, learningRateDecay=0.0002))
      .setEndWhen(Trigger.maxEpoch(maxEpoch)).optimize()

    model.saveWeights("./LeNet", true)

    val rawData = load(validationData, validationLabel)
    val iter = rawData.iterator
    val sampleIter = GreyImgToSample()(GreyImgNormalizer(trainMean, trainStd)(BytesToGreyImg(28, 28)(iter)))

    var samplesBuffer = ArrayBuffer[Sample[Float]]()

    while (sampleIter.hasNext) {
      val elem = sampleIter.next().clone()
      samplesBuffer += elem
    }

    val samples = samplesBuffer.toArray
    val localModel = LocalModule(model)
    val result = localModel.predict(samples)
    val result_class = localModel.predictClass(samples)
    result_class.foreach(r => println(s"${r}"))
  }

}
