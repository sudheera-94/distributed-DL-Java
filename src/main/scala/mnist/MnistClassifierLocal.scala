package mnist

import java.io.File
import java.util.Random

import org.apache.log4j.BasicConfigurator
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.common.io.ClassPathResource
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.evaluation.classification.Evaluation


object MnistClassifierLocal {
  @throws[Exception]
  def main(args: Array[String]): Unit = {
    BasicConfigurator.configure()

    val height = 28
    val width = 28
    val channels = 1
    val outputNum = 10
    val batchSize = 54
    val nEpochs = 1

    val seed = 1234
    val randNumGen = new Random(seed)

    println("Data load and vectorization...")
    // Vectorization of train data
    val trainData = new ClassPathResource("/mnist_png/training").getFile
    val trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
    val labelMaker = new ParentPathLabelGenerator();
    val trainRR = new ImageRecordReader(height, width, channels, labelMaker);
    trainRR.initialize(trainSplit);
    val trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

    // Pixel values from 0-255 to 0-1 (min-max scaling)
    val scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(trainIter);
    trainIter.setPreProcessor(scaler);

    val testData = new ClassPathResource("/mnist_png/testing").getFile
    val testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random(12345))
    val testRR = new ImageRecordReader(height, width, channels, labelMaker)
    testRR.initialize(testSplit)
    val testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum)
    testIter.setPreProcessor(scaler)

    // Network configuration
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .l2(0.0005)
      .weightInit(WeightInit.XAVIER)
      .updater(new Adam(1e-3))
      .list()
      .layer(new ConvolutionLayer.Builder(5, 5)
        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
        .nIn(channels)
        .stride(1, 1)
        .nOut(20)
        .activation(Activation.IDENTITY)
        .build())
      .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(new ConvolutionLayer.Builder(5, 5)
        //Note that nIn need not be specified in later layers
        .stride(1, 1)
        .nOut(50)
        .activation(Activation.IDENTITY)
        .build())
      .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(new DenseLayer.Builder().activation(Activation.RELU)
        .nOut(500).build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .activation(Activation.SOFTMAX)
        .build())
      .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
      .build();

    // Init the model
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))

    for (i <- 0 until nEpochs) {
      model.fit(trainIter)
      System.out.println("*** Completed epoch " + i + " ***")
      val eval = model.evaluate(testIter) : Evaluation
      println(eval.stats())
      trainIter.reset()
      testIter.reset()
    }

    // Save the serialized model
//    ModelSerializer.writeModel(model, new File(System.getProperty("user.home") + "/minist-model.zip"), true);
  }
}

