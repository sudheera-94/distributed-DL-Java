package org.DistributedDL.examples.mnist;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.log4j.BasicConfigurator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class MnistSpark {

    private static final Logger log = LoggerFactory.getLogger(MnistSpark.class);

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 16;

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 1;

    @Parameter(names = "-avgFreq", description = "Number of iterations per exploration step")
    private int avgFreq = 5;

    // Main function of the class
    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure(); // To configure logging
        new MnistSpark().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int height = 28;
        int width = 28;
        int channels = 1;
        int outputNum = 10;

        //Handle command line arguments
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try { Thread.sleep(500); } catch (Exception e2) { }
            throw e;
        }

        // Configuring JavaSparkContext
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[2]");
        }
        sparkConf.setAppName("DL4J Spark MLP Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load the data into memory then parallelize
        //This isn't a good approach in general - but is simple to use for this example

        // Creating DataSetIterators
        DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, true, 12345);

        // Preparing training data
            // Define the File Path
//        File trainDataFiles = new File("mnist_png/training");
//            // Define the FileSplit(PATH, ALLOWED FORMATS,random)
//        FileSplit train = new FileSplit(trainDataFiles, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
//
//            // Extract the parent path as the image label
//        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
//        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
//
//            // Initialize the record reader and iterator
//        recordReader.initialize(train);
//        DataSetIterator iterTrain = new RecordReaderDataSetIterator(recordReader,batchSizePerWorker,1,
//                outputNum);

//        DataSetIterator iterTrain = getDataSetIterator("mnist_png/training", rngseed, height, width, channels,
//                batchSizePerWorker, outputNum);

            // Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(iterTrain);
        iterTrain.setPreProcessor(scaler);

            // Parallelize dataset
        List<DataSet> trainDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }
        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);

        //Create network configuration
        MultiLayerConfiguration conf = getMnistNetwork();

        // Configuring synchronous training master
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
                .averagingFrequency(avgFreq)            // Number of iterations per exploration stage
                .workerPrefetchNumBatches(2)            // Number of minibatches to asynchronously prefetch on each worker when training.
                .batchSizePerWorker(batchSizePerWorker) // Number of examples per user per fit
                .build();

        //Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

        // Integrating UI dashboard server
        StatsStorageRouter remoteUiRouter = new RemoteUIStatsStorageRouter("http://localhost:9000");
        sparkNet.setListeners(remoteUiRouter, Collections.singletonList(new StatsListener(null)));

        //Execute training:
        for (int i = 0; i < numEpochs; i++) {
            sparkNet.fit(trainData);
            log.info("Completed Epoch {}", i);
        }

        // Test DataSet Iterator
//        DataSetIterator iterTest = getDataSetIterator("mnist_png/testing", rngseed, height, width, channels,
//                batchSizePerWorker, outputNum);
        scaler.fit(iterTest);
        iterTest.setPreProcessor(scaler);

            // Parallelize dataset
        List<DataSet> testDataList = new ArrayList<>();
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);

        //Perform evaluation (distributed)
        Evaluation evaluation = sparkNet.evaluate(testData);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        //Delete the temp training files
        tm.deleteTempFiles(sc);

        log.info("***** Example Complete *****");
    }

    // Function to give the network architecture
    public static MultiLayerConfiguration getMnistNetwork() {

        int iterations = 1; // Number of training iterations
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes

        // learning rate schedule in the form of <Iteration #, Learning Rate>
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, 0.01);
        lrSchedule.put(1000, 0.005);
        lrSchedule.put(3000, 0.001);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(iterations) // Training iterations as above
                .regularization(true).l2(0.0005)
                /*
                    Uncomment the following for learning decay and bias
                 */
                .learningRate(.01)//.biasLearningRate(0.02)

                /*
                    Alternatively, you can use a learning rate schedule.

                    NOTE: this LR schedule defined here overrides the rate set in .learningRate(). Also,
                    if you're using the Transfer Learning API, this same override will carry over to
                    your new model configuration.
                */
//                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
//                .learningRateSchedule(lrSchedule)

                /*
                    Below is an example of using inverse policy rate decay for learning rate
                */
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse)
                //.lrPolicyDecayRate(0.001)
                //.lrPolicyPower(0.75)

                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false).build();

        return conf;
    }

    // Function to load data
    /*
    * Inputs :-
    *   path      - Path to find dataset (One mail folder -> In it folder for each class. Folder name = class name)
    *   rngseed   - Random seed
    *   height    - Image height
    *   width     - Image width
    *   channels  - Number of channels in an image
    *   batchSize - Batch size per worker
    *   outputNum - Number of classes
    */
    public static DataSetIterator getDataSetIterator(String path, int rngseed, int height, int width, int channels,
                                                     int batchSize, int outputNum)
            throws IOException {
        Random randNumGen = new Random(rngseed);

        File data = new File(path);
        FileSplit dataSplit = new FileSplit(data, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(dataSplit);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        return dataIter;

    }

}
