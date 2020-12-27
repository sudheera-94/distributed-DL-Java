package org.DistributedDL.examples.mnist.mnistSuperConvergence;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.DistributedDL.examples.mnist.mnistTraditional.MnistSpark;
import org.apache.log4j.BasicConfigurator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.DistributedDL.examples.mnist.mnistTraditional.MnistSpark.getDataSetIterator;

public class MnistSuperConvergenceSpark {

    private static final Logger log = LoggerFactory.getLogger(MnistSuperConvergenceSpark.class);

    @Parameter(names = "-useSparkLocal",
            description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 512;

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 12;

    @Parameter(names = "-avgFreq", description = "Number of iterations per exploration step")
    private int avgFreq = 10;

    @Parameter(names = "-numWorkers", description = "Number of workers in the cluster")
    private int numWorkers = 1;

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure(); // To configure logging
        new MnistSuperConvergenceSpark().entryPoint(args);
    }

    private void entryPoint(String[] args) throws Exception {
        int rngseed = 123;      // Random Seed
        int height = 28;        // image height
        int width = 28;         // image width
        int channels = 1;       // # image channels
        int outputNum = 10;     // # classes in the dataset

        //Handle command line arguments
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        // Configuring JavaSparkContext
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[" + numWorkers +"]");
        }
        sparkConf.setAppName("DL4J Spark Mnist Super Convergence Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load the data into memory then parallelize
        //This isn't a good approach in general - but is simple to use for this example
        DataSetIterator iterTrain = getDataSetIterator("mnist_png/training", rngseed, height, width, channels,
                batchSizePerWorker, outputNum);

        // Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterTrain);
        iterTrain.setPreProcessor(scaler);

        // Parallelize dataset
        List<DataSet> trainDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }
        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);

        // Network configuration
        int perWorkerTrainSize = 60000/numWorkers;
        MnistSuperConvergenceConfig leNetSuperConvergenceConfig =
                new MnistSuperConvergenceConfig(batchSizePerWorker, numEpochs, perWorkerTrainSize);
        MultiLayerConfiguration conf = leNetSuperConvergenceConfig.getArchitecture();

        // Configuring synchronous training master
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
                //Each DataSet object: contains (by default) 16 examples
                .averagingFrequency(avgFreq)            // Number of iterations per exploration stage
                .workerPrefetchNumBatches(2)            // Number of minibatches to asynchronously
                // prefetch on each worker when training.
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
        DataSetIterator iterTest = getDataSetIterator("mnist_png/testing", rngseed, height, width, channels,
                batchSizePerWorker, outputNum);
        scaler.fit(iterTest);
        iterTest.setPreProcessor(scaler);

        // Create Eval object with 10 possible classes
        Evaluation eval = new Evaluation(outputNum);

        // Getting the multilayer network model from SparkDl4jMultiLayer class
        MultiLayerNetwork model = sparkNet.getNetwork();

        // Evaluate the network
        while (iterTest.hasNext()) {
            DataSet next = iterTest.next();
            INDArray output = model.output(next.getFeatureMatrix());
            // Compare the Feature Matrix from the model
            // with the labels from the RecordReader
            eval.eval(next.getLabels(), output);
        }
        log.info("***** Evaluation *****");
        log.info(eval.stats());

        //Delete the temp training files
        tm.deleteTempFiles(sc);

        log.info("***** Example Complete *****");
    }

}
