package org.DistributedDL.examples.mnist.mnistTraditional;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import org.DistributedDL.StandardArchitectures.LeNet5Architecture;

import static org.DistributedDL.examples.mnist.mnistTraditional.MnistSpark.getDataSetIterator;

/**
 * Trains a CNN using Mnist dataset locally. (without spark)
 */
public class MnistLocal {

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        int batchSize = 64;
        int numEpochs = 1;
        int height = 28;
        int width = 28;
        int channels = 1;
        int outputNum = 10;
        int rngseed = 123;

        // Loading training data
        System.out.println("Data load and vectorization...");

        // Initialize the training set iterator
        DataSetIterator trainIter = getDataSetIterator("mnist_png/training", rngseed, height, width,
                channels, batchSize, outputNum);

        // Pixel values from 0-255 to 0-1 (min-max scaling)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        // Initialize the testing set iterator
        DataSetIterator iterTest = getDataSetIterator("mnist_png/testing", rngseed, height, width,
                channels, batchSize, outputNum);
        scaler.fit(iterTest);
        iterTest.setPreProcessor(scaler);

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage();   //Alternative: new FileStatsStorage(File) - see UIStorageExample
        int listenerFrequency = 1;

        // Network configuration
        MultiLayerConfiguration conf = LeNet5Architecture.getArchitecture();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new StatsListener(statsStorage, listenerFrequency));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage
        // to be visualized
        uiServer.attach(statsStorage);

        // Training the network
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIter);
            System.out.println("*** Completed epoch " + i + " ***");
        }

        System.out.println("******EVALUATE MODEL******");

        // Create Eval object with 10 possible classes
        Evaluation eval = new Evaluation(outputNum);
        iterTest.reset();

        // Evaluate the network
        while (iterTest.hasNext()) {
            DataSet next = iterTest.next();
            INDArray output = model.output(next.getFeatureMatrix());
            // Compare the Feature Matrix from the model
            // with the labels from the RecordReader
            eval.eval(next.getLabels(), output);
        }

        System.out.println(eval.stats());
        System.out.println(eval.confusionToString());

    }

}
