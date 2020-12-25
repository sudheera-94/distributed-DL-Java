package org.DistributedDL.examples.mnist.mnistCyclicLr;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import static org.DistributedDL.examples.mnist.mnistTraditional.MnistSpark.getDataSetIterator;

public class MnistSuperConvergence {

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        int batchSize = 512;
        int height = 28;
        int width = 28;
        int channels = 1;
        int outputNum = 10;
        int rngseed = 123;

        int numEpochs = 12;

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
    }
}
