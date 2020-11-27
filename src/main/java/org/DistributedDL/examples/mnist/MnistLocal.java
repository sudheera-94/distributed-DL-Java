package org.DistributedDL.examples.mnist;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;

import static org.DistributedDL.examples.mnist.MnistSpark.getMnistNetwork;

public class MnistLocal {

    public static void main(String[] args) throws Exception{
        BasicConfigurator.configure();

        File trainData;

        {
            try {

                int batchSizePerWorker = 16;
                int numEpochs = 1;

                // Loading data and Visualization
                System.out.println("Data load and vectorization...");

                DataSetIterator trainIter = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
                DataSetIterator testIter = new MnistDataSetIterator(batchSizePerWorker, true, 12345);

                // Pixel values from 0-255 to 0-1 (min-max scaling)
                ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
                scaler.fit(trainIter);
                trainIter.setPreProcessor(scaler);

                // Network configuration
                MultiLayerConfiguration conf = getMnistNetwork();

                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(100));

                for (int i = 0; i < numEpochs; i++){
                    model.fit(trainIter);
                    System.out.println("*** Completed epoch " + i + " ***");
                    Evaluation eval = model.evaluate(testIter);
                    System.out.println(eval.stats());
                    trainIter.reset();
                    testIter.reset();
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

}
