package org.DistributedDL.examples.mnist;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static org.DistributedDL.examples.mnist.MnistSpark.getMnistNetwork;
import static org.DistributedDL.examples.mnist.MnistSpark.getDataSetIterator;

public class MnistLocal {

    public static void main(String[] args) throws Exception{
        BasicConfigurator.configure();

        File trainData;

        {
            try {

                int batchSizePerWorker = 16;
                int numEpochs = 1;
                int height = 28;
                int width = 28;
                int channels = 1;
                int outputNum = 10;
                int rngseed = 123;
                Random randNumGen = new Random(rngseed);

                // Loading data and Visualization
                System.out.println("Data load and vectorization...");

                // Initialize the record reader and iterator
                DataSetIterator trainIter = getDataSetIterator("mnist_png/training", rngseed, height, width,
                        channels, batchSizePerWorker, outputNum);

                // Pixel values from 0-255 to 0-1 (min-max scaling)
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
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
                }

                System.out.println("******EVALUATE MODEL******");

                // DataSet Iterator
                DataSetIterator iterTest = getDataSetIterator("mnist_png/testing", rngseed, height, width,
                        channels, batchSizePerWorker, outputNum);;
                scaler.fit(iterTest);
                iterTest.setPreProcessor(scaler);

                // Create Eval object with 10 possible classes
                Evaluation eval = new Evaluation(outputNum);


                // Evaluate the network
                while(iterTest.hasNext()){
                    DataSet next = iterTest.next();
                    INDArray output = model.output(next.getFeatureMatrix());
                    // Compare the Feature Matrix from the model
                    // with the labels from the RecordReader
                    eval.eval(next.getLabels(),output);

                }

                System.out.println(eval.stats());

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

}
