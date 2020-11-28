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
                Random randNumGen = new Random(123);

                // Loading data and Visualization
                System.out.println("Data load and vectorization...");

//                DataSetIterator trainIter = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
//                DataSetIterator testIter = new MnistDataSetIterator(batchSizePerWorker, true, 12345);

                // Preparing training data
                // Define the File Path
                File trainDataFiles = new File("mnist_png/training");
                // Define the FileSplit(PATH, ALLOWED FORMATS,random)
                FileSplit train = new FileSplit(trainDataFiles, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

                // Extract the parent path as the image label
                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
                ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

                // Initialize the record reader and iterator
                recordReader.initialize(train);
                DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader,batchSizePerWorker,1,
                        outputNum);

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
                recordReader.reset();

                // Preparing training data
                File testDataFiles = new File("mnist_png/testing");

                // Define the FileSplit(PATH, ALLOWED FORMATS,random)
                FileSplit test = new FileSplit(testDataFiles, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

                // Initialize the record reader
                recordReader.initialize(test);

                // DataSet Iterator
                DataSetIterator iterTest = new RecordReaderDataSetIterator(recordReader,batchSizePerWorker,1,
                        outputNum);
                scaler.fit(iterTest);
                iterTest.setPreProcessor(scaler);

                /*
                log the order of the labels for later use
                In previous versions the label order was consistent, but random
                In current verions label order is lexicographic
                preserving the RecordReader Labels order is no
                longer needed left in for demonstration
                purposes
                */
                System.out.println(recordReader.getLabels().toString());

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
