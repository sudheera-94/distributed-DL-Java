package org.DistributedDL.examples.mnist;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class MnistLocal {

    public static void main(String[] args) throws Exception{
        BasicConfigurator.configure();

        File trainData;

        {
            try {

                int height = 28;
                int width = 28;
                int channels = 1;
                int batchSize = 32;
                int outputNum = 10;
                int numEpochs = 1;

                // Loading data and Visualization
                System.out.println("Data load and vectorization...");

                trainData = new ClassPathResource("/mnist_png/training").getFile();
                FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random(12345));
                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
                ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
                trainRR.initialize(trainSplit);
                RecordReaderDataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

                System.out.println("*** Model Evaluation ***");
                File testData = new ClassPathResource("/mnist_png/testing").getFile();
                FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random(12345));
                ParentPathLabelGenerator labelMakerTest = new ParentPathLabelGenerator();
                ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMakerTest);
                testRR.initialize(testSplit);
                RecordReaderDataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);

                // Pixel values from 0-255 to 0-1 (min-max scaling)
                ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
                scaler.fit(trainIter);
                trainIter.setPreProcessor(scaler);

                // Network configuration
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .l2(0.0005)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Adam(1e-3))
                        .list()
                        .layer(new ConvolutionLayer.Builder(5, 5)
                                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                                .nIn(channels)
                                .stride(1,1)
                                .nOut(20)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(new ConvolutionLayer.Builder(5, 5)
                                //Note that nIn need not be specified in later layers
                                .stride(1,1)
                                .nOut(50)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(new DenseLayer.Builder().activation(Activation.RELU)
                                .nOut(500).build())
                        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(outputNum)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                        .build();

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
