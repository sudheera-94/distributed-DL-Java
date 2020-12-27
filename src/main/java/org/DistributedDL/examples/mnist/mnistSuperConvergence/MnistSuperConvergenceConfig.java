package org.DistributedDL.examples.mnist.mnistSuperConvergence;

import org.DistributedDL.SuperConvergence.SuperConvergenceScheduleGenerator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

public class MnistSuperConvergenceConfig {

    private int superConvergenceEpochCount = 12;
    private int batchSize = 512;
    private Map<Integer, Double> lrSchedule;
    private static double minLr = 0.01d;
    private static double maxLr = 0.1d;
    private static int trainSize = 60000;
    private static int stepSize = 5;

    // Constructors

    public MnistSuperConvergenceConfig(int batchSize, int superConvergenceEpochCount) {
        this.batchSize = batchSize;
        this.superConvergenceEpochCount = superConvergenceEpochCount;
    }

    public MnistSuperConvergenceConfig(int batchSize, int superConvergenceEpochCount, int trainSize) {
        this(batchSize, superConvergenceEpochCount);
        this.trainSize = trainSize;
    }

    // Other methods

    public MultiLayerConfiguration getArchitecture() {

        int iterations = 1; // Number of training iterations
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        this.setLrSchedule();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(iterations)     // Training iterations per minibatch

                // The base learning rate, momentum and the weight decay of the network.
                .learningRate(0.0001d)        // Learning rate
                .biasLearningRate(0.0002d)
                .updater(new Nesterovs(0.9d))
                .regularization(true)
                .l2(0.0005d)

                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)

                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        // nIn and nOut specify depth. nIn here is the nChannels and
                        // nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .name("conv1")
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("pool1")
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .name("conv2")
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("pool2")
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        // Cross entropy loss
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .backprop(true).pretrain(false).build();

        return conf;
    }

    private void setLrSchedule() {

        Map<Integer, Double> lrScheduleNew =
                SuperConvergenceScheduleGenerator.getSchedule(this.maxLr, this.minLr, this.superConvergenceEpochCount,
                        this.trainSize, this.batchSize, this.stepSize);
        this.lrSchedule = lrScheduleNew;

    }
}
