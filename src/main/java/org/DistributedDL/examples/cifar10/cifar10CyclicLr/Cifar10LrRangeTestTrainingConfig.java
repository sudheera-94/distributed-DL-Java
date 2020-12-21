package org.DistributedDL.examples.cifar10.cifar10CyclicLr;

import org.DistributedDL.CyclicLr.lrRangeTestScheduleGenerator;
import org.DistributedDL.StandardArchitectures.Cifar10CaffeArchitecture;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

public class Cifar10LrRangeTestTrainingConfig {


    private int rangeTestEpochCount = 4;
    private double maxLr = 0.02;
    private int batchSize = 100;
    private Map<Integer, Double> lrSchedule;

    public Map<Integer, Double> getLrSchedule() {
        return lrSchedule;
    }

    public int getRangeTestEpochCount() {
        return rangeTestEpochCount;
    }

    public void setRangeTestEpochCount(int rangeTestEpochCount) {
        this.rangeTestEpochCount = rangeTestEpochCount;
    }

    // Constructors
    public Cifar10LrRangeTestTrainingConfig(int batchSize, int rangeTestEpochCount) {
        this.batchSize = batchSize;
        this.rangeTestEpochCount = rangeTestEpochCount;
    }

    public Cifar10LrRangeTestTrainingConfig(int batchSize, int rangeTestEpochCount, double maxLr) {
        this(batchSize, rangeTestEpochCount);
        this.maxLr = maxLr;
    }

    // Other Methods
    public MultiLayerConfiguration getArchitecture() {

        int iterations = 1; // Number of training iterations
        int nChannels = 3; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        this.setLrSchedule();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(iterations)     // Training iterations per minibatch

                // The base learning rate, momentum and the weight decay of the network.
                .updater(new Nesterovs(0.9d))
                .regularization(true).l2(0.004d)

                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)

                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(nChannels)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.UNIFORM)
                        .name("conv1")
                        .nOut(32)
                        .padding(1, 1)
                        .stride(1, 1)
                        .build())
                .layer(1, new ConvolutionLayer.Builder(3, 3)
                        .name("conv2")
                        .nOut(32)
                        .weightInit(WeightInit.UNIFORM)
                        .activation(Activation.RELU)
                        .padding(1, 1)
                        .stride(1, 1)
                        .build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("pool1")
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .dropOut(0.2d)
                        .build())
                .layer(3, new ConvolutionLayer.Builder(3, 3)
                        .name("conv3")
                        .nOut(64)
                        .weightInit(WeightInit.UNIFORM)
                        .activation(Activation.RELU)
                        .padding(1, 1)
                        .stride(1, 1)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .name("conv4")
                        .nOut(64)
                        .weightInit(WeightInit.UNIFORM)
                        .activation(Activation.RELU)
                        .padding(1, 1)
                        .stride(1, 1)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("pool2")
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .dropOut(0.2d)
                        .build())
                .layer(6, new ConvolutionLayer.Builder(3, 3)
                        .name("conv5")
                        .nOut(128)
                        .weightInit(WeightInit.UNIFORM)
                        .activation(Activation.RELU)
                        .padding(1, 1)
                        .stride(1, 1)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(3, 3)
                        .name("conv6")
                        .nOut(128)
                        .weightInit(WeightInit.UNIFORM)
                        .activation(Activation.RELU)
                        .padding(1, 1)
                        .stride(1, 1)
                        .build())
                .layer(8, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("pool3")
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .dropOut(0.2d)
                        .build())
                .layer(9, new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.UNIFORM)
                        .build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(32, 32, 3))
                .backprop(true).pretrain(false)
                .build();

        return conf;

    }

    private void setLrSchedule() {
//        lrRangeTestScheduleGenerator.setMinLr(0.01d);

        Map<Integer, Double> lrScheduleNew =
                lrRangeTestScheduleGenerator.getSchedule(this.maxLr, 50000,
                        this.batchSize, this.rangeTestEpochCount);
        this.lrSchedule = lrScheduleNew;

    }

}