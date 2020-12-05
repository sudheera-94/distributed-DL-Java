package org.DistributedDL.StandardArchitectures;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Cifar10CaffeArchitecture extends BaseArchitecture {

    public Cifar10CaffeArchitecture() {
        super(3, 10, 50000);
    }

    public static MultiLayerConfiguration getArchitecture() {

        // Set variables in the base architecture using constructor
        Cifar10CaffeArchitecture setVars = new Cifar10CaffeArchitecture();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(getIterations())

                // The base learning rate, momentum and the weight decay of the network.
                .learningRate(0.001d)
                .biasLearningRate(0.002d)
                .updater(new Nesterovs(0.9d))
                .l2(0.004d)

                // learning rate policy
                .learningRateDecayPolicy(LearningRatePolicy.None)

                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(getnChannels())
                        .name("conv1")
                        .nOut(32)
                        .padding(2,2)
                        .stride(1,1)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("pool1")
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(2, new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .name("relu1").build())
                .layer(3, new ConvolutionLayer.Builder(5, 5)
                        .name("conv2")
                        .nOut(32)
                        .padding(2,2)
                        .stride(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                        .name("pool2")
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(6, new ConvolutionLayer.Builder(5, 5)
                        .name("conv3")
                        .nOut(64)
                        .padding(2,2)
                        .stride(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .nOut(64).build())
                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        // Cross entropy loss
                        .nOut(getnClasses())
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(32, 32, 3))
                .backprop(true).pretrain(false).build();

        return conf;
    }

}
