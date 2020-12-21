package org.DistributedDL.StandardArchitectures;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Vgg16Architecture extends BaseArchitecture {

    public Vgg16Architecture(int nChannels, int nClasses, int trainSize) {
        super(nChannels, nClasses, trainSize);
    }

    public static MultiLayerConfiguration getArchitecture(int nChannels, int nClasses, int trainSize) {

        // Set parameters in the base architecture using constructor
        Vgg16Architecture setVars = new Vgg16Architecture(nChannels, nClasses, trainSize);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(getIterations())

                // The base learning rate, momentum and the weight decay of the network.
                .learningRate(0.001d)
                .biasLearningRate(0.002d)
                .updater(new Nesterovs(0.9d))
                .regularization(true)
                .l2(0.004d)

                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(getnChannels())
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
                        .nOut(getnClasses())
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(32, 32, 3))
                .backprop(true).pretrain(false)
                .build();

        return conf;
    }
}
