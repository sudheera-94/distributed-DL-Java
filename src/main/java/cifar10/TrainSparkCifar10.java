package cifar10;

import com.beust.jcommander.Parameter;
import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.loader.impl.RecordReaderFileBatchLoader;
import org.deeplearning4j.datasets.fetchers.Cifar10Fetcher;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;

public class TrainSparkCifar10 {

    public static final Logger log = LoggerFactory.getLogger(TrainSparkCifar10.class);

    /* --- Required Arguments -- */

    @Parameter(names = {"--dataPath"}, description = "Path (on HDFS or similar) of data preprocessed by preprocessing script." +
        " See PreprocessLocal or PreprocessSpark", required = true)
    private String dataPath;

//    @Parameter(names = {"--masterIP"}, description = "Controller/master IP address - required. For example, 10.0.2.4", required = true)
//    private String masterIP;

//    @Parameter(names = {"--networkMask"}, description = "Network mask for Spark communication. For example, 10.0.0.0/16", required = true)
//    private String networkMask;

//    @Parameter(names = {"--numNodes"}, description = "Number of Spark nodes (machines)", required = true)
//    private int numNodes;

    @Parameter(names = {"--avgFreq"}, description = "Number of training iterations per exploitation", required = true)
    private int avgFreq;

    /* --- Optional Arguments -- */

    @Parameter(names = {"--saveDirectory"}, description = "If set: save the trained network plus evaluation to this directory." +
        " Otherwise, the trained net will not be saved")
    private String saveDirectory = null;

    @Parameter(names = {"--sparkAppName"}, description = "App name for spark. Optional - can set it to anything to identify your job")
    private String sparkAppName = "DL4JMnist";

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private int numEpochs = 20;

    @Parameter(names = {"--minibatch"}, description = "Minibatch size (of preprocessed minibatches). Also number of" +
        "minibatches per worker when fitting")
    private int minibatch = 32;

    @Parameter(names = {"--numWorkersPerNode"}, description = "Number of workers per Spark node. Usually use 1 per GPU, or 1 for CPU-only workers")
    private int numWorkersPerNode = 1;

    @Parameter(names = {"--gradientThreshold"}, description = "Gradient threshold. See ")
    private double gradientThreshold = 1E-3;

    @Parameter(names = {"--port"}, description = "Port number for Spark nodes. This can be any free port (port must be free on all nodes)")
    private int port = 40123;

    public static void main(String[] args) throws Exception {
        new TrainSparkCifar10().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommanderUtils.parseArgs(this, args);

        SparkConf conf = new SparkConf();
        conf.setAppName(sparkAppName);
        System.out.println(conf.toDebugString());
        JavaSparkContext sc = new JavaSparkContext(conf);


        //Set up TrainingMaster for gradient sharing training
        //Create the TrainingMaster instance
        int examplesPerDataSetObject = 1;
        TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
            .rngSeed(12345)
            .collectTrainingStats(false)
            .batchSizePerWorker(minibatch)              // Minibatch size for each worker
            .averagingFrequency(avgFreq)                // Number of training iterations per exploitation
            .build();


        MultiLayerNetwork net = getNetwork();
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, net, trainingMaster);
        sparkNet.setListeners(new PerformanceListener(10, true));

        //Create data loader
        int imageHeightWidth = 32;      //64x64 pixel input
        int imageChannels = 3;          //RGB
        int numClasses = Cifar10Fetcher.NUM_LABELS;

        PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(imageHeightWidth, imageHeightWidth, imageChannels, labelMaker);
        rr.setLabels(new Cifar10DataSetIterator(1).getLabels());
        RecordReaderFileBatchLoader loader = new RecordReaderFileBatchLoader(rr, minibatch, 1, numClasses);
        loader.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range

        //Fit the network
        String trainPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "train";
        JavaRDD<String> pathsTrain = SparkUtils.listPaths(sc, trainPath);
        for (int i = 0; i < numEpochs; i++) {
            log.info("--- Starting Training: Epoch {} of {} ---", (i + 1), numEpochs);
            sparkNet.fitPaths(pathsTrain, loader);
        }

        //Perform evaluation
        String testPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "test";
        JavaRDD<String> pathsTest = SparkUtils.listPaths(sc, testPath);
        Evaluation evaluation = new Evaluation(Cifar10DataSetIterator.getLabels(false), 5); //Set up for top 5 accuracy
        evaluation = (Evaluation) sparkNet.doEvaluation(pathsTest, loader, evaluation)[0];
        log.info("Evaluation statistics: {}", evaluation.stats());

        if (saveDirectory != null && saveDirectory.isEmpty()) {
            log.info("Saving the network and evaluation to directory: {}", saveDirectory);

            // Save network
            String networkPath = FilenameUtils.concat(saveDirectory, "network.bin");
            FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
            try (BufferedOutputStream os = new BufferedOutputStream(fileSystem.create(new Path(networkPath)))) {
                ModelSerializer.writeModel(sparkNet.getNetwork(), os, true);
            }

            // Save evaluation
            String evalPath = FilenameUtils.concat(saveDirectory, "evaluation.txt");
            SparkUtils.writeStringToFile(evalPath, evaluation.stats(), sc);
        }


        log.info("----- Cifar10 Complete -----");
    }

    public static MultiLayerNetwork getNetwork() {

        int seed = 123;
        int channels = 3; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int numLabels = Cifar10Fetcher.NUM_LABELS;
        int height = 32;
        int width = 32;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new AdaDelta())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nIn(channels).nOut(32).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(16).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(32).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(128).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(numLabels).build())
                .layer(new BatchNormalization())

                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.AVG).build())

                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .dropOut(0.8)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        return model;
    }

}
