//package org.DistributedDL.examples.cifar10;
//
//import com.beust.jcommander.Parameter;
//import org.apache.commons.io.FilenameUtils;
//import org.apache.hadoop.fs.FileSystem;
//import org.apache.hadoop.fs.Path;
//import org.apache.log4j.BasicConfigurator;
//import org.apache.spark.SparkConf;
//import org.apache.spark.api.java.JavaRDD;
//import org.apache.spark.api.java.JavaSparkContext;
//import org.datavec.api.io.labels.ParentPathLabelGenerator;
//import org.datavec.api.io.labels.PathLabelGenerator;
//import org.datavec.image.recordreader.ImageRecordReader;
//import org.deeplearning4j.core.loader.impl.RecordReaderFileBatchLoader;
//import org.deeplearning4j.datasets.fetchers.Cifar10Fetcher;
//import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
//import org.deeplearning4j.eval.Evaluation;
//import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
//import org.deeplearning4j.nn.conf.ConvolutionMode;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.layers.*;
//import org.deeplearning4j.nn.graph.ComputationGraph;
//import org.deeplearning4j.nn.weights.WeightInit;
//import org.deeplearning4j.optimize.listeners.PerformanceListener;
//import org.deeplearning4j.spark.api.TrainingMaster;
//import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
//import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
//import org.deeplearning4j.spark.util.SparkUtils;
//import org.deeplearning4j.util.ModelSerializer;
//import org.deeplearning4j.zoo.model.helper.DarknetHelper;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
//import org.nd4j.linalg.learning.config.AMSGrad;
//import org.nd4j.linalg.lossfunctions.LossFunctions;
//import org.nd4j.linalg.schedule.ISchedule;
//import org.nd4j.linalg.schedule.MapSchedule;
//import org.nd4j.linalg.schedule.ScheduleType;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.io.BufferedOutputStream;
//
//public class TrainSparkCifar10 {
//
//    public static final Logger log = LoggerFactory.getLogger(TrainSparkCifar10.class);
//
//    /* --- Required Arguments -- */
//
//    @Parameter(names = {"--dataPath"}, description = "Path (on HDFS or similar) of data preprocessed by preprocessing script." +
//        " See PreprocessLocal or PreprocessSpark", required = true)
//    private String dataPath;
//
//    @Parameter(names = {"--avgFreq"}, description = "Number of training iterations per exploitation", required = true)
//    private int avgFreq;
//
//    /* --- Optional Arguments -- */
//
//    @Parameter(names = {"--saveDirectory"}, description = "If set: save the trained network plus evaluation to this directory." +
//        " Otherwise, the trained net will not be saved")
//    private String saveDirectory = null;
//
//    @Parameter(names = {"--sparkAppName"}, description = "App name for spark. Optional - can set it to anything to identify your job")
//    private String sparkAppName = "DL4JMnist";
//
//    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
//    private int numEpochs = 1;
//
//    @Parameter(names = {"--minibatch"}, description = "Minibatch size (of preprocessed minibatches). Also number of" +
//        "minibatches per worker when fitting")
//    private int minibatch = 32;
//
//    @Parameter(names = {"--numWorkersPerNode"}, description = "Number of workers per Spark node. Usually use 1 per GPU, or 1 for CPU-only workers")
//    private int numWorkersPerNode = 1;
//
//    @Parameter(names = {"--gradientThreshold"}, description = "Gradient threshold. See ")
//    private double gradientThreshold = 1E-3;
//
//    @Parameter(names = {"--port"}, description = "Port number for Spark nodes. This can be any free port (port must be free on all nodes)")
//    private int port = 40123;
//
//    public static void main(String[] args) throws Exception {
//        BasicConfigurator.configure();
//        new TrainSparkCifar10().entryPoint(args);
//    }
//
//    protected void entryPoint(String[] args) throws Exception {
//        System.out.println("----- Cifar10 training Started -----");
//
//        JCommanderUtils.parseArgs(this, args);
//
//        SparkConf conf = new SparkConf();
//        conf.setAppName(sparkAppName);
//        System.out.println(conf.toDebugString());
//        JavaSparkContext sc = new JavaSparkContext(conf);
//
//
//        //Set up TrainingMaster for gradient sharing training
//        //Create the TrainingMaster instance
//        TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(minibatch)
//            .workerPrefetchNumBatches(2)
//            .batchSizePerWorker(minibatch)              // Minibatch size for each worker
//            .averagingFrequency(avgFreq)                // Number of training iterations per exploitation.
//            .build();
//
//
//        ComputationGraph net = getNetwork();
//        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, net, trainingMaster);
//        sparkNet.setListeners(new PerformanceListener(10, true));
//
//        //Create data loader
//        int imageHeightWidth = 32;      //32x32 pixel input
//        int imageChannels = 3;          //RGB
//        int numClasses = Cifar10Fetcher.NUM_LABELS;
//
//        PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
//        ImageRecordReader rr = new ImageRecordReader(imageHeightWidth, imageHeightWidth, imageChannels, labelMaker);
//        rr.setLabels(new Cifar10DataSetIterator(1).getLabels());
//        RecordReaderFileBatchLoader loader = new RecordReaderFileBatchLoader(rr, minibatch, 1, numClasses);
//        loader.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range
//
//        //Fit the network
//        String trainPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "train";
//        JavaRDD<String> pathsTrain = SparkUtils.listPaths(sc, trainPath);
//
//        for (int i = 0; i < numEpochs; i++) {
//            System.out.println("--- Starting Training: Epoch " + (i + 1) + " of "+ numEpochs +" --- ");
//            log.info("--- Starting Training: Epoch {} of {} ---", (i + 1), numEpochs);
//            sparkNet.fitPaths(pathsTrain, loader);
//        }
//
//        trainingMaster.deleteTempFiles(sc);
//
//        //Perform evaluation
//        String testPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "test";
//        JavaRDD<String> pathsTest = SparkUtils.listPaths(sc, testPath);
//
//        Evaluation evaluation = new Evaluation(Cifar10DataSetIterator.getLabels(true), 1); //Set up for top 1 accuracy
//        evaluation = (Evaluation) sparkNet.doEvaluation(pathsTest, loader, evaluation)[0];
//        log.info("Evaluation complete");
//        System.out.println("Evaluation statistics: " + evaluation.stats());
//        log.info("Evaluation statistics: {}", evaluation.stats());
//
//        if (saveDirectory != null && saveDirectory.isEmpty()) {
//            log.info("Saving the network and evaluation to directory: {}", saveDirectory);
//
//            // Save network
//            String networkPath = FilenameUtils.concat(saveDirectory, "network.bin");
//            FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
//            try (BufferedOutputStream os = new BufferedOutputStream(fileSystem.create(new Path(networkPath)))) {
//                ModelSerializer.writeModel(sparkNet.getNetwork(), os, true);
//            }
//
//            // Save evaluation
//            String evalPath = FilenameUtils.concat(saveDirectory, "evaluation.txt");
//            SparkUtils.writeStringToFile(evalPath, evaluation.stats(), sc);
//        }
//
//        log.info("----- Cifar10 Complete -----");
//    }
//
//    public static ComputationGraph getNetwork() {
//
//        ISchedule lrSchedule = new MapSchedule.Builder(ScheduleType.EPOCH)
//                .add(0, 8e-3)
//                .add(1, 6e-3)
//                .add(3, 3e-3)
//                .add(5, 1e-3)
//                .add(7, 5e-4).build();
//
//        ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
//                .convolutionMode(ConvolutionMode.Same)
//                .l2(1e-4)
//                .updater(new AMSGrad(lrSchedule))
//                .weightInit(WeightInit.RELU)
//                .graphBuilder()
//                .addInputs("input")
//                .setOutputs("output");
//
//        DarknetHelper.addLayers(b, 1, 3, 32, 64, 2);    //32x32 out
//        DarknetHelper.addLayers(b, 2, 2, 64, 128, 0);   //32x32 out
//        DarknetHelper.addLayers(b, 3, 2, 128, 256, 2);   //16x16 out
//        DarknetHelper.addLayers(b, 4, 2, 256, 256, 0);   //16x16 out
//        DarknetHelper.addLayers(b, 5, 2, 256, 512, 2);   //8x8 out
//
//        b.addLayer("convolution2d_6", new ConvolutionLayer.Builder(1, 1)
//                .nIn(512)
//                .nOut(Cifar10Fetcher.NUM_LABELS)
//                .weightInit(WeightInit.XAVIER)
//                .stride(1, 1)
//                .activation(Activation.IDENTITY)
//                .build(), "maxpooling2d_5")
//                .addLayer("globalpooling", new GlobalPoolingLayer.Builder(PoolingType.AVG)
//                        .build(), "convolution2d_6")
//                .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .activation(Activation.SOFTMAX).build(), "globalpooling")
//                .setOutputs("loss");
//
//        ComputationGraphConfiguration conf = b.build();
//
//        ComputationGraph net = new ComputationGraph(conf);
//        net.init();
//
//        return net;
//    }
//
//}
