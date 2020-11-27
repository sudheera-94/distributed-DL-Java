package org.DistributedDL.examples.cifar10;

import com.beust.jcommander.Parameter;
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrainLocalCifar10 {
    public static Logger log = LoggerFactory.getLogger(TrainLocalCifar10.class);

    @Parameter(names = {"--numEpochs"}, description = "Number of epochs for training")
    private int numEpochs = 1;

    @Parameter(names = {"--saveDir"}, description = "If set, the directory to save the trained network")
    private String saveDir;

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();
        new TrainLocalCifar10().entryPoint(args);
    }

    public void entryPoint(String[] args) throws Exception {
        log.info("Process started");
        JCommanderUtils.parseArgs(this, args);

        //Create the data pipeline
        int batchSize = 32;
        DataSetIterator iter = new Cifar10DataSetIterator(batchSize);
        iter.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range

        //Create the network
        ComputationGraph net = TrainSparkCifar10.getNetwork();
        net.setListeners(new PerformanceListener(50, true));

        //Reduce auto GC frequency for better performance
        Nd4j.getMemoryManager().setAutoGcWindow(10000);

        //Fit the network
        net.fit(iter, numEpochs);
        log.info("Training complete. Starting evaluation.");

        //Evaluate the network on test set data
        DataSetIterator test = new Cifar10DataSetIterator(batchSize, DataSetType.TEST);
        test.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range

//        Evaluation e = new Evaluation(Cifar10DataSetIterator.getLabels(true), 1); //Set up for top 1 accuracy
        System.out.println(net.evaluate(test).stats());
        log.info("Evaluation complete");
//        log.info(e.stats());

//        if (saveDir != null && !saveDir.isEmpty()) {
//            File sd = new File(saveDir);
//            if (!sd.exists())
//                sd.mkdirs();
//
//            log.info("Saving network and evaluation stats to directory: {}", saveDir);
//            net.save(new File(saveDir, "trainedNet.bin"));
//            FileUtils.writeStringToFile(new File(saveDir, "evaulation.txt"), e.stats(), StandardCharsets.UTF_8);
//        }

        log.info("----- Examples Complete -----");
    }
}
