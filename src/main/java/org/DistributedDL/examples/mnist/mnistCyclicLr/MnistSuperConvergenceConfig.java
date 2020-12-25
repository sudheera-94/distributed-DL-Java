package org.DistributedDL.examples.mnist.mnistCyclicLr;

import org.DistributedDL.CyclicLr.LrRangeTestScheduleGenerator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

import java.util.Map;

public class MnistSuperConvergenceConfig {

    private int superConvergenceEpochCount = 12;
    private int batchSize = 64;
    private Map<Integer, Double> lrSchedule;

    // Constructors

    public MnistSuperConvergenceConfig(int batchSize, int superConvergenceEpochCount) {
        this.batchSize = batchSize;
        this.superConvergenceEpochCount = superConvergenceEpochCount;
    }

    // Other methods

    public MultiLayerConfiguration getArchitecture() {

        int iterations = 1; // Number of training iterations
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        this.setLrSchedule();

    }

    private void setLrSchedule() {

        Map<Integer, Double> lrScheduleNew =
                LrRangeTestScheduleGenerator.getSchedule(this.maxLr, 60000,
                        this.batchSize, this.rangeTestEpochCount);
        this.lrSchedule = lrScheduleNew;

    }
}
