package org.DistributedDL.SuperConvergence;

import java.util.HashMap;
import java.util.Map;

public class SuperConvergenceScheduleGenerator {

    private static double minLr;
    private static double maxLr;
    private static int batchSize;
    private static int numEpochs;
    private static int trainSize;
    private static int stepSize;

    public static void setMinLr(double minLr) {
        SuperConvergenceScheduleGenerator.minLr = minLr;
    }

    public static void setMaxLr(double maxLr) {
        SuperConvergenceScheduleGenerator.maxLr = maxLr;
    }

    public static void setBatchSize(int batchSize) {
        SuperConvergenceScheduleGenerator.batchSize = batchSize;
    }

    public static void setNumEpochs(int numEpochs) {
        SuperConvergenceScheduleGenerator.numEpochs = numEpochs;
    }

    public static void setTrainSize(int trainSize) {
        SuperConvergenceScheduleGenerator.trainSize = trainSize;
    }

    public static void setStepSize(int stepSize) {
        SuperConvergenceScheduleGenerator.stepSize = stepSize;
    }

    // Other methods
    public static Map<Integer, Double> getSchedule(double maxLr, double minLr, int numEpochs, int trainSize,
                                                   int batchSize, int stepSize) {
        setBatchSize(batchSize);
        setMaxLr(maxLr);
        setNumEpochs(numEpochs);
        setMinLr(minLr);
        setTrainSize(trainSize);
        setStepSize(stepSize);

        int arrayLenPerEpoch = (int) (trainSize / batchSize) + 1;
        int totalIterationsCount = arrayLenPerEpoch * numEpochs;

        int[] iterArray = getIterArray(totalIterationsCount);
        double[] lrArray = getLrArray(totalIterationsCount, arrayLenPerEpoch);

        Map<Integer, Double> lrSchedule = new HashMap<>();
        for (int i = 0; i < totalIterationsCount; i++) {
            lrSchedule.put(iterArray[i], lrArray[i]);
        }
        return lrSchedule;
    }

    private static double[] getLrArray(int totalIterationsCount, int arrayLenPerEpoch) {

        double[] lrArray = new double[totalIterationsCount];
        int iterCountPerStep = stepSize * arrayLenPerEpoch;
        int count = 0;
        double startLr = minLr;
        double stopLr = maxLr;
        double memLr;

        for (int i = 0; i < totalIterationsCount; i++) {

            if ((count % iterCountPerStep) == 0) {
                count = 0;
                memLr = startLr;
                startLr = stopLr;
                stopLr = memLr;
            }

            lrArray[i] = startLr + (((stopLr - startLr)/iterCountPerStep) * count);
            count = count + 1;
        }

        return lrArray;

    }

    private static int[] getIterArray(int totalIterationsCount) {

        int[] iterArray = new int[totalIterationsCount];

        for (int i = 0; i < totalIterationsCount; i++) {
            iterArray[i] = i;
        }

        return iterArray;
    }
}
