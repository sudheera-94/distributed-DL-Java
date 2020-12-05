package org.DistributedDL.CyclicLr;

import java.util.HashMap;
import java.util.Map;

public class lrRangeTestScheduleGenerator {

    private static double lrResolution = 0.0001d;
    private static double minLr = 0.0001d;

    public static Map<Integer, Double> getSchedule(double maxLr, int trainSize, int batchSize,
                                                   int rangeTestEpochCount) {

        int lrArrayLen = (int) ((maxLr - minLr + lrResolution) / lrResolution);
        double[] lrArray = getLrArray(maxLr, lrArrayLen);
        int[] iterArray = getIterArray(lrArray, lrArrayLen, trainSize, batchSize, rangeTestEpochCount, maxLr);

        Map<Integer, Double> lrSchedule = new HashMap<>();
        for (int i = 0; i < lrArrayLen; i++) {
            lrSchedule.put(iterArray[i], lrArray[i]);
        }
        return lrSchedule;

    }

    private static double[] getLrArray(double maxLr, int lrArrayLen) {

        double[] lrArray = new double[lrArrayLen];

        for (double lr = minLr; lr <= maxLr; lr = lr + lrResolution) {
            lrArray[((int) (lr / lrResolution)) - 1] = lr;
        }

        return lrArray;

    }

    private static int[] getIterArray(double[] lrArray, int lrArrayLen, int trainSize, int batchSize,
                                      int rangeTestEpochCount, double maxLr) {

        int[] iterArray = new int[lrArrayLen];
        double gradDenominator = (double) rangeTestEpochCount * ((double) trainSize / (double) batchSize);
        double gradNumerator = maxLr - minLr;
        double grad = gradNumerator / gradDenominator;

        for (int i = 0; i < lrArrayLen; i++) {
            iterArray[i] = (int) ((lrArray[i] - minLr) / grad);
        }

        return iterArray;

    }

}