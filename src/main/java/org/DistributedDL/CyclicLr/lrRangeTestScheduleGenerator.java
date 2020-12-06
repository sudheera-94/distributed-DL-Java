package org.DistributedDL.CyclicLr;

import java.util.HashMap;
import java.util.Map;

/**
 * Before scheduling a Cyclic learning rate This Learning Rate Range test
 * Provides a Map<Integer, Double> which has <iteration, learning rate>
 * to the expected RangeTestConfig class.
 */
public class lrRangeTestScheduleGenerator {

    private static double lrResolution = 0.0001d;
    private static double minLr = 0.0001d;

    /**
     * Set the minimum learning rate to LR range test schedule generator.
     *
     * @param minLr minimum learning rate.
     */
    public static void setMinLr(double minLr) {
        lrRangeTestScheduleGenerator.minLr = minLr;
    }

    /**
     * Returns the expected learning rate schedule. Uses private getLrArray
     * method to get learning rate array and getIterArray
     *
     * @param maxLr               maximum learning rate in the tested range
     * @param trainSize           number of training data points
     * @param batchSize           training data points in a batch
     * @param rangeTestEpochCount number of range test epochs
     * @return learning rate schedule for learning rate range test
     */
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

    /**
     * Provides an array of learning rates starting from minimum learning rate to
     * maximum learning rate as lrResolution field as the step size.
     *
     * @param maxLr      maximum learning rate
     * @param lrArrayLen length of the array of learning rates
     * @return Array of learning rates to perform LR range test
     */
    private static double[] getLrArray(double maxLr, int lrArrayLen) {

        double[] lrArray = new double[lrArrayLen];

        for (double lr = minLr; lr <= maxLr; lr = lr + lrResolution) {

            lrArray[((int) ((lr - minLr) / lrResolution))] = lr;
        }

        return lrArray;

    }

    /**
     * Provides an array of iterations with respect to the array of learning rates. Helps
     * to set the correct learning rate in the correct iteration.
     *
     * @param lrArray             array of learning rates
     * @param lrArrayLen          length of the array of learning rates
     * @param trainSize           number of training data points
     * @param batchSize           training data points in a batch
     * @param rangeTestEpochCount number of range test epochs
     * @param maxLr               maximum learning rate in the tested range
     * @return array of iteration numbers
     */
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