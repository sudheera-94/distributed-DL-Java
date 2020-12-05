package org.DistributedDL.StandardArchitectures;

public class BaseArchitecture {

    private static int iterations = 1; // Number of training iterations
    private static int nChannels;  // Number of input channels
    private static int nClasses;  // The number of possible outcomes
    private static int trainSize;      // Size of the training set

    // Constructors
    public BaseArchitecture(int nChannels, int nClasses, int trainSize){
        this.trainSize = trainSize;
        this.nChannels = nChannels;
        this.nClasses = nClasses;
    }

    public BaseArchitecture(int iterations, int nChannels, int nClasses, int trainSize){
        this(nChannels, nClasses, trainSize);
        this.iterations = iterations;
    }

    // Getters and setters
    public static int getIterations() {
        return iterations;
    }

    public static void setIterations(int iterations) {
        BaseArchitecture.iterations = iterations;
    }

    public static int getnChannels() {
        return nChannels;
    }

    public static void setnChannels(int nChannels) {
        BaseArchitecture.nChannels = nChannels;
    }

    public static int getnClasses() {
        return nClasses;
    }

    public static void setnClasses(int nClasses) {
        BaseArchitecture.nClasses = nClasses;
    }

    public static int getTrainSize() {
        return trainSize;
    }

    public static void setTrainSize(int trainSize) {
        BaseArchitecture.trainSize = trainSize;
    }
}
