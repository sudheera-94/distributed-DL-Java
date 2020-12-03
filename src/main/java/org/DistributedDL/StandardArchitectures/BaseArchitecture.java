package org.DistributedDL.StandardArchitectures;

public abstract class BaseArchitecture {

    private int iterations = 1;
    private int nChannels = 1;
    private int nClasses = 10;

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public int getnChannels() {
        return nChannels;
    }

    public void setnChannels(int nChannels) {
        this.nChannels = nChannels;
    }

    public int getnClasses() {
        return nClasses;
    }

    public void setnClasses(int nClasses) {
        this.nClasses = nClasses;
    }
}
