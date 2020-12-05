package org.DistributedDL.examples.cifar10.cifar10CyclicLr;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.callbacks.EvaluationCallback;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.FileWriter;
import java.io.IOException;

import static org.DistributedDL.examples.mnist.mnistTraditional.MnistSpark.getDataSetIterator;

public class Cifar10LrRangeTest {

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        int batchSize = 100;
        int numEpochs = 4;
        int height = 32;
        int width = 32;
        int channels = 3;
        int outputNum = 10;
        int rngseed = 123;
        int evaluationListenerFreq = 200;

        // Loading training data
        System.out.println("Data load and vectorization...");

        // Initialize the training set iterator
        DataSetIterator trainIter = getDataSetIterator("cifar10_png/training", rngseed, height, width,
                channels, batchSize, outputNum);

        // Pixel values from 0-255 to 0-1 (min-max scaling)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        // Initialize the testing set iterator
        DataSetIterator iterTest = getDataSetIterator("cifar10_png/testing", rngseed, height, width,
                channels, batchSize, outputNum);
        scaler.fit(iterTest);
        iterTest.setPreProcessor(scaler);

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage();   //Alternative: new FileStatsStorage(File) - see UIStorageExample
        int listenerFrequency = 1;

        // Network configuration
        Cifar10LrRangeTestTrainingConfig cifar10LrRangeTestConfig =
                new Cifar10LrRangeTestTrainingConfig(batchSize, numEpochs);
        MultiLayerConfiguration conf = cifar10LrRangeTestConfig.getArchitecture();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Creating a JSON Array to store EvaluativeListener accuracy
        JSONArray EvaluativeListenerAccuracyArray = new JSONArray();

        // Setting evaluative listener for get accuracy and stats listener for UI.
        EvaluativeListener evalListener = new EvaluativeListener(iterTest, evaluationListenerFreq);
        EvaluationCallback evalCallback = new EvaluationCallback() {
            @Override
            public void call(EvaluativeListener listener, Model model, long invocationsCount, IEvaluation[] evaluations) {
                // Reading the evaluation listener results as a JSON string
                String evaluationJson = evaluations[0].toJson();
                JSONObject evalJsonObj = new JSONObject(evaluationJson);

                // Getting # testing data used in evaluation listener
                int dataCount = evalJsonObj.getInt("numRowCounter");
                // Getting # true positives
                JSONObject truePositiveObject = evalJsonObj.getJSONObject("truePositives");
                int truePositiveCount = truePositiveObject.getInt("totalCount");
                // Getting the accuracy of the listener
                float accuracy = (float) truePositiveCount / (float) dataCount;

                // Creating a JSON object with accuracy and the iteration
                JSONObject EvaluativeListenerAccuracyObject = new JSONObject();
                EvaluativeListenerAccuracyObject.put("iteration", invocationsCount * evaluationListenerFreq);
                EvaluativeListenerAccuracyObject.put("accuracy", accuracy);
                EvaluativeListenerAccuracyArray.put(EvaluativeListenerAccuracyObject);
            }
        };
        evalListener.setCallback(evalCallback);

        model.setListeners(evalListener,
                new StatsListener(statsStorage, listenerFrequency));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage
        // to be visualized
        uiServer.attach(statsStorage);

        // Training the network
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIter);
            System.out.println("*** Completed epoch " + i + " ***");
        }

        // Getting EvaluativeListenerAccuracyArray into a text file
        try (FileWriter file = new FileWriter("EvaluativeListenerAccuracyArray.json")) {
            file.write(EvaluativeListenerAccuracyArray.toString());
            file.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
