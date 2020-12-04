package org.DistributedDL.examples.mnistCyclicLr;

import org.DistributedDL.StandardArchitectures.LeNet5Architecture;
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.FileWriter;
import java.io.IOException;

import static org.DistributedDL.examples.mnistTraditional.MnistSpark.getDataSetIterator;

public class MnistLrRangeTest {

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        int batchSize = 64;
        int numEpochs = 1;
        int height = 28;
        int width = 28;
        int channels = 1;
        int outputNum = 10;
        int rngseed = 123;
        int evaluationListenerFreq = 200;

        // Loading training data
        System.out.println("Data load and vectorization...");

        // Initialize the training set iterator
        DataSetIterator trainIter = getDataSetIterator("mnist_png/training", rngseed, height, width,
                channels, batchSize, outputNum);

        // Pixel values from 0-255 to 0-1 (min-max scaling)
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        // Initialize the testing set iterator
        DataSetIterator iterTest = getDataSetIterator("mnist_png/testing", rngseed, height, width,
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
        MnistLrRangeTestTrainingConfig leNetLrRangeTestConfig = new MnistLrRangeTestTrainingConfig(batchSize);
        MultiLayerConfiguration conf = leNetLrRangeTestConfig.getArchitecture();

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

        System.out.println("******EVALUATE MODEL******");

        // Create Eval object with 10 possible classes
        Evaluation eval = new Evaluation(outputNum);
        iterTest.reset();

        // Evaluate the network
        while (iterTest.hasNext()) {
            DataSet next = iterTest.next();
            INDArray output = model.output(next.getFeatureMatrix());
            // Compare the Feature Matrix from the model
            // with the labels from the RecordReader
            eval.eval(next.getLabels(), output);
        }

        System.out.println(eval.stats());
        System.out.println(eval.confusionToString());

    }

}
