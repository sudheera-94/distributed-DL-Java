package org.DistributedDL.CyclicLr;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

public class test {

    public static void main(String[] args) throws IOException, ParseException {

        String file = "/home/sudheera/Documents/MSc/final_project/s3_upload_jar" +
                "/EvaluativeListenerAccuracyArray_mnist_4epochs.json";
        String xAxis = "learningRate";
        String yAxis = "accuracy";

        JSONParser jsonParser = new JSONParser();

        // Parsing the content of the JSON file
        JSONArray jsonArray = (JSONArray) jsonParser.parse(new FileReader(file));

        // Reading the X and Y elements
        int xValuesCount = jsonArray.size();
        Double[] xValuesList = new Double[xValuesCount];
        Double[] yValuesList = new Double[xValuesCount];
        Iterator<JSONObject> iterator = jsonArray.iterator();
        int i = 0;

        while (iterator.hasNext()){
            JSONObject iterObject = iterator.next();
            xValuesList[i] = (Double) iterObject.get(xAxis);
            yValuesList[i] = (Double) iterObject.get(yAxis);
            i = i+1;
        }

    }

}
