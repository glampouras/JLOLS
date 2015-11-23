/*
 2015, Gerasimos Lampouras
 Based on the implementation of AROW in python by Andreas Vlachos
 */
package uk.ac.ucl.jarow;

// Cost-sensitive multiclass classification with AROW

import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.logging.Level;
import java.util.logging.Logger;

// The instances consist of a dictionary of labels to costs and feature vectors (Huang-style)
public class Main {

    final static String filename = "C:\\Research\\Projects\\DIRIGENT\\Code\\JAROW\\news20.binary";

    public static void main(String[] args) {
        String line;
        ArrayList<Instance> instances = new ArrayList<>();
        try (
                InputStream fis = new FileInputStream(filename);
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader br = new BufferedReader(isr);) {            
            System.out.println("Reading the data");
            while ((line = br.readLine()) != null) {
                String[] details;
                details = line.split(" ");
                
                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                
                if (details[0].equals("-1")) {
                    costs.put("neg", 0.0);
                    costs.put("pos", 1.0);
                } else if (details[0].equals("+1")) {
                    costs.put("neg", 1.0);
                    costs.put("pos", 0.0);
                }
                
                for (int i = 1; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");
                    
                    featureVector.put(feature[0], Double.parseDouble(feature[1]));
                }
                instances.add(new Instance(featureVector, costs));
                //System.out.println(instances.get(instances.size()).getCosts());
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        Collections.shuffle(instances);

        //instances = instances[:100]
        // ORIGINAL EVALUATION
        // Keep some instances to check the performance        
        ArrayList<Instance> testingInstances = new ArrayList(instances.subList(((int) Math.round(instances.size() * 0.75)) + 1, instances.size()));
        ArrayList<Instance> trainingInstances = new ArrayList(instances.subList(0, (int) Math.round(instances.size() * 0.75)));

        System.out.println("training data: " + trainingInstances.size() + " instances");
        //classifier_p.train(trainingInstances, True, True, 10, 10)

        // the last parameter can be set to True if probabilities are needed.
        Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
        JAROW classifier_p = JAROW.trainOpt(trainingInstances, 10, params, 0.1, true, false, 0);

        Double cost = classifier_p.batchPredict(testingInstances);
        Double avgCost = cost/(double)testingInstances.size();
        System.out.println("Avg Cost per instance " + avgCost + " on " + testingInstances.size() + " testing instances");

        //10-FOLD CROSS VALIDATION
        /*for (double f = 0.0; f < 1.0; f += 0.1) {            
            int from = ((int) Math.round(instances.size() * f)) + 1;
            if (from < instances.size()) {
                int to = (int) Math.round(instances.size() * (f + 0.1));
                if (to > instances.size()) {
                    to = instances.size();
                }
                ArrayList<Instance> testingInstances = new ArrayList(instances.subList(from, to));
                ArrayList<Instance> trainingInstances = new ArrayList(instances);
                for (Instance testInstance : testingInstances) {
                    trainingInstances.remove(testInstance);
                }

                System.out.println("training data: " + trainingInstances.size() + " instances");
                //classifier_p.train(trainingInstances, True, True, 10, 10)

                // the last parameter can be set to True if probabilities are needed.
                Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
                JAROW classifier_p = JAROW.trainOpt(trainingInstances, 10, params, 0.1, true, false);

                System.out.println("test data: " + testingInstances.size() + " instances");
                Double cost = classifier_p.batchPredict(testingInstances);
                Double avgCost = cost/(double)testingInstances.size();
                System.out.println("Avg Cost per instance " + avgCost + " on " + testingInstances.size() + " testing instances");
            }
        }*/
        
        /*#avgRatio = classifier_p.batchPredict(testingInstances, True)
        #print "entropy sums: " + str(avgRatio)

        # Save the parameters:
        #print "saving"
        #classifier_p.save(sys.argv[1] + ".arow")    
        #print "done"
        # load again:
        #classifier_new = AROW()
        #print "loading model"
        #classifier_new.load(sys.argv[1] + ".arow")
        #print "done"

        #avgRatio = classifier_new.batchPredict(testingInstances, True)
        #print "entropy sums: " + str(avgRatio)*/
    }

}
