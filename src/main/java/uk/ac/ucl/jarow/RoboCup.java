package uk.ac.ucl.jarow;

import edu.stanford.nlp.mt.metrics.BLEUMetric;
import edu.stanford.nlp.mt.metrics.NISTMetric;
import edu.stanford.nlp.mt.tools.NISTTokenizer;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.util.IStrings;
import edu.stanford.nlp.mt.util.ScoredFeaturizedTranslation;
import edu.stanford.nlp.mt.util.Sequence;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
import similarity_measures.Levenshtein;
import uk.ac.ucl.jdagger.JDAgger;

public class RoboCup {

    static Random r = new Random();

    static ArrayList<String> dictionary = new ArrayList<>();
    static ArrayList<String> argDictionary = new ArrayList<>();
    static HashMap<String, HashMap<String, Integer>> argDictionaryMap = new HashMap<>();
    static ArrayList<String> arguments = new ArrayList<>();
    static ArrayList<String> predicates = new ArrayList<>();
    static ArrayList<MeaningRepresentation> meaningReprs = new ArrayList<>();

    static HashSet<ArrayList<String>> patterns = new HashSet<>();

    final public static String TOKEN_END = "@end@";
    final public static String TOKEN_ARG1 = "@arg1@";
    final public static String TOKEN_ARG2 = "@arg2@";

    static HashMap<String, ArrayList<Double>> NISTScoresPerPredicate = new HashMap<>();
    static HashMap<String, ArrayList<Double>> BLEUScoresPerPredicate = new HashMap<>();
    static HashMap<String, ArrayList<Double>> BLEUSmoothScoresPerPredicate = new HashMap<>();

    static HashMap<String, ArrayList<Double>> unbiasedNISTScoresPerPredicate = new HashMap<>();
    static HashMap<String, ArrayList<Double>> unbiasedBLEUScoresPerPredicate = new HashMap<>();
    static HashMap<String, ArrayList<Double>> unbiasedBLEUSmoothScoresPerPredicate = new HashMap<>();

    static HashMap<String, ArrayList<Double>> oneRefNISTScoresPerPredicate = new HashMap<>();
    static HashMap<String, ArrayList<Double>> oneRefBLEUScoresPerPredicate = new HashMap<>();
    static HashMap<String, ArrayList<Double>> oneRefBLEUSmoothScoresPerPredicate = new HashMap<>();

    static ArrayList<Double> NISTScores = new ArrayList<>();
    static ArrayList<Double> BLEUScores = new ArrayList<>();
    static ArrayList<Double> BLEUSmoothScores = new ArrayList<>();

    static ArrayList<Double> unbiasedNISTScores = new ArrayList<>();
    static ArrayList<Double> unbiasedBLEUScores = new ArrayList<>();
    static ArrayList<Double> unbiasedBLEUSmoothScores = new ArrayList<>();

    static ArrayList<Double> oneRefNISTScores = new ArrayList<>();
    static ArrayList<Double> oneRefBLEUScores = new ArrayList<>();
    static ArrayList<Double> oneRefBLEUSmoothScores = new ArrayList<>();

    public static void main(String[] args) {
        runTestWithJAROW();
    }

    public static void runTestWithJAROW() {
        File dataFolder = new File("robocup_data\\gold\\");
        /*createLists(dataFolder, -1);
         saveLists("robocup_data\\");
         //readLists("robocup_data\\");        
         createTrainingDatasets(new File("robocup_data\\gold\\"), "robocup_data\\goldTrainingData", -1);
                
         nlgTest("robocup_data\\");*/

        if (dataFolder.isDirectory()) {
            for (int f = 0; f < dataFolder.listFiles().length; f++) {
                File file = dataFolder.listFiles()[f];
                createLists(dataFolder, f);
                createTrainingDatasets(new File("robocup_data\\gold\\"), "robocup_data\\goldTrainingData", f);

                for (String predicateStr : predicates) {
                    if (!NISTScoresPerPredicate.containsKey(predicateStr)) {
                        NISTScoresPerPredicate.put(predicateStr, new ArrayList<>());
                    }
                    if (!BLEUScoresPerPredicate.containsKey(predicateStr)) {
                        BLEUScoresPerPredicate.put(predicateStr, new ArrayList<>());
                    }
                    if (!BLEUSmoothScoresPerPredicate.containsKey(predicateStr)) {
                        BLEUSmoothScoresPerPredicate.put(predicateStr, new ArrayList<>());
                    }

                    if (!unbiasedNISTScoresPerPredicate.containsKey(predicateStr)) {
                        unbiasedNISTScoresPerPredicate.put(predicateStr, new ArrayList<>());
                    }
                    if (!unbiasedBLEUScoresPerPredicate.containsKey(predicateStr)) {
                        unbiasedBLEUScoresPerPredicate.put(predicateStr, new ArrayList<>());
                    }
                    if (!unbiasedBLEUSmoothScoresPerPredicate.containsKey(predicateStr)) {
                        unbiasedBLEUSmoothScoresPerPredicate.put(predicateStr, new ArrayList<>());
                    }

                    if (!oneRefNISTScoresPerPredicate.containsKey(predicateStr)) {
                        oneRefNISTScoresPerPredicate.put(predicateStr, new ArrayList<>());
                    }
                    if (!oneRefBLEUScoresPerPredicate.containsKey(predicateStr)) {
                        oneRefBLEUScoresPerPredicate.put(predicateStr, new ArrayList<>());
                    }
                    if (!oneRefBLEUSmoothScoresPerPredicate.containsKey(predicateStr)) {
                        oneRefBLEUSmoothScoresPerPredicate.put(predicateStr, new ArrayList<>());
                    }
                }
                for (String predicateStr : predicates) {
                    //if (predicateStr.startsWith("playmode")) {
                        genTest("robocup_data\\", file, f, predicateStr);
                    //}
                }
                //genTest("robocup_data\\", file, f, "badpass");
            }
        }
        for (String predicateStr : predicates) {
            double avgNISTScores = 0.0;
            double avgBLEUScores = 0.0;
            double avgBLEUSmoothScores = 0.0;

            double avgUnbiasedNISTScores = 0.0;
            double avgUnbiasedBLEUScores = 0.0;
            double avgUnbiasedBLEUSmoothScores = 0.0;

            double avgOneRefNISTScores = 0.0;
            double avgOneRefBLEUScores = 0.0;
            double avgOneRefBLEUSmoothScores = 0.0;

            for (int i = 0; i < NISTScoresPerPredicate.get(predicateStr).size(); i++) {
                avgNISTScores += NISTScoresPerPredicate.get(predicateStr).get(i);
                avgBLEUScores += BLEUScoresPerPredicate.get(predicateStr).get(i);
                avgBLEUSmoothScores += BLEUSmoothScoresPerPredicate.get(predicateStr).get(i);

                avgUnbiasedNISTScores += unbiasedNISTScoresPerPredicate.get(predicateStr).get(i);
                avgUnbiasedBLEUScores += unbiasedBLEUScoresPerPredicate.get(predicateStr).get(i);
                avgUnbiasedBLEUSmoothScores += unbiasedBLEUSmoothScoresPerPredicate.get(predicateStr).get(i);

                avgOneRefNISTScores += oneRefNISTScoresPerPredicate.get(predicateStr).get(i);
                avgOneRefBLEUScores += oneRefBLEUScoresPerPredicate.get(predicateStr).get(i);
                avgOneRefBLEUSmoothScores += oneRefBLEUSmoothScoresPerPredicate.get(predicateStr).get(i);
            }
            avgNISTScores /= (double) NISTScoresPerPredicate.get(predicateStr).size();
            avgBLEUScores /= (double) BLEUScoresPerPredicate.get(predicateStr).size();
            avgBLEUSmoothScores /= (double) BLEUSmoothScoresPerPredicate.get(predicateStr).size();

            avgUnbiasedNISTScores /= (double) unbiasedNISTScoresPerPredicate.get(predicateStr).size();
            avgUnbiasedBLEUScores /= (double) unbiasedBLEUScoresPerPredicate.get(predicateStr).size();
            avgUnbiasedBLEUSmoothScores /= (double) unbiasedBLEUSmoothScoresPerPredicate.get(predicateStr).size();

            avgOneRefNISTScores /= (double) oneRefNISTScoresPerPredicate.get(predicateStr).size();
            avgOneRefBLEUScores /= (double) oneRefBLEUScoresPerPredicate.get(predicateStr).size();
            avgOneRefBLEUSmoothScores /= (double) oneRefBLEUSmoothScoresPerPredicate.get(predicateStr).size();

            System.out.println("============================================================");
            System.out.println("============================================================");
            System.out.println("============================================================");
            System.out.println(predicateStr + " BATCH NIST SCORE:\t" + avgNISTScores);
            System.out.println(predicateStr + " BATCH BLEU SCORE:\t" + avgBLEUScores);
            System.out.println(predicateStr + " BATCH BLEU SMOOTH SCORE:\t" + avgBLEUSmoothScores);
            System.out.println("============================================================");
            System.out.println(predicateStr + " UNBIASED BATCH NIST SCORE:\t" + avgUnbiasedNISTScores);
            System.out.println(predicateStr + " UNBIASED BATCH BLEU SCORE:\t" + avgUnbiasedBLEUScores);
            System.out.println(predicateStr + " UNBIASED BATCH BLEU SMOOTH SCORE:\t" + avgUnbiasedBLEUSmoothScores);
            System.out.println("============================================================");
            System.out.println(predicateStr + " ONEREF BATCH NIST SCORE:\t" + avgOneRefNISTScores);
            System.out.println(predicateStr + " ONEREF BATCH BLEU SCORE:\t" + avgOneRefBLEUScores);
            System.out.println(predicateStr + " ONEREF BATCH BLEU SMOOTH SCORE:\t" + avgOneRefBLEUSmoothScores);
            System.out.println("============================================================");
            System.out.println("============================================================");
            System.out.println("============================================================");
        }
        double avgNISTScores = 0.0;
        double avgBLEUScores = 0.0;
        double avgBLEUSmoothScores = 0.0;

        double avgUnbiasedNISTScores = 0.0;
        double avgUnbiasedBLEUScores = 0.0;
        double avgUnbiasedBLEUSmoothScores = 0.0;

        double avgOneRefNISTScores = 0.0;
        double avgOneRefBLEUScores = 0.0;
        double avgOneRefBLEUSmoothScores = 0.0;

        for (int i = 0; i < NISTScores.size(); i++) {
            avgNISTScores += NISTScores.get(i);
            avgBLEUScores += BLEUScores.get(i);
            avgBLEUSmoothScores += BLEUSmoothScores.get(i);

            avgUnbiasedNISTScores += unbiasedNISTScores.get(i);
            avgUnbiasedBLEUScores += unbiasedBLEUScores.get(i);
            avgUnbiasedBLEUSmoothScores += unbiasedBLEUSmoothScores.get(i);

            avgOneRefNISTScores += oneRefNISTScores.get(i);
            avgOneRefBLEUScores += oneRefBLEUScores.get(i);
            avgOneRefBLEUSmoothScores += oneRefBLEUSmoothScores.get(i);
        }
        avgNISTScores /= (double) NISTScores.size();
        avgBLEUScores /= (double) BLEUScores.size();
        avgBLEUSmoothScores /= (double) BLEUSmoothScores.size();

        avgUnbiasedNISTScores /= (double) unbiasedNISTScores.size();
        avgUnbiasedBLEUScores /= (double) unbiasedBLEUScores.size();
        avgUnbiasedBLEUSmoothScores /= (double) unbiasedBLEUSmoothScores.size();

        avgOneRefNISTScores /= (double) oneRefNISTScores.size();
        avgOneRefBLEUScores /= (double) oneRefBLEUScores.size();
        avgOneRefBLEUSmoothScores /= (double) oneRefBLEUSmoothScores.size();

        System.out.println("============================================================");
        System.out.println("============================================================");
        System.out.println("============================================================");
        System.out.println("BATCH NIST SCORE:\t" + avgNISTScores);
        System.out.println("BATCH BLEU SCORE:\t" + avgBLEUScores);
        System.out.println("BATCH BLEU SMOOTH SCORE:\t" + avgBLEUSmoothScores);
        System.out.println("============================================================");
        System.out.println("UNBIASED BATCH NIST SCORE:\t" + avgUnbiasedNISTScores);
        System.out.println("UNBIASED BATCH BLEU SCORE:\t" + avgUnbiasedBLEUScores);
        System.out.println("UNBIASED BATCH BLEU SMOOTH SCORE:\t" + avgUnbiasedBLEUSmoothScores);
        System.out.println("============================================================");
        System.out.println("ONEREF BATCH NIST SCORE:\t" + avgOneRefNISTScores);
        System.out.println("ONEREF BATCH BLEU SCORE:\t" + avgOneRefBLEUScores);
        System.out.println("ONEREF BATCH BLEU SMOOTH SCORE:\t" + avgOneRefBLEUSmoothScores);
        System.out.println("============================================================");
        System.out.println("============================================================");
        System.out.println("============================================================");
    }

    public static void genTest(String modelPath, File testFile, int excludeFile, String predicateStr) {
        String line;
        ArrayList<Instance> wordInstances = new ArrayList<>();
        ArrayList<Instance> argInstances = new ArrayList<>();

        try (
                InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_words_" + predicateStr + "_excl" + excludeFile);
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader br = new BufferedReader(isr);) {
            System.out.println("Reading the word data");
            while ((line = br.readLine()) != null) {
                String[] details;
                details = line.split(" ");

                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                for (int i = 0; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");

                    if (feature[0].startsWith("feature_")) {
                        featureVector.put(feature[0], Double.parseDouble(feature[1]));
                    } else if (feature[0].startsWith("cost_")) {                        
                        costs.put(feature[0].substring(5), Double.parseDouble(feature[1]));
                    }
                }
                wordInstances.add(new Instance(featureVector, costs));
                //System.out.println(instances.get(instances.size() - 1).getCosts());
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        /*try (
         InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_args_pass_excl" + excludeFile);
         InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
         BufferedReader br = new BufferedReader(isr);) {            
         System.out.println("Reading the arg data");
         while ((line = br.readLine()) != null) {
         String[] details;
         details = line.split(" ");
                
         TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
         TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                
         for (String word : argDictionary) {                    
         costs.put(word, 1.0);
         }
         costs.put(argDictionary.get(Integer.parseInt(details[0])), 0.0);
                
         for (int i = 1; i < details.length; i++) {
         String[] feature;
         feature = details[i].split(":");
                    
         featureVector.put(feature[0], Double.parseDouble(feature[1]));
         }
         argInstances.add(new Instance(featureVector, costs));
         //System.out.println(instances.get(instances.size() - 1).getCosts());
         }
         } catch (FileNotFoundException ex) {
         Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
         } catch (IOException ex) {
         Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
         }*/

        //Collections.shuffle(argInstances);
        //DAGGER USE
        ArrayList<Action> availableActions = new ArrayList();
        for (String word : dictionary) {
            availableActions.add(new Action(word));
        }
        HashMap<ActionSequence, Integer> referencePolicy = getReferencePolicy(new File("robocup_data\\gold\\"), "robocup_data\\goldTrainingData", excludeFile);
        JAROW classifierWords = JDAgger.runStochasticDAgger(wordInstances, meaningReprs, availableActions, referencePolicy, 5, 0.7);
        evaluateGeneration(classifierWords, testFile, predicateStr);
         
        //NO DAGGER USE
        /*JAROW classifierWords = new JAROW();
        Collections.shuffle(wordInstances);
        classifierWords.train(wordInstances, true, true, 10, 0.1, true);
        //Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
        //classifierWords = JAROW.trainOpt(wordInstances, 10, params, 0.1, true, false);
        evaluateGeneration(classifierWords, testFile, predicateStr);*/
    }

    public static void nlgTest(String modelPath, String predicateStr) {
        String line;
        ArrayList<Instance> wordInstances = new ArrayList<>();
        ArrayList<Instance> argInstances = new ArrayList<>();

        try (
                InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_words_" + predicateStr);
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader br = new BufferedReader(isr);) {
            System.out.println("Reading the word data");
            while ((line = br.readLine()) != null) {
                String[] details;
                details = line.split(" ");

                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                for (int i = 0; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");

                    if (feature[0].startsWith("feature_")) {
                        featureVector.put(feature[0], Double.parseDouble(feature[1]));
                    } else if (feature[0].startsWith("cost_")) {                        
                        costs.put(feature[0].substring(5), Double.parseDouble(feature[1]));
                    }
                }
                wordInstances.add(new Instance(featureVector, costs));
                //System.out.println(instances.get(instances.size() - 1).getCosts());
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (
                InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_args_" + predicateStr);
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader br = new BufferedReader(isr);) {
            System.out.println("Reading the arg data");
            while ((line = br.readLine()) != null) {
                String[] details;
                details = line.split(" ");
                
                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                for (int i = 0; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");

                    if (feature[0].startsWith("feature_")) {
                        featureVector.put(feature[0], Double.parseDouble(feature[1]));
                    } else if (feature[0].startsWith("cost_")) {                        
                        costs.put(feature[0].substring(5), Double.parseDouble(feature[1]));
                    }
                }
                argInstances.add(new Instance(featureVector, costs));
                //System.out.println(instances.get(instances.size() - 1).getCosts());
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }

        Collections.shuffle(wordInstances);
        Collections.shuffle(argInstances);

        getGradedTrainingErrorRate(wordInstances, 10.0);
        //getCrossValidatedGradedErrorRate(wordInstances, 10.0);
    }

    public static void generalTest(String predicateStr) {
        String line;
        ArrayList<Instance> instances = new ArrayList<>();
        try (
                InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_" + predicateStr);
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader br = new BufferedReader(isr);) {
            System.out.println("Reading the data");
            while ((line = br.readLine()) != null) {
                String[] details;
                details = line.split(" ");
                
                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                for (int i = 0; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");

                    if (feature[0].startsWith("feature_")) {
                        featureVector.put(feature[0], Double.parseDouble(feature[1]));
                    } else if (feature[0].startsWith("cost_")) {                        
                        costs.put(feature[0].substring(5), Double.parseDouble(feature[1]));
                    }
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
        /*ArrayList<Instance> testingInstances = new ArrayList(instances.subList(((int) Math.round(instances.size() * 0.75)) + 1, instances.size()));
         ArrayList<Instance> trainingInstances = new ArrayList(instances.subList(0, (int) Math.round(instances.size() * 0.75)));

         System.out.println("training data: " + trainingInstances.size() + " instances");
         //classifier_p.train(trainingInstances, True, True, 10, 10)

         // the last parameter can be set to True if probabilities are needed.
         Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
         JAROW classifier_p = JAROW.trainOpt(trainingInstances, 10, params, 0.1, true, false);

         Double cost = classifier_p.batchPredict(testingInstances);
         Double avgCost = cost/(double)testingInstances.size();
         System.out.println("Avg Cost per instance " + avgCost + " on " + testingInstances.size() + " testing instances");*/
        //10-FOLD CROSS VALIDATION
        for (double f = 0.0; f < 1.0; f += 0.1) {
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
                Double avgCost = cost / (double) testingInstances.size();
                System.out.println("Avg Cost per instance " + avgCost + " on " + testingInstances.size() + " testing instances");
            }
        }
    }

    public static void createLists(File dataFolder, int excludeFileID) {
        dictionary = new ArrayList<>();
        argDictionary = new ArrayList<>();
        argDictionaryMap = new HashMap<>();
        arguments = new ArrayList<>();
        predicates = new ArrayList<>();
        patterns = new HashSet<>();
        meaningReprs = new ArrayList<>();

        dictionary.add(RoboCup.TOKEN_END);
        dictionary.add(RoboCup.TOKEN_ARG1);
        dictionary.add(RoboCup.TOKEN_ARG2);
        if (dataFolder.isDirectory()) {
            for (int f = 0; f < dataFolder.listFiles().length; f++) {
                File file = dataFolder.listFiles()[f];
                if (file.isFile()) {
                    Document dom = null;
                    DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
                    try {
                        DocumentBuilder db = dbf.newDocumentBuilder();
                        dom = db.parse(file);
                    } catch (ParserConfigurationException pce) {
                        pce.printStackTrace();
                    } catch (SAXException se) {
                        se.printStackTrace();
                    } catch (IOException ioe) {
                        ioe.printStackTrace();
                    }
                    if (dom != null) {
                        Element docEle = dom.getDocumentElement();
                        NodeList nodeList = docEle.getElementsByTagName("example");
                        if (nodeList != null && nodeList.getLength() > 0) {
                            for (int i = 0; i < nodeList.getLength(); i++) {
                                Element node = (Element) nodeList.item(i);

                                NodeList mrl = node.getElementsByTagName("mrl");
                                String predicate = null;
                                ArrayList<String> args = new ArrayList<>();
                                if (mrl != null && mrl.getLength() > 0) {
                                    String mrNode = ((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase();
                                    mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ");
                                    String[] words = mrNode.split(" ");

                                    for (String word : words) {
                                        if (!word.trim().isEmpty()) {
                                            if (predicate == null) {
                                                predicate = word.trim();
                                            } else if (!args.contains(word.trim())) {
                                                if (predicate.equals("playmode")) {
                                                    predicate += "_" + word.trim().substring(0, word.lastIndexOf("_")).trim();
                                                    args.add(word.trim().substring(word.lastIndexOf("_") + 1).trim());
                                                } else {
                                                    args.add(word.trim());
                                                }
                                            }
                                        }
                                    }
                                }
                                if (f != excludeFileID) {
                                    if (!predicates.contains(predicate) && predicate != null) {
                                        predicates.add(predicate);
                                    }
                                    for (String arg : args) {
                                        if (!arguments.contains(arg)) {
                                            arguments.add(arg);
                                        }
                                        if (!argDictionaryMap.containsKey(arg)) {
                                            argDictionaryMap.put(arg, new HashMap<>());
                                        }
                                    }
                                    meaningReprs.add(new MeaningRepresentation(predicate, arguments));
                                }

                                NodeList nl = node.getElementsByTagName("nl");
                                if (nl != null && nl.getLength() > 0) {
                                    String[] nlWords = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", "").trim().toLowerCase().split(" ");

                                    HashMap<String[], Double> alignments = new HashMap<>();
                                    HashMap<String, String> bestAlignments = new HashMap<>();
                                    //Calculate all alignment similarities
                                    for (String arg : args) {
                                        for (String nlWord : nlWords) {
                                            if (!nlWord.equals(RoboCup.TOKEN_END) && !nlWord.replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                                String[] alignment = new String[2];
                                                alignment[0] = arg;
                                                alignment[1] = nlWord;

                                                Double distance;
                                                if (arg.equals("l") || arg.equals("r")) {
                                                    distance = Math.max(Levenshtein.getSimilarity("pink", nlWord, true, false), Levenshtein.getSimilarity("purple", nlWord, true, false));
                                                } else if (arg.endsWith("1")) {
                                                    distance = Math.max(Levenshtein.getSimilarity(arg, nlWord, true, false), Levenshtein.getSimilarity("goalie", nlWord, true, false) - 0.1);
                                                } else {
                                                    distance = Levenshtein.getSimilarity(arg, nlWord, true, false);
                                                }
                                                alignments.put(alignment, distance);
                                            }
                                        }
                                    }
                                    //Keep only the best for each arguement
                                    for (String arg : args) {
                                        Double max = Double.MIN_VALUE;
                                        String[] bestAlignment = new String[2];
                                        for (String[] alignment : alignments.keySet()) {
                                            if (alignment[0].equals(arg)) {
                                                if (alignments.get(alignment) > max) {
                                                    max = alignments.get(alignment);
                                                    bestAlignment = alignment;
                                                }
                                            }
                                        }
                                        if (max >= 0.3)
                                            bestAlignments.put(bestAlignment[1], bestAlignment[0]);
                                    }

                                    String phrase = "";
                                    String arg = "";
                                    ArrayList<String> nlWordsList = new ArrayList<>();
                                    for (int w = 0; w < nlWords.length; w++) {
                                        if (bestAlignments.keySet().contains(nlWords[w])) {
                                            arg = bestAlignments.get(nlWords[w]);
                                            if (!nlWordsList.isEmpty()) {
                                                if (nlWordsList.get(nlWordsList.size() - 1).equals("the")) {
                                                    nlWordsList.remove(nlWordsList.size() - 1);
                                                    phrase = " the " + phrase;
                                                }
                                            }
                                            phrase += " " + nlWords[w].trim();
                                            if (w + 1 < nlWords.length) {
                                                if (nlWords[w + 1].equals("goalie")
                                                        || nlWords[w + 1].equals("team")) {
                                                    w++;
                                                    phrase += " " + nlWords[w].trim();
                                                }
                                            }
                                        } else {
                                            if (!phrase.isEmpty() && !arg.isEmpty()) {
                                                if (args.indexOf(arg) == 0) {
                                                    nlWordsList.add(RoboCup.TOKEN_ARG1);
                                                } else if (args.indexOf(arg) == 1) {
                                                    nlWordsList.add(RoboCup.TOKEN_ARG2);
                                                }
                                                if (f != excludeFileID) {
                                                    argDictionary.add(phrase.replaceAll("\\s+", " ").trim());
                                                }
                                                phrase = "";
                                                arg = "";
                                            }
                                            if (!nlWords[w].replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                                nlWordsList.add(nlWords[w].trim());
                                            }
                                        }
                                    }
                                    if (!phrase.isEmpty() && !arg.isEmpty()) {
                                        if (args.indexOf(arg) == 0) {
                                            nlWordsList.add(RoboCup.TOKEN_ARG1);
                                        } else if (args.indexOf(arg) == 1) {
                                            nlWordsList.add(RoboCup.TOKEN_ARG2);
                                        }
                                        if (f != excludeFileID) {
                                            argDictionary.add(phrase.replaceAll("\\s+", " ").trim());
                                        }
                                    }

                                    if (f != excludeFileID) {
                                        for (String word : nlWordsList) {
                                            if (!word.trim().isEmpty() && !dictionary.contains(word.trim()) && !word.replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                                dictionary.add(word.trim());
                                            }
                                        }
                                    };
                                    patterns.add(nlWordsList);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public static void saveLists(String writeFolderPath) {
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_dictionary"), "utf-8"))) {
            for (int i = 0; i < dictionary.size(); i++) {
                writer.write(i + ":" + dictionary.get(i) + "\n");
            }
            writer.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_arguments"), "utf-8"))) {
            for (int i = 0; i < arguments.size(); i++) {
                writer.write(i + ":" + arguments.get(i) + "\n");
            }
            writer.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_predicates"), "utf-8"))) {
            for (int i = 0; i < predicates.size(); i++) {
                writer.write(i + ":" + predicates.get(i) + "\n");
            }
            writer.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void readLists(String readFolderPath) {
        String line;
        try (
                InputStream fis = new FileInputStream(readFolderPath + "_dictionary");
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader reader = new BufferedReader(isr);) {

            ArrayList<String> lines = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line.trim());
                }
            }

            dictionary = new ArrayList();
            for (int i = 0; i <= Integer.parseInt(lines.get(lines.size() - 1).split(":")[0]); i++) {
                dictionary.add("");
            }

            for (String l : lines) {
                String[] details;
                details = l.split(":");

                dictionary.set(Integer.parseInt(details[0]), details[1]);
            }
            reader.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (
                InputStream fis = new FileInputStream(readFolderPath + "_arguments");
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader reader = new BufferedReader(isr);) {

            ArrayList<String> lines = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line.trim());
                }
            }

            arguments = new ArrayList();
            for (int i = 0; i <= Integer.parseInt(lines.get(lines.size() - 1).split(":")[0]); i++) {
                arguments.add("");
            }

            for (String l : lines) {
                String[] details;
                details = l.split(":");

                arguments.set(Integer.parseInt(details[0]), details[1]);
            }
            reader.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (
                InputStream fis = new FileInputStream(readFolderPath + "_predicates");
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader reader = new BufferedReader(isr);) {

            ArrayList<String> lines = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line.trim());
                }
            }

            predicates = new ArrayList();
            for (int i = 0; i <= Integer.parseInt(lines.get(lines.size() - 1).split(":")[0]); i++) {
                predicates.add("");
            }

            for (String l : lines) {
                String[] details;
                details = l.split(":");

                predicates.set(Integer.parseInt(details[0]), details[1]);
            }
            reader.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void createTrainingDatasets(File dataFolder, String trainingFilePath, int excludeFileID) {
        if (!dictionary.isEmpty() && !predicates.isEmpty() && !arguments.isEmpty()) {
            HashMap<String, ArrayList<String>> predicateWordTrainingData = new HashMap<>();
            HashMap<String, ArrayList<String>> predicateArgTrainingData = new HashMap<>();
            for (String predicate : predicates) {
                predicateWordTrainingData.put(predicate, new ArrayList<>());
                predicateArgTrainingData.put(predicate, new ArrayList<>());
            }
            if (dataFolder.isDirectory()) {
                for (int f = 0; f < dataFolder.listFiles().length; f++) {
                    if (f != excludeFileID) {
                        File file = dataFolder.listFiles()[f];
                        if (file.isFile()) {
                            Document dom = null;
                            DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
                            try {
                                DocumentBuilder db = dbf.newDocumentBuilder();
                                dom = db.parse(file);
                            } catch (ParserConfigurationException pce) {
                                pce.printStackTrace();
                            } catch (SAXException se) {
                                se.printStackTrace();
                            } catch (IOException ioe) {
                                ioe.printStackTrace();
                            }
                            if (dom != null) {
                                Element docEle = dom.getDocumentElement();

                                NodeList nodeList = docEle.getElementsByTagName("example");
                                if (nodeList != null && nodeList.getLength() > 0) {
                                    for (int i = 0; i < nodeList.getLength(); i++) {
                                        Element node = (Element) nodeList.item(i);

                                        String[] nlWords = null;
                                        String predicate = null;

                                        ArrayList<String> args = new ArrayList<>();
                                        NodeList mrl = node.getElementsByTagName("mrl");
                                        if (mrl != null && mrl.getLength() > 0) {
                                            String mrNode = ((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase();
                                            mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ");
                                            String[] words = mrNode.split(" ");

                                            for (String word : words) {
                                                if (!word.trim().isEmpty()) {
                                                    if (predicate == null) {
                                                        predicate = word.trim();
                                                    } else if (!args.contains(word.trim())) {
                                                        if (predicate.equals("playmode")) {
                                                            predicate += "_" + word.trim().substring(0, word.lastIndexOf("_")).trim();
                                                            args.add(word.trim().substring(word.lastIndexOf("_") + 1).trim());
                                                        } else {
                                                            args.add(word.trim());
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if (!predicates.contains(predicate) && predicate != null) {
                                            predicates.add(predicate);
                                        }
                                        for (String arg : args) {
                                            if (!arguments.contains(arg)) {
                                                arguments.add(arg);
                                            }
                                            if (!argDictionaryMap.containsKey(arg)) {
                                                argDictionaryMap.put(arg, new HashMap<>());
                                            }
                                        }

                                        NodeList nl = node.getElementsByTagName("nl");
                                        if (nl != null && nl.getLength() > 0) {
                                            String nlPhrase = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim() + " " + RoboCup.TOKEN_END;
                                            nlWords = nlPhrase.replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", "").split(" ");
                                        }

                                        HashMap<String[], Double> alignments = new HashMap<>();
                                        HashMap<String, String> bestAlignments = new HashMap<>();
                                        //Calculate all alignment similarities
                                        for (String arg : args) {
                                            for (String nlWord : nlWords) {
                                                if (!nlWord.equals(RoboCup.TOKEN_END) && !nlWord.replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                                    String[] alignment = new String[2];
                                                    alignment[0] = arg;
                                                    alignment[1] = nlWord;

                                                    Double distance;
                                                    if (arg.equals("l") || arg.equals("r")) {
                                                        distance = Math.max(Levenshtein.getSimilarity("pink", nlWord, true, false), Levenshtein.getSimilarity("purple", nlWord, true, false));
                                                    } else if (arg.endsWith("1")) {
                                                        distance = Math.max(Levenshtein.getSimilarity(arg, nlWord, true, false), Levenshtein.getSimilarity("goalie", nlWord, true, false) - 0.1);
                                                    } else {
                                                        distance = Levenshtein.getSimilarity(arg, nlWord, true, false);
                                                    }
                                                    alignments.put(alignment, distance);
                                                }
                                            }
                                        }
                                        //Keep only the best for each arguement
                                        for (String arg : args) {
                                            Double max = Double.MIN_VALUE;
                                            String[] bestAlignment = new String[2];
                                            for (String[] alignment : alignments.keySet()) {
                                                if (alignment[0].equals(arg)) {
                                                    if (alignments.get(alignment) > max) {
                                                        if (!bestAlignments.keySet().contains(alignment[1])) {
                                                            max = alignments.get(alignment);
                                                            bestAlignment = alignment;
                                                        }
                                                    }
                                                }
                                            }
                                            if (max >= 0.3)
                                                bestAlignments.put(bestAlignment[1], bestAlignment[0]);
                                        }

                                        String phrase = "";
                                        String arg = "";
                                        HashMap<String, String> bestPhraseAlignments = new HashMap<>();
                                        ArrayList<String> nlWordsList = new ArrayList<>();
                                        for (int w = 0; w < nlWords.length; w++) {
                                            if (bestAlignments.keySet().contains(nlWords[w])) {
                                                arg = bestAlignments.get(nlWords[w]);
                                                if (!nlWordsList.isEmpty()) {
                                                    if (nlWordsList.get(nlWordsList.size() - 1).equals("the")) {
                                                        nlWordsList.remove(nlWordsList.size() - 1);
                                                        phrase = " the " + phrase;
                                                    }
                                                }
                                                phrase += " " + nlWords[w].trim();
                                                if (w + 1 < nlWords.length) {
                                                    if (nlWords[w + 1].equals("goalie")
                                                            || nlWords[w + 1].equals("team")) {
                                                        w++;
                                                        phrase += " " + nlWords[w].trim();
                                                    }
                                                }
                                            } else {
                                                if (!phrase.isEmpty() && !arg.isEmpty()) {
                                                    if (args.indexOf(arg) == 0) {
                                                        nlWordsList.add(RoboCup.TOKEN_ARG1);
                                                    } else if (args.indexOf(arg) == 1) {
                                                        nlWordsList.add(RoboCup.TOKEN_ARG2);
                                                    }
                                                    phrase = phrase.replaceAll("\\s+", " ").trim();
                                                    argDictionary.add(phrase);
                                                    bestPhraseAlignments.put(arg, phrase);
                                                    phrase = "";
                                                    arg = "";
                                                }
                                                if (!nlWords[w].replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                                    nlWordsList.add(nlWords[w].trim());
                                                }
                                            }
                                        }
                                        if (!phrase.isEmpty() && !arg.isEmpty()) {
                                            if (args.indexOf(arg) == 0) {
                                                nlWordsList.add(RoboCup.TOKEN_ARG1);
                                            } else if (args.indexOf(arg) == 1) {
                                                nlWordsList.add(RoboCup.TOKEN_ARG2);
                                            }
                                            phrase = phrase.replaceAll("\\s+", " ").trim();
                                            argDictionary.add(phrase);
                                            bestPhraseAlignments.put(arg, phrase);
                                        }

                                        boolean arg1toBeMentioned = false;
                                        boolean arg2toBeMentioned = false;
                                        for (String argument : args) {
                                            if (bestPhraseAlignments.keySet().contains(argument)) {
                                                if (args.indexOf(argument) == 0) {
                                                    arg1toBeMentioned = true;
                                                } else if (args.indexOf(argument) == 1) {
                                                    arg2toBeMentioned = true;
                                                }
                                            }
                                        }
                                        /*if (predicate.equals("defense")) {
                                         System.out.println(((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase());
                                         System.out.println(nlWordsList);
                                         }*/
                                        for (int w = 0; w < nlWordsList.size(); w++) {
                                            String wordTrainingVector = createStringWordInstance(nlWordsList, w, arg1toBeMentioned, arg2toBeMentioned);
                                            /*if (predicate.equals("defense")) {
                                             System.out.println(nlWordsList.get(w));
                                             System.out.println(wordTrainingVector);
                                             }*/
                                            if (!wordTrainingVector.isEmpty()) {
                                                predicateWordTrainingData.get(predicate).add(wordTrainingVector);
                                            }
                                            if (nlWordsList.get(w).equals(RoboCup.TOKEN_ARG1)) {
                                                arg1toBeMentioned = false;
                                            } else if (nlWordsList.get(w).equals(RoboCup.TOKEN_ARG2)) {
                                                arg2toBeMentioned = false;
                                            }
                                        }

                                        for (String arguement : bestPhraseAlignments.keySet()) {
                                            String argTrainingVector1 = createArgTrainingVector(arguments.indexOf(arguement), bestPhraseAlignments.get(arguement));
                                            if (!argTrainingVector1.isEmpty()) {
                                                predicateArgTrainingData.get(predicate).add(argTrainingVector1);
                                            }
                                            if (argDictionaryMap.get(arguement).containsKey(bestPhraseAlignments.get(arguement))) {
                                                argDictionaryMap.get(arguement).put(bestPhraseAlignments.get(arguement), argDictionaryMap.get(arguement).get(bestPhraseAlignments.get(arguement)) + 1);
                                            } else {
                                                argDictionaryMap.get(arguement).put(bestPhraseAlignments.get(arguement), 1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (String predicate : predicateWordTrainingData.keySet()) {
                String wordsPath = trainingFilePath + "_words_" + predicate;
                if (excludeFileID != -1) {
                    wordsPath += "_excl" + excludeFileID;
                }
                String argsPath = trainingFilePath + "_args_" + predicate;
                if (excludeFileID != -1) {
                    argsPath += "_excl" + excludeFileID;
                }
                try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(wordsPath), "utf-8"))) {
                    for (String trainingVector : predicateWordTrainingData.get(predicate)) {
                        writer.write(trainingVector + "\n");
                    }
                    writer.close();
                } catch (UnsupportedEncodingException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                }
                try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(argsPath), "utf-8"))) {
                    for (String trainingVector : predicateArgTrainingData.get(predicate)) {
                        writer.write(trainingVector + "\n");
                    }
                    writer.close();
                } catch (UnsupportedEncodingException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
    }

    public static HashMap<ActionSequence, Integer> getReferencePolicy(File dataFolder, String trainingFilePath, int excludeFileID) {
        HashMap<ActionSequence, Integer> referencePolicy = new HashMap<>();

        if (!dictionary.isEmpty() && !predicates.isEmpty() && !arguments.isEmpty()) {
            HashMap<String, ArrayList<String>> predicateWordTrainingData = new HashMap<>();
            HashMap<String, ArrayList<String>> predicateArgTrainingData = new HashMap<>();
            for (String predicate : predicates) {
                predicateWordTrainingData.put(predicate, new ArrayList<>());
                predicateArgTrainingData.put(predicate, new ArrayList<>());
            }
            if (dataFolder.isDirectory()) {
                for (int f = 0; f < dataFolder.listFiles().length; f++) {
                    if (f != excludeFileID) {
                        File file = dataFolder.listFiles()[f];
                        if (file.isFile()) {
                            Document dom = null;
                            DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
                            try {
                                DocumentBuilder db = dbf.newDocumentBuilder();
                                dom = db.parse(file);
                            } catch (ParserConfigurationException pce) {
                                pce.printStackTrace();
                            } catch (SAXException se) {
                                se.printStackTrace();
                            } catch (IOException ioe) {
                                ioe.printStackTrace();
                            }
                            if (dom != null) {
                                Element docEle = dom.getDocumentElement();

                                NodeList nodeList = docEle.getElementsByTagName("example");
                                if (nodeList != null && nodeList.getLength() > 0) {
                                    for (int i = 0; i < nodeList.getLength(); i++) {
                                        Element node = (Element) nodeList.item(i);

                                        String[] nlWords = null;
                                        String predicate = null;

                                        ArrayList<String> args = new ArrayList<>();
                                        NodeList mrl = node.getElementsByTagName("mrl");
                                        if (mrl != null && mrl.getLength() > 0) {
                                            String mrNode = ((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase();
                                            mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ");
                                            String[] words = mrNode.split(" ");

                                            for (String word : words) {
                                                if (!word.trim().isEmpty()) {
                                                    if (predicate == null) {
                                                        predicate = word.trim();
                                                    } else if (!args.contains(word.trim())) {
                                                        if (predicate.equals("playmode")) {
                                                            predicate += "_" + word.trim().substring(0, word.lastIndexOf("_")).trim();
                                                            args.add(word.trim().substring(word.lastIndexOf("_") + 1).trim());
                                                        } else {
                                                            args.add(word.trim());
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if (!predicates.contains(predicate) && predicate != null) {
                                            predicates.add(predicate);
                                        }
                                        for (String arg : args) {
                                            if (!arguments.contains(arg)) {
                                                arguments.add(arg);
                                            }
                                            if (!argDictionaryMap.containsKey(arg)) {
                                                argDictionaryMap.put(arg, new HashMap<>());
                                            }
                                        }

                                        NodeList nl = node.getElementsByTagName("nl");
                                        if (nl != null && nl.getLength() > 0) {
                                            String nlPhrase = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim() + " " + RoboCup.TOKEN_END;
                                            nlWords = nlPhrase.replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", "").split(" ");
                                        }

                                        HashMap<String[], Double> alignments = new HashMap<>();
                                        HashMap<String, String> bestAlignments = new HashMap<>();
                                        //Calculate all alignment similarities
                                        for (String arg : args) {
                                            for (String nlWord : nlWords) {
                                                if (!nlWord.equals(RoboCup.TOKEN_END) && !nlWord.replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                                    String[] alignment = new String[2];
                                                    alignment[0] = arg;
                                                    alignment[1] = nlWord;

                                                    Double distance;
                                                    if (arg.equals("l") || arg.equals("r")) {
                                                        distance = Math.max(Levenshtein.getSimilarity("pink", nlWord, true, false), Levenshtein.getSimilarity("purple", nlWord, true, false));
                                                    } else if (arg.endsWith("1")) {
                                                        distance = Math.max(Levenshtein.getSimilarity(arg, nlWord, true, false), Levenshtein.getSimilarity("goalie", nlWord, true, false) - 0.1);
                                                    } else {
                                                        distance = Levenshtein.getSimilarity(arg, nlWord, true, false);
                                                    }
                                                    alignments.put(alignment, distance);
                                                }
                                            }
                                        }
                                        //Keep only the best for each arguement
                                        for (String arg : args) {
                                            Double max = Double.MIN_VALUE;
                                            String[] bestAlignment = new String[2];
                                            for (String[] alignment : alignments.keySet()) {
                                                if (alignment[0].equals(arg)) {
                                                    if (alignments.get(alignment) > max) {
                                                        if (!bestAlignments.keySet().contains(alignment[1])) {
                                                            max = alignments.get(alignment);
                                                            bestAlignment = alignment;
                                                        }
                                                    }
                                                }
                                            }
                                            if (max >= 0.3)
                                                bestAlignments.put(bestAlignment[1], bestAlignment[0]);
                                        }

                                        String phrase = "";
                                        String arg = "";
                                        HashMap<String, String> bestPhraseAlignments = new HashMap<>();
                                        ArrayList<Action> nlWordsActionList = new ArrayList<>();
                                        for (int w = 0; w < nlWords.length; w++) {
                                            if (bestAlignments.keySet().contains(nlWords[w])) {
                                                arg = bestAlignments.get(nlWords[w]);
                                                if (!nlWordsActionList.isEmpty()) {
                                                    if (nlWordsActionList.get(nlWordsActionList.size() - 1).equals("the")) {
                                                        nlWordsActionList.remove(nlWordsActionList.size() - 1);
                                                        phrase = " the " + phrase;
                                                    }
                                                }
                                                phrase += " " + nlWords[w].trim();
                                                if (w + 1 < nlWords.length) {
                                                    if (nlWords[w + 1].equals("goalie")
                                                            || nlWords[w + 1].equals("team")) {
                                                        w++;
                                                        phrase += " " + nlWords[w].trim();
                                                    }
                                                }
                                            } else {
                                                if (!phrase.isEmpty() && !arg.isEmpty()) {
                                                    if (args.indexOf(arg) == 0) {
                                                        nlWordsActionList.add(new Action(RoboCup.TOKEN_ARG1));
                                                    } else if (args.indexOf(arg) == 1) {
                                                        nlWordsActionList.add(new Action(RoboCup.TOKEN_ARG2));
                                                    }
                                                    phrase = phrase.replaceAll("\\s+", " ").trim();
                                                    argDictionary.add(phrase);
                                                    bestPhraseAlignments.put(arg, phrase);
                                                    phrase = "";
                                                    arg = "";
                                                }
                                                if (!nlWords[w].replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                                    nlWordsActionList.add(new Action(nlWords[w].trim()));
                                                }
                                            }
                                        }
                                        if (!phrase.isEmpty() && !arg.isEmpty()) {
                                            if (args.indexOf(arg) == 0) {
                                                nlWordsActionList.add(new Action(RoboCup.TOKEN_ARG1));
                                            } else if (args.indexOf(arg) == 1) {
                                                nlWordsActionList.add(new Action(RoboCup.TOKEN_ARG2));
                                            }
                                            phrase = phrase.replaceAll("\\s+", " ").trim();
                                            argDictionary.add(phrase);
                                            bestPhraseAlignments.put(arg, phrase);
                                        }
                                        ActionSequence as = new ActionSequence(nlWordsActionList, 0.0);
                                        if (!referencePolicy.containsKey(as)) {
                                            referencePolicy.put(as, 1);
                                        } else {
                                            referencePolicy.put(as, referencePolicy.get(as) + 1);
                                        }

                                        for (String arguement : args) {
                                            /*String argTrainingVector1 = createArgTrainingVector(arguments.indexOf(arguement), bestPhraseAlignments.get(arguement));
                                            if (!argTrainingVector1.isEmpty()) {
                                                predicateArgTrainingData.get(predicate).add(argTrainingVector1);
                                            }*/
                                            if (argDictionaryMap.get(arguement).containsKey(bestPhraseAlignments.get(arguement))) {
                                                argDictionaryMap.get(arguement).put(bestPhraseAlignments.get(arguement), argDictionaryMap.get(arguement).get(bestPhraseAlignments.get(arguement)) + 1);
                                            } else {
                                                argDictionaryMap.get(arguement).put(bestPhraseAlignments.get(arguement), 1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return referencePolicy;
    }

    public static String createStringWordInstance(ArrayList<String> nlWords, int w, boolean arg1toBeMentioned, boolean arg2toBeMentioned) {
        String trainingVector = "";

        String bestAction = nlWords.get(w).toLowerCase().trim();
        int featureNo = 1;
        if (!bestAction.isEmpty()) {
            //COSTS
            for (String action : dictionary) {
                if (action.equals(bestAction)) {
                    trainingVector += " " + "cost_" + action + ":0.0";
                } else {
                    trainingVector += " " + "cost_" + action + ":1.0";
                }
            }
            
            //FEATURES
            //Arg1 ID
            /*for (int i = 0; i < arguments.size(); i++) {
             int featureValue = 0;
             if (i == arg1ID) {
             featureValue = 1;
             }
             trainingVector += " " + "feature_" + (featureNo++) + ":" + featureValue;
             }
             //Arg2 ID
             for (int i = 0; i < arguments.size(); i++) {
             int featureValue = 0;
             if (i == arg2ID) {
             featureValue = 1;
             }
             trainingVector += " " + "feature_" + (featureNo++) + ":" + featureValue;
             }*/
            //Previous word features
            for (int j = 1; j <= 5; j++) {
                /*int previousWordID = -1;
                 if (w - j >= 0) {
                 String previousWord = nlWords[w - j].trim();
                 previousWordID = dictionary.indexOf(previousWord);
                 }*/
                String previousWord = "";
                if (w - j >= 0) {
                    previousWord = nlWords.get(w - j).trim();
                }
                for (int i = 0; i < dictionary.size(); i++) {
                    int featureValue = 0;
                    if (!previousWord.isEmpty() && dictionary.get(i).equals(previousWord)) {
                        featureValue = 1;
                    }
                    trainingVector += " " + "feature_" + (featureNo++) + ":" + featureValue;
                }
                if (previousWord.isEmpty()) {
                    trainingVector += " " + "feature_" + (featureNo++) + ":1";
                } else {
                    trainingVector += " " + "feature_" + (featureNo++) + ":0";
                }
            }
            //Word Positions
            //trainingVector += " " + "feature_" + (featureNo++) + ":" + w/20;
            //If arguments have already been generated or not
            if (arg1toBeMentioned) {
                trainingVector += " " + "feature_" + (featureNo++) + ":1";
            } else {
                trainingVector += " " + "feature_" + (featureNo++) + ":0";
            }
            if (arg2toBeMentioned) {
                trainingVector += " " + "feature_" + (featureNo++) + ":1";
            } else {
                trainingVector += " " + "feature_" + (featureNo++) + ":0";
            }
        }
        return trainingVector;
    }
    
    public static Instance createWordInstance(ArrayList<String> nlWords, int w, boolean arg1toBeMentioned, boolean arg2toBeMentioned) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        String bestAction = nlWords.get(w).toLowerCase().trim();
        if (!bestAction.isEmpty()) {
            //COSTS
            for (String action : dictionary) {
                if (action.equals(bestAction)) {
                    costs.put(action, 0.0);
                } else {
                    costs.put(action, 1.0);
                }
            }
        }
        return createWordInstance(nlWords, w, costs, arg1toBeMentioned, arg2toBeMentioned);
    }
    
    public static Instance createWordInstance(ArrayList<String> nlWords, int w, double cost, boolean arg1toBeMentioned, boolean arg2toBeMentioned) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        String bestAction = nlWords.get(w).toLowerCase().trim();
        if (!bestAction.isEmpty()) {
            //COSTS
            for (String action : dictionary) {
                if (action.equals(bestAction)) {
                    costs.put(action, cost);
                } else {
                    costs.put(action, 1.0);
                }
            }
        }
        return createWordInstance(nlWords, w, costs, arg1toBeMentioned, arg2toBeMentioned);
    }
    
    public static Instance createWordInstance(ArrayList<String> nlWords, int w, TObjectDoubleHashMap<String> costs, boolean arg1toBeMentioned, boolean arg2toBeMentioned) {
        TObjectDoubleHashMap<String> features = new TObjectDoubleHashMap<>();

        int featureNo = 0;            
        //Previous word features
        for (int j = 1; j <= 5; j++) {
            /*int previousWordID = -1;
             if (w - j >= 0) {
             String previousWord = nlWords[w - j].trim();
             previousWordID = dictionary.indexOf(previousWord);
             }*/
            String previousWord = "";
            if (w - j >= 0) {
                previousWord = nlWords.get(w - j).trim();
            }
            for (int i = 0; i < dictionary.size(); i++) {
                int featureValue = 0;
                if (!previousWord.isEmpty() && dictionary.get(i).equals(previousWord)) {
                    featureValue = 1;
                }
                features.put("feature_" + (featureNo++), featureValue);
            }
            if (previousWord.isEmpty()) {
                features.put("feature_" + (featureNo++), 1.0);
            } else {
                features.put("feature_" + (featureNo++), 0.0);
            }
        }
        //Word Positions
        //trainingVector += " " + "feature_" + (featureNo++) + ":" + w/20;
        //If arguments have already been generated or not
        if (arg1toBeMentioned) {
            features.put("feature_" + (featureNo++), 1.0);
        } else {
            features.put("feature_" + (featureNo++), 0.0);
        }
        if (arg2toBeMentioned) {
            features.put("feature_" + (featureNo++), 1.0);
        } else {
            features.put("feature_" + (featureNo++), 0.0);
        }
        return new Instance(features, costs);
    }

    public static String createArgTrainingVector(int argID, String bestAction) {
        String trainingVector = "";

        int featureNo = 1;
        if (!bestAction.isEmpty()) {
            //COSTS
            for (String action : dictionary) {
                if (action.equals(bestAction)) {
                    trainingVector += " " + "cost_" + action + ":0.0";
                } else {
                    trainingVector += " " + "cost_" + action + ":1.0";
                }
            }
            //Arg ID
            for (int i = 0; i < arguments.size(); i++) {
                int featureValue = 0;
                if (i == argID) {
                    featureValue = 1;
                }
                trainingVector += " " + "feature_" + (featureNo++) + ":" + featureValue;
            }
        }
        return trainingVector;
    }

    public static void getGradedTrainingErrorRate(ArrayList<Instance> instances, Double param) {
        System.out.println("training data: " + instances.size() + " instances");

        for (double g = 0.1; g < 1.0; g += 0.1) {
            int grade = (int) Math.round(instances.size() * (g + 0.1));
            if (grade > instances.size()) {
                grade = instances.size();
            }
            ArrayList<Instance> gradedTrainingInstances = new ArrayList(instances.subList(0, grade));

            // the last parameter can be set to True if probabilities are needed.
            JAROW classifierGraded = new JAROW();
            classifierGraded.train(gradedTrainingInstances, true, false, 10, param, true);

            System.out.println("test data: " + gradedTrainingInstances.size() + " instances");
            int errors = 0;
            for (Instance instance : gradedTrainingInstances) {
                Prediction predict = classifierGraded.predict(instance);
                if (!instance.getCorrectLabels().contains(predict.getLabel())) {
                    errors++;
                }
            }
            Double errorRate = errors / (double) gradedTrainingInstances.size();
            System.out.println("Training error rate of training on\t" + g + "\tof training data is:\t" + errorRate);
        }
    }

    public static void getCrossValidatedGradedErrorRate(ArrayList<Instance> instances, Double param) {
        //10-FOLD STAGED CROSS VALIDATION ON WORDS
        for (double f = 0.0; f < 1.0; f += 0.1) {
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

                for (double g = 0.1; g < 1.0; g += 0.1) {
                    int grade = (int) Math.round(instances.size() * (g + 0.1));
                    if (grade > instances.size()) {
                        grade = instances.size();
                    }
                    ArrayList<Instance> gradedTrainingInstances = new ArrayList(instances.subList(0, grade));

                    // the last parameter can be set to True if probabilities are needed.
                    JAROW classifierGraded = new JAROW();
                    classifierGraded.train(gradedTrainingInstances, true, false, 10, param, true);

                    System.out.println("test data: " + testingInstances.size() + " instances");
                    int errors = 0;
                    for (Instance instance : testingInstances) {
                        Prediction predict = classifierGraded.predict(instance);
                        if (!instance.getCorrectLabels().contains(predict.getLabel())) {
                            errors++;
                        }
                    }
                    Double errorRate = errors / (double) testingInstances.size();
                    System.out.println("Testing error rate of training on\t" + g + "\tof training data is:\t" + errorRate);
                }
            }
        }
    }

    public static void evaluateGeneration(JAROW classifierWords, /*ArrayList<Instance> trainingArgInstances, */ File testFile, String predicateStr) {
        NISTTokenizer.lowercase(true);
        NISTTokenizer.normalize(true);

        ArrayList<String> mrNodes = new ArrayList<>();
        HashMap<String, ArrayList<Sequence<IString>>> references = new HashMap<>();
        HashMap<String, ArrayList<String>> strReferences = new HashMap<>();
        ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();

        Document dom = null;
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        try {
            DocumentBuilder db = dbf.newDocumentBuilder();
            dom = db.parse(testFile);
        } catch (ParserConfigurationException pce) {
            pce.printStackTrace();
        } catch (SAXException se) {
            se.printStackTrace();
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
        if (dom != null) {
            Element docEle = dom.getDocumentElement();

            NodeList nodeList = docEle.getElementsByTagName("example");
            if (nodeList != null && nodeList.getLength() > 0) {
                for (int i = 0; i < nodeList.getLength(); i++) {
                    Element node = (Element) nodeList.item(i);
                    NodeList nl = node.getElementsByTagName("nl");
                    if (nl != null && nl.getLength() > 0) {
                        String[] nlWords = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim().replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", "").toLowerCase().split(" ");

                        String cleanedWords = "";
                        for (String nlWord : nlWords) {
                            if (!nlWord.replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                cleanedWords += nlWord + " ";
                            }
                        }
                        Sequence<IString> reference = IStrings.tokenize(NISTTokenizer.tokenize(cleanedWords.trim()));

                        String predicate = null;
                        String arg1 = null;
                        NodeList mrl = node.getElementsByTagName("mrl");
                        if (mrl != null && mrl.getLength() > 0) {
                            String mrNode = ((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase();
                            mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ").replaceAll("\\s+", " ");
                            String[] words = mrNode.split(" ");

                            for (String word : words) {
                                if (!word.trim().isEmpty()) {
                                    if (predicate == null) {
                                        predicate = word.trim();
                                    } else if (arg1 == null) {
                                        if (predicate.equals("playmode")) {
                                            predicate += "_" + word.trim().substring(0, word.lastIndexOf("_")).trim();
                                            arg1 = word.trim().substring(word.lastIndexOf("_") + 1).trim();
                                        } else {
                                            arg1 = word.trim();
                                        }
                                    }
                                }
                            }

                            if (predicate != null && predicate.equals(predicateStr)) {
                                mrNodes.add(mrNode);
                                if (references.containsKey(mrNode)) {
                                    references.get(mrNode).add(reference);
                                    strReferences.get(mrNode).add((((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim()).replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", ""));
                                } else {
                                    references.put(mrNode, new ArrayList<>());
                                    references.get(mrNode).add(reference);

                                    strReferences.put(mrNode, new ArrayList<>());
                                    strReferences.get(mrNode).add((((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim()).replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", ""));
                                }
                            }
                        }
                    }
                }

                ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
                for (String mrNode : mrNodes) {
                    finalReferences.add(references.get(mrNode));
                }
                NISTMetric NIST = new NISTMetric(finalReferences);
                //IncrementalEvaluationMetric<IString,String> NISTinc = NIST.getIncrementalMetric();
                BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
                //IncrementalEvaluationMetric<IString,String> BLEUinc = BLEU.getIncrementalMetric();
                BLEUMetric BLEUsmooth = new BLEUMetric(finalReferences, 4, true);
                //IncrementalEvaluationMetric<IString,String> BLEUincSmooth = BLEUsmooth.getIncrementalMetric();

                //double totalUnbiasedBLEU = 0.0;
                //double totalOneRefBLEU = 0.0;
                ArrayList<ArrayList<Sequence<IString>>> unbiasedFinalReferences = new ArrayList<>();
                ArrayList<ArrayList<Sequence<IString>>> oneRefFinalReferences = new ArrayList<>();
                for (int i = 0; i < nodeList.getLength(); i++) {
                    Element node = (Element) nodeList.item(i);

                    String predicate = null;
                    NodeList nl = node.getElementsByTagName("nl");
                    if (nl != null && nl.getLength() > 0) {
                        ArrayList<String> args = new ArrayList<>();
                        NodeList mrl = node.getElementsByTagName("mrl");
                        String mrNode = "";
                        if (mrl != null && mrl.getLength() > 0) {
                            mrNode = ((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase();
                            mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ");
                            String[] words = mrNode.split(" ");

                            for (String word : words) {
                                if (!word.trim().isEmpty()) {
                                    if (predicate == null) {
                                        predicate = word.trim();
                                    } else if (!args.contains(word.trim())) {
                                        if (predicate.equals("playmode")) {
                                            predicate += "_" + word.trim().substring(0, word.lastIndexOf("_")).trim();
                                            args.add(word.trim().substring(word.lastIndexOf("_") + 1).trim());
                                        } else {
                                            args.add(word.trim());
                                        }
                                    }
                                }
                            }
                        }

                        if (predicate != null && predicate.equals(predicateStr)) {
                            //PHRASE GENERATION EVALUATION
                            String predictedWord = "";
                            int w = 0;
                            ArrayList<String> predictedWordsList = new ArrayList<>();
                            boolean arg1toBeMentioned = false;
                            boolean arg2toBeMentioned = false;
                            for (String argument : args) {
                                if (args.indexOf(argument) == 0) {
                                    arg1toBeMentioned = true;
                                } else if (args.indexOf(argument) == 1) {
                                    arg2toBeMentioned = true;
                                }
                            }
                            while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10000) {
                                ArrayList<String> tempList = new ArrayList(predictedWordsList);
                                tempList.add("@TOK@");
                                Instance trainingVector = RoboCup.createWordInstance(tempList, w, arg1toBeMentioned, arg2toBeMentioned);

                                if (trainingVector != null) {
                                    Prediction predict = classifierWords.predict(trainingVector);
                                    predictedWord = predict.getLabel().trim();
                                    predictedWordsList.add(predictedWord);

                                    if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                                        arg1toBeMentioned = false;
                                    } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                                        arg2toBeMentioned = false;
                                    }
                                }
                                w++;
                            }

                            String predictedString = "";
                            for (String word : predictedWordsList) {
                                predictedString += word + " ";
                            }
                            predictedString = predictedString.trim();

                            /*for (int p = 0; p < predictedWordsList.size(); p++) {
                             predictedWord = predictedWordsList.get(p);
                             if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                             String argTrainingVector = createArgTrainingVector(arg1ID, arg1);
                             if (!argTrainingVector.isEmpty()) {
                             TObjectDoubleHashMap<String> argFeatureVector = new TObjectDoubleHashMap<>();
                             TObjectDoubleHashMap<String> argCosts = new TObjectDoubleHashMap<>();

                             String[] argDetails;
                             argDetails = argTrainingVector.split(" ");

                             for (int j = 1; j < argDetails.length; j++) {
                             String[] feature;
                             feature = argDetails[j].split(":");

                             argFeatureVector.put(feature[0], Double.parseDouble(feature[1]));
                             }        

                             Prediction argPredict = classifierArgs.predict(new Instance(argFeatureVector, argCosts));
                             predictedWord = argPredict.getLabel().trim();
                             predictedWordsList.set(p, predictedWord);
                             }
                             } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                             String argTrainingVector = createArgTrainingVector(arg2ID, arg2);
                             if (!argTrainingVector.isEmpty()) {
                             TObjectDoubleHashMap<String> argFeatureVector = new TObjectDoubleHashMap<>();
                             TObjectDoubleHashMap<String> argCosts = new TObjectDoubleHashMap<>();

                             String[] argDetails;
                             argDetails = argTrainingVector.split(" ");

                             for (int j = 1; j < argDetails.length; j++) {
                             String[] feature;
                             feature = argDetails[j].split(":");

                             argFeatureVector.put(feature[0], Double.parseDouble(feature[1]));
                             }        

                             Prediction argPredict = classifierArgs.predict(new Instance(argFeatureVector, argCosts));
                             predictedWord = argPredict.getLabel().trim();
                             predictedWordsList.set(p, predictedWord);
                             }
                             }
                             }

                             String predictedStringWithArgs = "";
                             for (String word : predictedWordsList) {
                             predictedStringWithArgs += word + " ";
                             }*/
                            String arg1name = "";
                            String arg2name = "";
                            for (int p = 0; p < predictedWordsList.size(); p++) {
                                predictedWord = predictedWordsList.get(p);
                                switch (predictedWord) {
                                    case RoboCup.TOKEN_ARG1: {
                                        if (argDictionaryMap.containsKey(args.get(0))) {
                                            if (!argDictionaryMap.get(args.get(0)).isEmpty()) {
                                                int max = 0;
                                                for (String n : argDictionaryMap.get(args.get(0)).keySet()) {
                                                    int freq = argDictionaryMap.get(args.get(0)).get(n);
                                                    if (freq > max) {
                                                        max = freq;
                                                        arg1name = n;
                                                    }
                                                }
                                            }
                                        }
                                        if (arg1name.isEmpty() && args.get(0) != null) {
                                            arg1name = args.get(0).trim();
                                        }
                                        predictedWordsList.set(p, arg1name);
                                        break;
                                    }
                                    case RoboCup.TOKEN_ARG2: {
                                        if (argDictionaryMap.containsKey(args.get(1))) {
                                            if (!argDictionaryMap.get(args.get(1)).isEmpty()) {
                                                int max = 0;
                                                for (String n : argDictionaryMap.get(args.get(1)).keySet()) {
                                                    int freq = argDictionaryMap.get(args.get(1)).get(n);
                                                    if (freq > max) {
                                                        max = freq;
                                                        arg2name = n;
                                                    }
                                                }
                                            }
                                        }
                                        if (arg2name.isEmpty() && args.get(1) != null) {
                                            arg2name = args.get(1).trim();
                                        }
                                        predictedWordsList.set(p, arg2name);
                                        break;
                                    }
                                }
                            }

                            String predictedStringWithArgs = "";
                            //ArrayList<IString> seqList = new ArrayList<>();
                            for (String word : predictedWordsList) {
                                if (!word.equals(RoboCup.TOKEN_END) && !word.isEmpty()) {
                                    predictedStringWithArgs += word + " ";
                                    //seqList.add(new IString(word));
                                }
                            }
                            predictedStringWithArgs = predictedStringWithArgs.trim();

                            //SimpleSequence<IString> seq = new SimpleSequence<>(seqList);                            
                            //generations.add(seq);
                            Double unbiasedBLEUScore = BLEUMetric.computeLocalSmoothScore(predictedStringWithArgs, createUnbiasedReferenceList(arg1name, arg2name), 4);
                            //totalUnbiasedBLEU += unbiasedBLEUScore;

                            unbiasedFinalReferences.add(createUnbiasedReferenceListSeq(arg1name, arg2name));

                            ArrayList<Sequence<IString>> oneRef = new ArrayList<>();
                            String[] nlWords = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim().replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", "").toLowerCase().split(" ");
                            String cleanedWords = "";
                            for (String nlWord : nlWords) {
                                if (!nlWord.replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                    cleanedWords += nlWord + " ";
                                }
                            }
                            oneRef.add(IStrings.tokenize(NISTTokenizer.tokenize(cleanedWords.trim())));
                            oneRefFinalReferences.add(oneRef);
                            //Double oneRefBLEUScore = BLEUMetric.computeLocalSmoothScore(predictedStringWithArgs, oneRef, 4);
                            //totalOneRefBLEU += oneRefBLEUScore;

                            //total++;
                            Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedStringWithArgs));
                            ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
                            generations.add(tran);
                            //NISTinc.add(tran);
                            //BLEUinc.add(tran);
                            //BLEUincSmooth.add(tran);

                            //if (unbiasedBLEUScore < 1.0) {
                            System.out.println("M: " + ((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase());
                            System.out.println("T: " + ((Element) nl.item(0)).getFirstChild().getNodeValue().trim().toLowerCase());
                            System.out.println("P: " + predictedString);
                            System.out.println("A: " + predictedStringWithArgs);
                            //System.out.println("REFS: " + createUnbiasedReferenceList(arg1name, arg2name));
                            System.out.println("BLEU: " + unbiasedBLEUScore);
                            System.out.println("==============");
                            //}

                        }
                    }
                }
                if (generations.size() > 0) {
                    System.out.println("PREDICATE:\t" + predicateStr);

                    System.out.println(generations);
                    System.out.println("==============");
                    System.out.println(unbiasedFinalReferences);
                    System.out.println("==============");
                    Double nistScore = NIST.score(generations);
                    Double bleuScore = BLEU.score(generations);
                    Double bleuSmoothScore = BLEUsmooth.score(generations);
                    //System.out.println("INC NIST SCORE:\t" + NISTinc.score());
                    System.out.println("BATCH NIST SCORE:\t" + nistScore);
                    //System.out.println("INC BLEU SCORE:\t" + BLEUinc.score());
                    System.out.println("BATCH BLEU SCORE:\t" + bleuScore);
                    //System.out.println("INC BLEU SMOOTH SCORE:\t" + BLEUincSmooth.score());
                    System.out.println("BATCH BLEU SMOOTH SCORE:\t" + bleuSmoothScore);
                    //System.out.println("SINGLE-REF BLEU SMOOTH SCORE:\t" + totalOneRefBLEU/total);
                    //System.out.println("UNBIASED BLEU SCORE:\t" + totalUnbiasedBLEU/total);
                    if (!nistScore.isNaN()) {
                        NISTScores.add(nistScore);
                    }
                    if (!bleuScore.isNaN()) {
                        BLEUScores.add(bleuScore);
                    }
                    if (!bleuSmoothScore.isNaN()) {
                        BLEUSmoothScores.add(bleuSmoothScore);
                    }
                    //System.out.println(NISTScoresPerPredicate.keySet());
                    if (!nistScore.isNaN()) {
                        NISTScoresPerPredicate.get(predicateStr).add(nistScore);
                    }
                    if (!bleuScore.isNaN()) {
                        BLEUScoresPerPredicate.get(predicateStr).add(bleuScore);
                    }
                    if (!bleuSmoothScore.isNaN()) {
                        BLEUSmoothScoresPerPredicate.get(predicateStr).add(bleuSmoothScore);
                    }

                    //System.out.println(generations.size() + " == " + unbiasedFinalReferences.size());
                    NISTMetric unbiasedNIST = new NISTMetric(unbiasedFinalReferences);
                    BLEUMetric unbiasedBLEU = new BLEUMetric(unbiasedFinalReferences, 4, false);
                    BLEUMetric unbiasedBLEUsmooth = new BLEUMetric(unbiasedFinalReferences, 4, true);
                    Double unbiasedNistScore = unbiasedNIST.score(generations);
                    Double unbiasedBleuScore = unbiasedBLEU.score(generations);
                    Double unbiasedBleuSmoothScore = unbiasedBLEUsmooth.score(generations);

                    System.out.println("UNBIASED BATCH NIST SCORE:\t" + unbiasedNistScore);
                    System.out.println("UNBIASED BATCH BLEU SCORE:\t" + unbiasedBleuScore);
                    System.out.println("UNBIASED BATCH BLEU SMOOTH SCORE:\t" + unbiasedBleuSmoothScore);
                    if (!unbiasedNistScore.isNaN()) {
                        unbiasedNISTScores.add(unbiasedNistScore);
                    }
                    if (!unbiasedBleuScore.isNaN()) {
                        unbiasedBLEUScores.add(unbiasedBleuScore);
                    }
                    if (!unbiasedBleuSmoothScore.isNaN()) {
                        unbiasedBLEUSmoothScores.add(unbiasedBleuSmoothScore);
                    }
                    if (!unbiasedNistScore.isNaN()) {
                        unbiasedNISTScoresPerPredicate.get(predicateStr).add(unbiasedNistScore);
                    }
                    if (!unbiasedBleuScore.isNaN()) {
                        unbiasedBLEUScoresPerPredicate.get(predicateStr).add(unbiasedBleuScore);
                    }
                    if (!unbiasedBleuSmoothScore.isNaN()) {
                        unbiasedBLEUSmoothScoresPerPredicate.get(predicateStr).add(unbiasedBleuSmoothScore);
                    }

                    //System.out.println(generations.size() + " == " + oneRefFinalReferences.size());
                    NISTMetric oneRefNIST = new NISTMetric(oneRefFinalReferences);
                    BLEUMetric oneRefBLEU = new BLEUMetric(oneRefFinalReferences, 4, false);
                    BLEUMetric oneRefBLEUsmooth = new BLEUMetric(oneRefFinalReferences, 4, true);
                    Double oneRefNistScore = oneRefNIST.score(generations);
                    Double oneRefBleuScore = oneRefBLEU.score(generations);
                    Double oneRefBleuSmoothScore = oneRefBLEUsmooth.score(generations);

                    System.out.println("ONE REF BATCH NIST SCORE:\t" + oneRefNistScore);
                    System.out.println("ONE REF BATCH BLEU SCORE:\t" + oneRefBleuScore);
                    System.out.println("ONE REF BATCH BLEU SMOOTH SCORE:\t" + oneRefBleuSmoothScore);
                    if (!oneRefNistScore.isNaN()) {
                        oneRefNISTScores.add(oneRefNistScore);
                    }
                    if (!oneRefBleuScore.isNaN()) {
                        oneRefBLEUScores.add(oneRefBleuScore);
                    }
                    if (!oneRefBleuSmoothScore.isNaN()) {
                        oneRefBLEUSmoothScores.add(oneRefBleuSmoothScore);
                    }
                    if (!oneRefNistScore.isNaN()) {
                        oneRefNISTScoresPerPredicate.get(predicateStr).add(oneRefNistScore);
                    }
                    if (!oneRefBleuScore.isNaN()) {
                        oneRefBLEUScoresPerPredicate.get(predicateStr).add(oneRefBleuScore);
                    }
                    if (!oneRefBleuSmoothScore.isNaN()) {
                        oneRefBLEUSmoothScoresPerPredicate.get(predicateStr).add(oneRefBleuSmoothScore);
                    }
                }
            }
        }
    }

    public static ArrayList<String> createUnbiasedReferenceList() {
        ArrayList<String> references = new ArrayList<>();
        for (ArrayList<String> pattern : patterns) {
            String reference = "";
            for (String word : pattern) {
                reference += word + " ";
            }
            references.add(reference.trim());
        }
        return references;
    }

    public static ArrayList<String> createUnbiasedReferenceList(String arg1word, String arg2word) {
        ArrayList<String> references = new ArrayList<>();
        for (ArrayList<String> pattern : patterns) {
            String reference = "";
            for (String word : pattern) {
                if (word.equals(RoboCup.TOKEN_ARG1)) {
                    reference += arg1word + " ";
                } else if (word.equals(RoboCup.TOKEN_ARG2)) {
                    reference += arg2word + " ";
                } else if (!word.equals(RoboCup.TOKEN_END)) {
                    reference += word + " ";
                }
            }
            references.add(reference.trim());
        }
        return references;
    }

    public static ArrayList<Sequence<IString>> createUnbiasedReferenceListSeq(String arg1word, String arg2word) {
        ArrayList<Sequence<IString>> references = new ArrayList<>();
        for (ArrayList<String> pattern : patterns) {
            String reference = "";
            for (String word : pattern) {
                if (word.equals(RoboCup.TOKEN_ARG1)) {
                    reference += arg1word + " ";
                } else if (word.equals(RoboCup.TOKEN_ARG2)) {
                    reference += arg2word + " ";
                } else if (!word.equals(RoboCup.TOKEN_END) && !word.replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                    reference += word + " ";
                }
            }
            Sequence<IString> referenceSeq = IStrings.tokenize(NISTTokenizer.tokenize(reference.trim()));
            references.add(referenceSeq);
        }
        return references;
    }
}
