/* 
 * Copyright (C) 2016 Gerasimos Lampouras
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package imitationNLG;

import edu.stanford.nlp.mt.metrics.BLEUMetric;
import edu.stanford.nlp.mt.metrics.NISTMetric;
import edu.stanford.nlp.mt.tools.NISTTokenizer;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.util.IStrings;
import edu.stanford.nlp.mt.util.ScoredFeaturizedTranslation;
import edu.stanford.nlp.mt.util.Sequence;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import jarow.Instance;
import jarow.JAROW;
import jarow.Prediction;
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
import jdagger.JDAgger;

public class RoboCup {

    static Random r = new Random();

    static HashMap<String, ArrayList<String>> dictionary = new HashMap<>();
    static ArrayList<String> dictionaryBi = new ArrayList<>();
    static ArrayList<String> dictionaryTri = new ArrayList<>();

    static HashMap<String, ArrayList<String>> argDictionary = new HashMap<>();
    static HashMap<String, HashMap<String, Integer>> argDictionaryMap = new HashMap<>();
    static HashMap<String, ArrayList<String>> arguments = new HashMap<>();
    static ArrayList<String> predicates = new ArrayList<>();
    static HashMap<String, ArrayList<MeaningRepresentation>> meaningReprs = new HashMap<>();

    static HashMap<String, HashSet<ArrayList<String>>> patterns = new HashMap<>();
    static HashMap<MeaningRepresentation, ArrayList<String>> oneRefPatterns = new HashMap<>();

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

    static HashMap<String, HashSet<String>> generationsPerPredicate = new HashMap<>();

    static ArrayList<Double> NISTDocScores = new ArrayList<>();
    static ArrayList<Double> BLEUDocScores = new ArrayList<>();
    static ArrayList<Double> BLEUSmoothDocScores = new ArrayList<>();

    static ArrayList<Double> unbiasedNISTDocScores = new ArrayList<>();
    static ArrayList<Double> unbiasedBLEUDocScores = new ArrayList<>();
    static ArrayList<Double> unbiasedBLEUSmoothDocScores = new ArrayList<>();

    static ArrayList<Double> oneRefNISTDocScores = new ArrayList<>();
    static ArrayList<Double> oneRefBLEUDocScores = new ArrayList<>();
    static ArrayList<Double> oneRefBLEUSmoothDocScores = new ArrayList<>();

    public static int rounds = 10;

    public static void main(String[] args) {
        boolean useDAgger = false;

        /*System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to @arg2@", "@arg1@ kicks to @arg1@ passes to @arg2@", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to @arg2@", "@arg1@ kicks to passes to @arg2@", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to @arg2@", "@arg1@ kicks to to @arg2@", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to @arg2@", "@arg1@ kicks to @arg2@", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to @arg2@", "@arg1@ passes to", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to @arg2@", "@arg1@ passes passes", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to @arg2@", "@arg1@ kicks to", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to @arg2@", "@arg1@ kicks passes", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes", "@arg1@ passes passes", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes", "@arg1@ passes kicks", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to", "@arg1@ passes passes passes", 3, false));        
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to", "@arg1@ passes passes", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to", "@arg1@ passes kicks", 3, false));
         System.out.println(Cosine.getNGramSimilarity("@arg1@ passes to @arg2@", "", 3, false));
         System.exit(0);*/
        runTestWithJAROW(useDAgger);
    }

    public static void runTestWithJAROW(boolean useDAgger) {
        File dataFolder = new File("robocup_data\\gold\\");

        createLists(dataFolder, -1);
        initializeEvaluation();

        HashMap<String, JAROW> classifiersWords = new HashMap<>();
        if (dataFolder.isDirectory()) {
            for (int f = 0; f < dataFolder.listFiles().length; f++) {
                File file = dataFolder.listFiles()[f];
                createLists(dataFolder, f);
                createTrainingDatasets(new File("robocup_data\\gold\\"), "robocup_data\\goldTrainingData", f);
                for (String predicateStr : predicates) {
                    //if (predicateStr.startsWith("pass")) {
                    //int f = 0;
                    classifiersWords.put(predicateStr, genTest("robocup_data\\", file, f, predicateStr, useDAgger));
                    //System.exit(0);                    
                    //}
                }
                evaluateGenerationDoc(classifiersWords, file);
            }
        }
        printEvaluation("robocup_data\\results", -1);
    }

    public static JAROW genTest(String modelPath, File testFile, int excludeFile, String predicateStr, boolean useDAgger) {
        String line;
        ArrayList<Instance> wordInstances = new ArrayList<>();
        if ((new File("robocup_data\\goldTrainingData_words_" + predicateStr + "_excl" + excludeFile)).exists()) {
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
                    wordInstances.add(new Instance(featureVector, null, costs));
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            }

            if (useDAgger) {
                //DAGGER USE
                ArrayList<Action> availableActions = new ArrayList();
                for (String word : dictionary.get(predicateStr)) {
                    availableActions.add(new Action(word, ""));
                }
                HashMap<ActionSequence, Integer> referencePolicy = getReferencePolicy(new File("robocup_data\\gold\\"), "robocup_data\\goldTrainingData", excludeFile);

                ArrayList<MeaningRepresentation> meaningReprsSubset = new ArrayList<>();
                for (MeaningRepresentation m : meaningReprs.get(predicateStr)) {
                    if (m.getPredicate().equals(predicateStr)) {
                        meaningReprsSubset.add(m);
                    }
                }
                JDAgger dagger = new JDAgger();
                System.out.println("Run dagger for property " + predicateStr + " while excluding file " + excludeFile);
                JAROW classifierWords = dagger.runDAgger(predicateStr, wordInstances, meaningReprsSubset, availableActions, referencePolicy, oneRefPatterns, 5, 0.7);
                evaluateGeneration(classifierWords, testFile, predicateStr);

                return classifierWords;
            } else {
                //NO DAGGER USE
                JAROW classifierWords = new JAROW();
                //Collections.shuffle(wordInstances);
                //classifierWords.train(wordInstances, true, true, rounds, 0.1, true);
                Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
                //wordInstances = Instance.removeHapaxLegomena(wordInstances);
                classifierWords = JAROW.trainOpt(wordInstances, rounds, params, 0.2, false, false);
                evaluateGeneration(classifierWords, testFile, predicateStr);

                return classifierWords;
            }
        }
        return null;
    }

    public static void initializeEvaluation() {
        for (String predicateStr : predicates) {
            if (!generationsPerPredicate.containsKey(predicateStr)) {
                generationsPerPredicate.put(predicateStr, new HashSet<String>());
            }

            if (!NISTScoresPerPredicate.containsKey(predicateStr)) {
                NISTScoresPerPredicate.put(predicateStr, new ArrayList<Double>());
            }
            if (!BLEUScoresPerPredicate.containsKey(predicateStr)) {
                BLEUScoresPerPredicate.put(predicateStr, new ArrayList<Double>());
            }
            if (!BLEUSmoothScoresPerPredicate.containsKey(predicateStr)) {
                BLEUSmoothScoresPerPredicate.put(predicateStr, new ArrayList<Double>());
            }

            if (!unbiasedNISTScoresPerPredicate.containsKey(predicateStr)) {
                unbiasedNISTScoresPerPredicate.put(predicateStr, new ArrayList<Double>());
            }
            if (!unbiasedBLEUScoresPerPredicate.containsKey(predicateStr)) {
                unbiasedBLEUScoresPerPredicate.put(predicateStr, new ArrayList<Double>());
            }
            if (!unbiasedBLEUSmoothScoresPerPredicate.containsKey(predicateStr)) {
                unbiasedBLEUSmoothScoresPerPredicate.put(predicateStr, new ArrayList<Double>());
            }

            if (!oneRefNISTScoresPerPredicate.containsKey(predicateStr)) {
                oneRefNISTScoresPerPredicate.put(predicateStr, new ArrayList<Double>());
            }
            if (!oneRefBLEUScoresPerPredicate.containsKey(predicateStr)) {
                oneRefBLEUScoresPerPredicate.put(predicateStr, new ArrayList<Double>());
            }
            if (!oneRefBLEUSmoothScoresPerPredicate.containsKey(predicateStr)) {
                oneRefBLEUSmoothScoresPerPredicate.put(predicateStr, new ArrayList<Double>());
            }
        }
    }

    public static void printEvaluation(String writeFolderPath, int epoch) {
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

            System.out.println("^^^^^^^^^^^^^^^^");
            System.out.println("^^^^^^^^^^^^^^^^");
            System.out.println("\t" + generationsPerPredicate.get(predicateStr));
            System.out.println("^^^^^^^^^^^^^^^^");
            System.out.println(predicateStr + " BATCH NIST SCORE:\t" + avgNISTScores);
            System.out.println(predicateStr + " BATCH BLEU SCORE:\t" + avgBLEUScores);
            System.out.println(predicateStr + " BATCH BLEU SMOOTH SCORE:\t" + avgBLEUSmoothScores);
            System.out.println("^^^^^^^^^^^^^^^^");
            System.out.println(predicateStr + " UNBIASED BATCH NIST SCORE:\t" + avgUnbiasedNISTScores);
            System.out.println(predicateStr + " UNBIASED BATCH BLEU SCORE:\t" + avgUnbiasedBLEUScores);
            System.out.println(predicateStr + " UNBIASED BATCH BLEU SMOOTH SCORE:\t" + avgUnbiasedBLEUSmoothScores);
            System.out.println("^^^^^^^^^^^^^^^^");
            System.out.println(predicateStr + " ONEREF BATCH NIST SCORE:\t" + avgOneRefNISTScores);
            System.out.println(predicateStr + " ONEREF BATCH BLEU SCORE:\t" + avgOneRefBLEUScores);
            System.out.println(predicateStr + " ONEREF BATCH BLEU SMOOTH SCORE:\t" + avgOneRefBLEUSmoothScores);
            System.out.println("^^^^^^^^^^^^^^^^");
            System.out.println("^^^^^^^^^^^^^^^^");

            try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_evaluation_" + predicateStr + "_" + epoch), "utf-8"))) {
                writer.write("^^^^^^^^^^^^^^^^" + "\n");
                writer.write("^^^^^^^^^^^^^^^^" + "\n");
                writer.write("\t" + generationsPerPredicate.get(predicateStr).toString());
                writer.write("^^^^^^^^^^^^^^^^");
                writer.write(predicateStr + " BATCH NIST SCORE:\t" + avgNISTScores + "\n");
                writer.write(predicateStr + " BATCH BLEU SCORE:\t" + avgBLEUScores + "\n");
                writer.write(predicateStr + " BATCH BLEU SMOOTH SCORE:\t" + avgBLEUSmoothScores + "\n");
                writer.write("^^^^^^^^^^^^^^^^" + "\n");
                writer.write(predicateStr + " UNBIASED BATCH NIST SCORE:\t" + avgUnbiasedNISTScores + "\n");
                writer.write(predicateStr + " UNBIASED BATCH BLEU SCORE:\t" + avgUnbiasedBLEUScores + "\n");
                writer.write(predicateStr + " UNBIASED BATCH BLEU SMOOTH SCORE:\t" + avgUnbiasedBLEUSmoothScores + "\n");
                writer.write("^^^^^^^^^^^^^^^^" + "\n");
                writer.write(predicateStr + " ONEREF BATCH NIST SCORE:\t" + avgOneRefNISTScores + "\n");
                writer.write(predicateStr + " ONEREF BATCH BLEU SCORE:\t" + avgOneRefBLEUScores + "\n");
                writer.write(predicateStr + " ONEREF BATCH BLEU SMOOTH SCORE:\t" + avgOneRefBLEUSmoothScores + "\n");
                writer.write("^^^^^^^^^^^^^^^^" + "\n");
                writer.write("^^^^^^^^^^^^^^^^" + "\n");
                writer.close();
            } catch (UnsupportedEncodingException ex) {
                Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
            } catch (FileNotFoundException ex) {
                Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
            }
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

        double avgNISTDocScores = 0.0;
        double avgBLEUDocScores = 0.0;
        double avgBLEUSmoothDocScores = 0.0;

        double avgUnbiasedNISTDocScores = 0.0;
        double avgUnbiasedBLEUDocScores = 0.0;
        double avgUnbiasedBLEUSmoothDocScores = 0.0;

        double avgOneRefNISTDocScores = 0.0;
        double avgOneRefBLEUDocScores = 0.0;
        double avgOneRefBLEUSmoothDocScores = 0.0;

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

        for (int i = 0; i < NISTDocScores.size(); i++) {
            avgNISTDocScores += NISTDocScores.get(i);
            avgBLEUDocScores += BLEUDocScores.get(i);
            avgBLEUSmoothDocScores += BLEUSmoothDocScores.get(i);

            avgUnbiasedNISTDocScores += unbiasedNISTDocScores.get(i);
            avgUnbiasedBLEUDocScores += unbiasedBLEUDocScores.get(i);
            avgUnbiasedBLEUSmoothDocScores += unbiasedBLEUSmoothDocScores.get(i);

            avgOneRefNISTDocScores += oneRefNISTDocScores.get(i);
            avgOneRefBLEUDocScores += oneRefBLEUDocScores.get(i);
            avgOneRefBLEUSmoothDocScores += oneRefBLEUSmoothDocScores.get(i);
        }
        avgNISTDocScores /= (double) NISTDocScores.size();
        avgBLEUDocScores /= (double) BLEUDocScores.size();
        avgBLEUSmoothDocScores /= (double) BLEUSmoothDocScores.size();

        avgUnbiasedNISTDocScores /= (double) unbiasedNISTDocScores.size();
        avgUnbiasedBLEUDocScores /= (double) unbiasedBLEUDocScores.size();
        avgUnbiasedBLEUSmoothDocScores /= (double) unbiasedBLEUSmoothDocScores.size();

        avgOneRefNISTDocScores /= (double) oneRefNISTDocScores.size();
        avgOneRefBLEUDocScores /= (double) oneRefBLEUDocScores.size();
        avgOneRefBLEUSmoothDocScores /= (double) oneRefBLEUSmoothDocScores.size();

        System.out.println("^^^^^^^^^^^^^^^^");
        System.out.println("^^^^^^^^^^^^^^^^");
        System.out.println("BATCH NIST SCORE:\t" + avgNISTScores);
        System.out.println("BATCH BLEU SCORE:\t" + avgBLEUScores);
        System.out.println("BATCH BLEU SMOOTH SCORE:\t" + avgBLEUSmoothScores);
        System.out.println("^^^^^^^^^^^^^^^^");
        System.out.println("UNBIASED BATCH NIST SCORE:\t" + avgUnbiasedNISTScores);
        System.out.println("UNBIASED BATCH BLEU SCORE:\t" + avgUnbiasedBLEUScores);
        System.out.println("UNBIASED BATCH BLEU SMOOTH SCORE:\t" + avgUnbiasedBLEUSmoothScores);
        System.out.println("^^^^^^^^^^^^^^^^");
        System.out.println("ONEREF BATCH NIST SCORE:\t" + avgOneRefNISTScores);
        System.out.println("ONEREF BATCH BLEU SCORE:\t" + avgOneRefBLEUScores);
        System.out.println("ONEREF BATCH BLEU SMOOTH SCORE:\t" + avgOneRefBLEUSmoothScores);
        System.out.println("^^^^^^^^^^^^^^^^");
        System.out.println("^^^^^^^^^^^^^^^^");
        System.out.println("DOC BATCH NIST SCORE:\t" + avgNISTDocScores);
        System.out.println("DOC BATCH BLEU SCORE:\t" + avgBLEUDocScores);
        System.out.println("DOC BATCH BLEU SMOOTH SCORE:\t" + avgBLEUSmoothDocScores);
        System.out.println("^^^^^^^^^^^^^^^^");
        System.out.println("DOC UNBIASED BATCH NIST SCORE:\t" + avgUnbiasedNISTDocScores);
        System.out.println("DOC UNBIASED BATCH BLEU SCORE:\t" + avgUnbiasedBLEUDocScores);
        System.out.println("DOC UNBIASED BATCH BLEU SMOOTH SCORE:\t" + avgUnbiasedBLEUSmoothDocScores);
        System.out.println("^^^^^^^^^^^^^^^^");
        System.out.println("DOC ONEREF BATCH NIST SCORE:\t" + avgOneRefNISTDocScores);
        System.out.println("DOC ONEREF BATCH BLEU SCORE:\t" + avgOneRefBLEUDocScores);
        System.out.println("DOC ONEREF BATCH BLEU SMOOTH SCORE:\t" + avgOneRefBLEUSmoothDocScores);
        System.out.println("^^^^^^^^^^^^^^^^");
        System.out.println("^^^^^^^^^^^^^^^^");

        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_evaluation_batch_" + epoch), "utf-8"))) {
            writer.write("^^^^^^^^^^^^^^^^" + "\n");
            writer.write("^^^^^^^^^^^^^^^^" + "\n");
            writer.write("BATCH NIST SCORE:\t" + avgNISTScores + "\n");
            writer.write("BATCH BLEU SCORE:\t" + avgBLEUScores + "\n");
            writer.write("BATCH BLEU SMOOTH SCORE:\t" + avgBLEUSmoothScores + "\n");
            writer.write("^^^^^^^^^^^^^^^^" + "\n");
            writer.write("UNBIASED BATCH NIST SCORE:\t" + avgUnbiasedNISTScores + "\n");
            writer.write("UNBIASED BATCH BLEU SCORE:\t" + avgUnbiasedBLEUScores + "\n");
            writer.write("UNBIASED BATCH BLEU SMOOTH SCORE:\t" + avgUnbiasedBLEUSmoothScores + "\n");
            writer.write("^^^^^^^^^^^^^^^^" + "\n");
            writer.write("ONEREF BATCH NIST SCORE:\t" + avgOneRefNISTScores + "\n");
            writer.write("ONEREF BATCH BLEU SCORE:\t" + avgOneRefBLEUScores + "\n");
            writer.write("ONEREF BATCH BLEU SMOOTH SCORE:\t" + avgOneRefBLEUSmoothScores + "\n");
            writer.write("^^^^^^^^^^^^^^^^");
            writer.write("^^^^^^^^^^^^^^^^");
            writer.write("DOC BATCH NIST SCORE:\t" + avgNISTDocScores);
            writer.write("DOC BATCH BLEU SCORE:\t" + avgBLEUDocScores);
            writer.write("DOC BATCH BLEU SMOOTH SCORE:\t" + avgBLEUSmoothDocScores);
            writer.write("^^^^^^^^^^^^^^^^");
            writer.write("DOC UNBIASED BATCH NIST SCORE:\t" + avgUnbiasedNISTDocScores);
            writer.write("DOC UNBIASED BATCH BLEU SCORE:\t" + avgUnbiasedBLEUDocScores);
            writer.write("DOC UNBIASED BATCH BLEU SMOOTH SCORE:\t" + avgUnbiasedBLEUSmoothDocScores);
            writer.write("^^^^^^^^^^^^^^^^");
            writer.write("DOC ONEREF BATCH NIST SCORE:\t" + avgOneRefNISTDocScores);
            writer.write("DOC ONEREF BATCH BLEU SCORE:\t" + avgOneRefBLEUDocScores);
            writer.write("DOC ONEREF BATCH BLEU SMOOTH SCORE:\t" + avgOneRefBLEUSmoothDocScores);
            writer.write("^^^^^^^^^^^^^^^^" + "\n");
            writer.write("^^^^^^^^^^^^^^^^" + "\n");
            writer.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
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
                wordInstances.add(new Instance(featureVector, null, costs));
                //System.out.println(instances.get(instances.size() - 1).getCosts());

            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class
                    .getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class
                    .getName()).log(Level.SEVERE, null, ex);
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
                argInstances.add(new Instance(featureVector, null, costs));
                //System.out.println(instances.get(instances.size() - 1).getCosts());

            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class
                    .getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class
                    .getName()).log(Level.SEVERE, null, ex);
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
                instances.add(new Instance(featureVector, null, costs));
                //System.out.println(instances.get(instances.size()).getCosts());

            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class
                    .getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class
                    .getName()).log(Level.SEVERE, null, ex);
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
                JAROW classifier_p = JAROW.trainOpt(trainingInstances, rounds, params, 0.1, true, false);

                System.out.println("test data: " + testingInstances.size() + " instances");
                Double cost = classifier_p.batchPredict(testingInstances);
                Double avgCost = cost / (double) testingInstances.size();
                System.out.println("Avg Cost per instance " + avgCost + " on " + testingInstances.size() + " testing instances");
            }
        }
    }

    public static void createLists(File dataFolder, int excludeFileID) {
        dictionary = new HashMap<>();
        argDictionary = new HashMap<>();
        argDictionaryMap = new HashMap<>();
        arguments = new HashMap<>();
        predicates = new ArrayList<>();
        patterns = new HashMap<>();
        meaningReprs = new HashMap<>();
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

                                        if (!argDictionary.containsKey(predicate)) {
                                            argDictionary.put(predicate, new ArrayList<String>());
                                        }
                                        if (!arguments.containsKey(predicate)) {
                                            arguments.put(predicate, new ArrayList<String>());
                                        }
                                        if (!meaningReprs.containsKey(predicate)) {
                                            meaningReprs.put(predicate, new ArrayList<MeaningRepresentation>());
                                        }
                                        if (!dictionary.containsKey(predicate)) {
                                            dictionary.put(predicate, new ArrayList<String>());

                                            dictionary.get(predicate).add(RoboCup.TOKEN_END);
                                            dictionary.get(predicate).add(RoboCup.TOKEN_ARG1);
                                            dictionary.get(predicate).add(RoboCup.TOKEN_ARG2);
                                        }
                                    }
                                    HashMap<String, HashSet<String>> passedArgs = new HashMap<>();
                                    int a = 0;
                                    for (String arg : args) {
                                        a++;
                                        if (!arguments.get(predicate).contains(arg)) {
                                            arguments.get(predicate).add(arg);
                                        }
                                        if (!argDictionaryMap.containsKey(arg)) {
                                            argDictionaryMap.put(arg, new HashMap<String, Integer>());
                                        }
                                        HashSet<String> values = new HashSet<String>();
                                        values.add(arg);
                                        passedArgs.put("@arg" + a + "@", values);
                                    }
                                    meaningReprs.get(predicate).add(new MeaningRepresentation(predicate, passedArgs, ""));
                                }

                                NodeList nl = node.getElementsByTagName("nl");
                                if (nl != null && nl.getLength() > 0) {
                                    String[] nlWords = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", "").trim().split(" ");

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
                                    HashMap<String, HashSet<String>> passedArgs = new HashMap<>();
                                    int a = 0;
                                    for (String arg : args) {
                                        a++;
                                        Double max = -Double.MAX_VALUE;
                                        String[] bestAlignment = new String[2];
                                        for (String[] alignment : alignments.keySet()) {
                                            if (alignment[0].equals(arg)) {
                                                if (alignments.get(alignment) > max) {
                                                    max = alignments.get(alignment);
                                                    bestAlignment = alignment;
                                                }
                                            }
                                        }
                                        if (max >= 0.3) {
                                            bestAlignments.put(bestAlignment[1], bestAlignment[0]);
                                        }
                                        HashSet<String> values = new HashSet<String>();
                                        values.add(arg);
                                        passedArgs.put("@arg" + a + "@", values);
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
                                            if (nlWordsList.size() > 2) {
                                                if (nlWordsList.get(nlWordsList.size() - 2).equals("the")) {
                                                    phrase = " the " + nlWordsList.get(nlWordsList.size() - 1) + " " + phrase;
                                                    nlWordsList.remove(nlWordsList.size() - 1);
                                                    nlWordsList.remove(nlWordsList.size() - 1);
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
                                                    argDictionary.get(predicate).add(phrase.replaceAll("\\s+", " ").trim());
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
                                            argDictionary.get(predicate).add(phrase.replaceAll("\\s+", " ").trim());
                                        }
                                    }

                                    if (f != excludeFileID) {
                                        for (String word : nlWordsList) {
                                            if (!word.trim().isEmpty() && !dictionary.get(predicate).contains(word.trim()) && !word.replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                                dictionary.get(predicate).add(word.trim());
                                            }
                                        }

                                        oneRefPatterns.put(new MeaningRepresentation(predicate, passedArgs, ""), nlWordsList);
                                    }
                                    if (!patterns.containsKey(predicate)) {
                                        patterns.put(predicate, new HashSet<ArrayList<String>>());
                                    }
                                    patterns.get(predicate).add(nlWordsList);
                                }
                            }
                        }
                    }
                }
            }
        }

        dictionaryBi = new ArrayList<>();
        /*dictionaryBi.add("@@|@@");
         for (String word1 : dictionary) {
         for (String word2 : dictionary) {
         if (!word1.equals(RoboCup.TOKEN_END)) {
         if (!dictionaryBi.contains(word1 + "|" + word2)) {
         dictionaryBi.add(word1 + "|" + word2);
         }
         }
         }
         if (!dictionaryBi.contains("@@|" + word1)) {
         dictionaryBi.add("@@|" + word1);
         }
         }*/

        dictionaryTri = new ArrayList<>();
        /*dictionaryTri.add("@@|@@|@@");
         for (String word1 : dictionary) {
         for (String word2 : dictionary) {
         System.out.println(word2);
         for (String word3 : dictionary) {
         if (!word1.equals(RoboCup.TOKEN_END) && !word2.equals(RoboCup.TOKEN_END)) {
         if (!dictionaryTri.contains(word1 + "|" + word2 + "|" + word3)) {
         dictionaryTri.add(word1 + "|" + word2 + "|" + word3);
         }
         }
         }
         if (!dictionaryTri.contains("@@|" + word1 + "|" + word2)) {
         dictionaryTri.add("@@|" + word1 + "|" + word2);
         }
         }
         if (!dictionaryTri.contains("@@|" + "@@|" + word1)) {
         dictionaryTri.add("@@|" + "@@|" + word1);
         }
         }*/
        Collections.sort(predicates);
    }

    public static void saveLists(String predicate, String writeFolderPath) {
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_dictionary"), "utf-8"))) {
            for (int i = 0; i < dictionary.get(predicate).size(); i++) {
                writer.write(i + ":" + dictionary.get(predicate).get(i) + "\n");
            }
            writer.close();

        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class
                    .getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class
                    .getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class
                    .getName()).log(Level.SEVERE, null, ex);
        }
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_arguments"), "utf-8"))) {
            for (int i = 0; i < arguments.get(predicate).size(); i++) {
                writer.write(i + ":" + arguments.get(i) + "\n");
            }
            writer.close();

        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class
                    .getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class
                    .getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class
                    .getName()).log(Level.SEVERE, null, ex);
        }
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_predicates"), "utf-8"))) {
            for (int i = 0; i < predicates.size(); i++) {
                writer.write(i + ":" + predicates.get(i) + "\n");
            }
            writer.close();

        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class
                    .getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class
                    .getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class
                    .getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void createTrainingDatasets(File dataFolder, String trainingFilePath, int excludeFileID) {
        if (!dictionary.isEmpty() && !predicates.isEmpty() && !arguments.isEmpty()) {
            HashMap<String, ArrayList<String>> predicateWordTrainingData = new HashMap<>();
            HashMap<String, ArrayList<String>> predicateArgTrainingData = new HashMap<>();
            for (String predicate : predicates) {
                predicateWordTrainingData.put(predicate, new ArrayList<String>());
                predicateArgTrainingData.put(predicate, new ArrayList<String>());
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
                                            if (!arguments.get(predicate).contains(arg)) {
                                                arguments.get(predicate).add(arg);
                                            }
                                            if (!argDictionaryMap.containsKey(arg)) {
                                                argDictionaryMap.put(arg, new HashMap<String, Integer>());
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
                                            Double max = -Double.MAX_VALUE;
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
                                            if (max >= 0.3) {
                                                bestAlignments.put(bestAlignment[1], bestAlignment[0]);
                                            }
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
                                                if (nlWordsList.size() > 2) {
                                                    if (nlWordsList.get(nlWordsList.size() - 2).equals("the")) {
                                                        phrase = " the " + nlWordsList.get(nlWordsList.size() - 1) + " " + phrase;
                                                        nlWordsList.remove(nlWordsList.size() - 1);
                                                        nlWordsList.remove(nlWordsList.size() - 1);
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
                                                    argDictionary.get(predicate).add(phrase);
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
                                            argDictionary.get(predicate).add(phrase);
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
                                            String wordTrainingVector = createStringWordInstance(predicate, nlWordsList, w, arg1toBeMentioned, arg2toBeMentioned);
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
                                            String argTrainingVector1 = createArgTrainingVector(predicate, arguments.get(predicate).indexOf(arguement), bestPhraseAlignments.get(arguement));
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
                    Logger.getLogger(RoboCup.class
                            .getName()).log(Level.SEVERE, null, ex);
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(RoboCup.class
                            .getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(RoboCup.class
                            .getName()).log(Level.SEVERE, null, ex);
                }
                try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(argsPath), "utf-8"))) {
                    for (String trainingVector : predicateArgTrainingData.get(predicate)) {
                        writer.write(trainingVector + "\n");
                    }
                    writer.close();

                } catch (UnsupportedEncodingException ex) {
                    Logger.getLogger(RoboCup.class
                            .getName()).log(Level.SEVERE, null, ex);
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(RoboCup.class
                            .getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(RoboCup.class
                            .getName()).log(Level.SEVERE, null, ex);
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
                predicateWordTrainingData.put(predicate, new ArrayList<String>());
                predicateArgTrainingData.put(predicate, new ArrayList<String>());
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
                                            if (!arguments.get(predicate).contains(arg)) {
                                                arguments.get(predicate).add(arg);
                                            }
                                            if (!argDictionaryMap.containsKey(arg)) {
                                                argDictionaryMap.put(arg, new HashMap<String, Integer>());
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
                                            Double max = -Double.MAX_VALUE;
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
                                            if (max >= 0.3) {
                                                bestAlignments.put(bestAlignment[1], bestAlignment[0]);
                                            }
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
                                                if (nlWordsActionList.size() > 2) {
                                                    if (nlWordsActionList.get(nlWordsActionList.size() - 2).equals("the")) {
                                                        phrase = " the " + nlWordsActionList.get(nlWordsActionList.size() - 1) + " " + phrase;
                                                        nlWordsActionList.remove(nlWordsActionList.size() - 1);
                                                        nlWordsActionList.remove(nlWordsActionList.size() - 1);
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
                                                        nlWordsActionList.add(new Action(RoboCup.TOKEN_ARG1, ""));
                                                    } else if (args.indexOf(arg) == 1) {
                                                        nlWordsActionList.add(new Action(RoboCup.TOKEN_ARG2, ""));
                                                    }
                                                    phrase = phrase.replaceAll("\\s+", " ").trim();
                                                    argDictionary.get(predicate).add(phrase);
                                                    bestPhraseAlignments.put(arg, phrase);
                                                    phrase = "";
                                                    arg = "";
                                                }
                                                if (!nlWords[w].replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                                    nlWordsActionList.add(new Action(nlWords[w].trim(), ""));
                                                }
                                            }
                                        }
                                        if (!phrase.isEmpty() && !arg.isEmpty()) {
                                            if (args.indexOf(arg) == 0) {
                                                nlWordsActionList.add(new Action(RoboCup.TOKEN_ARG1, ""));
                                            } else if (args.indexOf(arg) == 1) {
                                                nlWordsActionList.add(new Action(RoboCup.TOKEN_ARG2, ""));
                                            }
                                            phrase = phrase.replaceAll("\\s+", " ").trim();
                                            argDictionary.get(predicate).add(phrase);
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

    public static String createStringWordInstance(String predicate, ArrayList<String> nlWords, int w, boolean arg1toBeMentioned, boolean arg2toBeMentioned) {
        String trainingVector = "";

        String bestAction = nlWords.get(w).toLowerCase().trim();
        if (!bestAction.isEmpty()) {
            //COSTS
            for (String action : dictionary.get(predicate)) {
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
                for (int i = 0; i < dictionary.get(predicate).size(); i++) {
                    double featureValue = 0.0;
                    if (!previousWord.isEmpty() && dictionary.get(predicate).get(i).equals(previousWord)) {
                        featureValue = 1.0;
                    }
                    trainingVector += " " + "feature_" + j + "_" + dictionary.get(predicate).get(i) + ":" + featureValue;
                }
                if (previousWord.isEmpty()) {
                    trainingVector += " " + "feature_" + j + "_@@" + ":1.0";
                } else {
                    trainingVector += " " + "feature_" + j + "_@@" + ":0.0";
                }
            }
            //Word Positions
            //trainingVector += " " + "feature_" + (featureNo++) + ":" + w/20;
            //If arguments have already been generated or not
            if (arg1toBeMentioned) {
                trainingVector += " " + "feature_arg1m" + ":1.0";
            } else {
                trainingVector += " " + "feature_arg1m" + ":0.0";
            }
            if (arg2toBeMentioned) {
                trainingVector += " " + "feature_arg2m" + ":1.0";
            } else {
                trainingVector += " " + "feature_arg2m" + ":0.0";
            }
        }
        return trainingVector.trim();
    }

    public static Instance createWordInstance(String predicate, ArrayList<String> nlWords, int w, HashMap<String, Boolean> argumentsToBeMentioned) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        String bestAction = nlWords.get(w).toLowerCase().trim();
        if (!bestAction.isEmpty()) {
            //COSTS
            if (!dictionary.containsKey(predicate)) {
                dictionary.put(predicate, new ArrayList<String>());

                dictionary.get(predicate).add(RoboCup.TOKEN_END);
                dictionary.get(predicate).add(RoboCup.TOKEN_ARG1);
                dictionary.get(predicate).add(RoboCup.TOKEN_ARG2);
            }
            for (String action : dictionary.get(predicate)) {
                if (action.equals(bestAction)) {
                    costs.put(action, 0.0);
                } else {
                    costs.put(action, 1.0);
                }
            }
        }
        return createWordInstance(predicate, nlWords, w, costs, argumentsToBeMentioned);
    }

    public static Instance createWordInstance(String predicate, ArrayList<String> nlWords, int w, double cost, HashMap<String, Boolean> argumentsToBeMentioned) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        String bestAction = nlWords.get(w).toLowerCase().trim();
        if (!bestAction.isEmpty()) {
            //COSTS
            for (String action : dictionary.get(predicate)) {
                if (action.equals(bestAction)) {
                    costs.put(action, 1.0 - cost);
                } else {
                    costs.put(action, 1.0);
                }
            }
        }
        return createWordInstance(predicate, nlWords, w, costs, argumentsToBeMentioned);
    }

    public static Instance createWordInstance(String predicate, ArrayList<String> nlWords, int w, TObjectDoubleHashMap<String> costs, HashMap<String, Boolean> argumentsToBeMentioned) {
        TObjectDoubleHashMap<String> features = new TObjectDoubleHashMap<>();

        //Previous word features
        //System.out.println("£££ "  + nlWords);
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
            for (int i = 0; i < dictionary.get(predicate).size(); i++) {
                double featureValue = 0.0;
                if (!previousWord.isEmpty() && dictionary.get(predicate).get(i).equals(previousWord)) {
                    featureValue = 1.0;
                }
                features.put("feature_" + j + "_" + dictionary.get(predicate).get(i), featureValue);
            }
            if (previousWord.isEmpty()) {
                features.put("feature_" + j + "_@@", 1.0);
            } else {
                features.put("feature_" + j + "_@@", 0.0);
            }
        }
        //Word N-Grams            
        String prevWord = "@@";
        if (w - 1 >= 0) {
            prevWord = nlWords.get(w - 1).trim();
        }
        String prevPrevWord = "@@";
        if (w - 2 >= 0) {
            prevPrevWord = nlWords.get(w - 2).trim();
        }
        String prevPrevPrevWord = "@@";
        if (w - 3 >= 0) {
            prevPrevPrevWord = nlWords.get(w - 3).trim();
        }

        String prevBigram = prevPrevWord + "|" + prevWord;
        String prevTrigram = prevPrevPrevWord + "|" + prevPrevWord + "|" + prevWord;
        for (String bigram : dictionaryBi) {
            double featureValue = 0.0;

            if (prevBigram.equals(bigram)) {
                featureValue = 1.0;
            }
            features.put("feature_bigram_" + bigram, featureValue);
        }
        for (String trigram : dictionaryTri) {
            double featureValue = 0.0;

            if (prevTrigram.equals(trigram)) {
                featureValue = 1.0;
            }
            features.put("feature_trigram_" + trigram, featureValue);
        }

        //Word Positions
        //trainingVector += " " + "feature_" + (featureNo++) + ":" + w/20;
        //If arguments have already been generated or not
        boolean arg1toBeMentioned = true;
        if (argumentsToBeMentioned.containsKey(RoboCup.TOKEN_ARG1)) {
            arg1toBeMentioned = argumentsToBeMentioned.get(RoboCup.TOKEN_ARG1);
        }
        boolean arg2toBeMentioned = true;
        if (argumentsToBeMentioned.containsKey(RoboCup.TOKEN_ARG2)) {
            arg2toBeMentioned = argumentsToBeMentioned.get(RoboCup.TOKEN_ARG2);
        }
        if (arg1toBeMentioned) {
            features.put("feature_arg1m", 1.0);
        } else {
            features.put("feature_arg1m", 0.0);
        }
        if (arg2toBeMentioned) {
            features.put("feature_arg2m", 1.0);
        } else {
            features.put("feature_arg2m", 0.0);
        }
        //if (w == 12) {
        /*if (JDAgger.ep > 1 && JDAgger.train) {
         System.out.println("WORD INDEX " + w);
         for (String f : features.keySet()) {
         if (features.get(f) == 1.0) {
         System.out.print(f + "=" + features.get(f) + " , ");
         }
         }
         System.out.println();
         for (String c : costs.keySet()) {
         System.out.print(c + "=" + costs.get(c) + " , ");
         }
         System.out.println();
         }*/
        //System.exit(0);
        return new Instance(features, null, costs);
    }

    public static String createArgTrainingVector(String predicate, int argID, String bestAction) {
        String trainingVector = "";

        int featureNo = 1;
        if (!bestAction.isEmpty()) {
            //COSTS
            for (String action : dictionary.get(predicate)) {
                if (action.equals(bestAction)) {
                    trainingVector += " " + "cost_" + action + ":0.0";
                } else {
                    trainingVector += " " + "cost_" + action + ":1.0";
                }
            }
            //Arg ID
            for (int i = 0; i < arguments.get(predicate).size(); i++) {
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
            classifierGraded.train(gradedTrainingInstances, true, false, rounds, param, true);

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
                    classifierGraded.train(gradedTrainingInstances, true, false, rounds, param, true);

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

        System.out.println("Evaluate " + predicateStr);
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
                                    references.put(mrNode, new ArrayList<Sequence<IString>>());
                                    references.get(mrNode).add(reference);

                                    strReferences.put(mrNode, new ArrayList<String>());
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
                            mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ").replaceAll("\\s+", " ");
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
                            HashMap<String, Boolean> argumentsToBeMentioned = new HashMap<>();
                            for (String argument : args) {
                                argumentsToBeMentioned.put(argument, true);
                            }
                            while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10000) {
                                ArrayList<String> tempList = new ArrayList(predictedWordsList);
                                tempList.add("@TOK@");
                                Instance trainingVector = RoboCup.createWordInstance(predicate, tempList, w, argumentsToBeMentioned);

                                if (trainingVector != null) {
                                    Prediction predict = classifierWords.predict(trainingVector);
                                    predictedWord = predict.getLabel().trim();
                                    predictedWordsList.add(predictedWord);

                                    for (String arg : argumentsToBeMentioned.keySet()) {
                                        if (predictedWord.equals(arg)) {
                                            argumentsToBeMentioned.put(arg, false);
                                        }
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

                            unbiasedFinalReferences.add(createUnbiasedReferenceListSeq(arg1name, arg2name, predicate));

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

                            //totalUnbiasedBLEU += unbiasedBLEUScore;
                            //SimpleSequence<IString> seq = new SimpleSequence<>(seqList);                            
                            //generations.add(seq);
                            Double unbiasedBLEUScore = BLEUMetric.computeLocalSmoothScore(predictedStringWithArgs, createUnbiasedReferenceList(arg1name, arg2name, predicate), 4);
                            //totalUnbiasedBLEU += unbiasedBLEUScore;

                            //if (unbiasedBLEUScore < 1.0) {
                            generationsPerPredicate.get(predicateStr).add(predictedString);
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

    public static void evaluateGenerationDoc(HashMap<String, JAROW> classifiersWords, File testFile) {
        NISTTokenizer.lowercase(true);
        NISTTokenizer.normalize(true);

        System.out.println("Evaluate " + testFile);
        ArrayList<String> mrNodes = new ArrayList<>();
        HashMap<String, ArrayList<Sequence<IString>>> references = new HashMap<>();
        HashMap<String, ArrayList<String>> strReferences = new HashMap<>();

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

                            if (predicate != null) {
                                mrNodes.add(mrNode);
                                if (references.containsKey(mrNode)) {
                                    references.get(mrNode).add(reference);
                                    strReferences.get(mrNode).add((((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim()).replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", ""));
                                } else {
                                    references.put(mrNode, new ArrayList<Sequence<IString>>());
                                    references.get(mrNode).add(reference);

                                    strReferences.put(mrNode, new ArrayList<String>());
                                    strReferences.get(mrNode).add((((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim()).replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", ""));
                                }
                            }
                        }
                    }
                }
                //double totalUnbiasedBLEU = 0.0;
                //double totalOneRefBLEU = 0.0;
                String genDoc = "";
                String refDoc = "";
                String unbiasedRefDoc = "";
                String oneRefDoc = "";
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
                            mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ").replaceAll("\\s+", " ");
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

                        if (predicate != null && classifiersWords.containsKey(predicate)) {
                            //PHRASE GENERATION EVALUATION
                            String predictedWord = "";
                            int w = 0;
                            ArrayList<String> predictedWordsList = new ArrayList<>();
                            HashMap<String, Boolean> argumentsToBeMentioned = new HashMap<>();
                            for (String argument : args) {
                                argumentsToBeMentioned.put(argument, true);
                            }
                            while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10000) {
                                ArrayList<String> tempList = new ArrayList(predictedWordsList);
                                tempList.add("@TOK@");
                                Instance trainingVector = RoboCup.createWordInstance(predicate, tempList, w, argumentsToBeMentioned);

                                if (trainingVector != null) {
                                    Prediction predict = classifiersWords.get(predicate).predict(trainingVector);
                                    predictedWord = predict.getLabel().trim();
                                    predictedWordsList.add(predictedWord);

                                    for (String arg : argumentsToBeMentioned.keySet()) {
                                        if (predictedWord.equals(arg)) {
                                            argumentsToBeMentioned.put(arg, false);
                                        }
                                    }
                                }
                                w++;
                            }

                            String predictedString = "";
                            for (String word : predictedWordsList) {
                                predictedString += word + " ";
                            }
                            predictedString = predictedString.trim();

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

                            String[] nlWords = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim().replaceAll("\\'", " \\'").replaceAll("[\\p{Punct}&&[^\\'@]]", "").toLowerCase().split(" ");
                            String cleanedWords = "";
                            for (String nlWord : nlWords) {
                                if (!nlWord.replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().isEmpty()) {
                                    cleanedWords += nlWord + " ";
                                }
                            }
                            oneRefDoc += cleanedWords.trim() + " ";

                            //total++;
                            Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedStringWithArgs));
                            ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
                            genDoc += predictedStringWithArgs + " ";

                            double minScore = 0.0;
                            String minRef = null;
                            int n = 4;
                            while (minRef == null && n >= 0) {
                                for (Sequence<IString> ref : references.get(mrNode)) {
                                    ArrayList<String> refList = new ArrayList<>();
                                    String refStr = "";
                                    for (IString word : ref) {
                                        refStr += word.toString() + " ";
                                    }
                                    refStr = refStr.trim();
                                    refList.add(refStr);

                                    Double bScore = BLEUMetric.computeLocalSmoothScore(predictedStringWithArgs, refList, n);
                                    if (bScore > minScore) {
                                        minScore = bScore;
                                        minRef = refStr;
                                    }
                                }
                                n--;
                            }
                            refDoc += minRef + " ";

                            minScore = 0.0;
                            minRef = null;
                            n = 4;
                            while (minRef == null && n >= 0) {
                                for (Sequence<IString> ref : createUnbiasedReferenceListSeq(arg1name, arg2name, predicate)) {
                                    ArrayList<String> refList = new ArrayList<>();
                                    String refStr = "";
                                    for (IString word : ref) {
                                        refStr += word.toString() + " ";
                                    }
                                    refStr = refStr.trim();
                                    refList.add(refStr);

                                    Double bScore = BLEUMetric.computeLocalSmoothScore(predictedStringWithArgs, refList, n);
                                    if (bScore > minScore) {
                                        minScore = bScore;
                                        minRef = refStr;
                                    }
                                }
                                n--;
                            }
                            unbiasedRefDoc += minRef + " ";
                        }
                    }
                }
                ArrayList<ScoredFeaturizedTranslation> genDocList = new ArrayList<>();
                ArrayList<Sequence<IString>> refDocList = new ArrayList<>();
                ArrayList<Sequence<IString>> unbiasedRefDocList = new ArrayList<>();
                ArrayList<Sequence<IString>> oneRefDocList = new ArrayList<>();
                genDocList.add(new ScoredFeaturizedTranslation<>(IStrings.tokenize(NISTTokenizer.tokenize(genDoc.trim())), null, 0));
                refDocList.add(IStrings.tokenize(NISTTokenizer.tokenize(refDoc.trim())));
                unbiasedRefDocList.add(IStrings.tokenize(NISTTokenizer.tokenize(unbiasedRefDoc.trim())));
                oneRefDocList.add(IStrings.tokenize(NISTTokenizer.tokenize(oneRefDoc.trim())));

                ArrayList<ArrayList<Sequence<IString>>> finalRefDoc = new ArrayList<>();
                ArrayList<ArrayList<Sequence<IString>>> finalUnbiasedRefDoc = new ArrayList<>();
                ArrayList<ArrayList<Sequence<IString>>> finalOneRefDoc = new ArrayList<>();
                finalRefDoc.add(refDocList);
                finalUnbiasedRefDoc.add(unbiasedRefDocList);
                finalOneRefDoc.add(oneRefDocList);

                NISTMetric NISTDoc = new NISTMetric(finalRefDoc);
                BLEUMetric BLEUDoc = new BLEUMetric(finalRefDoc, 4, false);
                BLEUMetric BLEUsmoothDoc = new BLEUMetric(finalRefDoc, 4, true);
                Double nistDocScore = NISTDoc.score(genDocList);
                Double bleuDocScore = BLEUDoc.score(genDocList);
                Double bleuSmoothDocScore = BLEUsmoothDoc.score(genDocList);
                System.out.println("DOC BATCH NIST SCORE:\t" + nistDocScore);
                System.out.println("DOC BATCH BLEU SCORE:\t" + bleuDocScore);
                System.out.println("DOC BATCH BLEU SMOOTH SCORE:\t" + bleuSmoothDocScore);
                if (!nistDocScore.isNaN()) {
                    NISTDocScores.add(nistDocScore);
                }
                if (!bleuDocScore.isNaN()) {
                    BLEUDocScores.add(bleuDocScore);
                }
                if (!bleuSmoothDocScore.isNaN()) {
                    BLEUSmoothDocScores.add(bleuSmoothDocScore);
                }

                NISTMetric unbiasedNISTDoc = new NISTMetric(finalUnbiasedRefDoc);
                BLEUMetric unbiasedBLEUDoc = new BLEUMetric(finalUnbiasedRefDoc, 4, false);
                BLEUMetric unbiasedBLEUsmoothDoc = new BLEUMetric(finalUnbiasedRefDoc, 4, true);
                Double unbiasedNistDocScore = unbiasedNISTDoc.score(genDocList);
                Double unbiasedBleuDocScore = unbiasedBLEUDoc.score(genDocList);
                Double unbiasedBleuSmoothDocScore = unbiasedBLEUsmoothDoc.score(genDocList);

                System.out.println("DOC UNBIASED BATCH NIST SCORE:\t" + unbiasedNistDocScore);
                System.out.println("DOC UNBIASED BATCH BLEU SCORE:\t" + unbiasedBleuDocScore);
                System.out.println("DOC UNBIASED BATCH BLEU SMOOTH SCORE:\t" + unbiasedBleuSmoothDocScore);
                if (!unbiasedNistDocScore.isNaN()) {
                    unbiasedNISTDocScores.add(unbiasedNistDocScore);
                }
                if (!unbiasedBleuDocScore.isNaN()) {
                    unbiasedBLEUDocScores.add(unbiasedBleuDocScore);
                }
                if (!unbiasedBleuSmoothDocScore.isNaN()) {
                    unbiasedBLEUSmoothDocScores.add(unbiasedBleuSmoothDocScore);
                }

                NISTMetric oneRefNISTDoc = new NISTMetric(finalOneRefDoc);
                BLEUMetric oneRefBLEUDoc = new BLEUMetric(finalOneRefDoc, 4, false);
                BLEUMetric oneRefBLEUsmoothDoc = new BLEUMetric(finalOneRefDoc, 4, true);
                Double oneRefNistDocScore = oneRefNISTDoc.score(genDocList);
                Double oneRefBleuDocScore = oneRefBLEUDoc.score(genDocList);
                Double oneRefBleuSmoothDocScore = oneRefBLEUsmoothDoc.score(genDocList);

                System.out.println("DOC ONE REF BATCH NIST SCORE:\t" + oneRefNistDocScore);
                System.out.println("DOC ONE REF BATCH BLEU SCORE:\t" + oneRefBleuDocScore);
                System.out.println("DOC ONE REF BATCH BLEU SMOOTH SCORE:\t" + oneRefBleuSmoothDocScore);
                if (!oneRefNistDocScore.isNaN()) {
                    oneRefNISTDocScores.add(oneRefNistDocScore);
                }
                if (!oneRefBleuDocScore.isNaN()) {
                    oneRefBLEUDocScores.add(oneRefBleuDocScore);
                }
                if (!oneRefBleuSmoothDocScore.isNaN()) {
                    oneRefBLEUSmoothDocScores.add(oneRefBleuSmoothDocScore);
                }
            }
        }
    }

    public static ArrayList<String> createUnbiasedReferenceList(String predicate) {
        ArrayList<String> references = new ArrayList<>();
        for (ArrayList<String> pattern : patterns.get(predicate)) {
            String reference = "";
            for (String word : pattern) {
                reference += word + " ";
            }
            references.add(reference.trim());
        }
        return references;
    }

    public static ArrayList<String> createUnbiasedReferenceList(String arg1word, String arg2word, String predicate) {
        ArrayList<String> references = new ArrayList<>();
        for (ArrayList<String> pattern : patterns.get(predicate)) {
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

    public static ArrayList<Sequence<IString>> createUnbiasedReferenceListSeq(String arg1word, String arg2word, String predicate) {
        ArrayList<Sequence<IString>> references = new ArrayList<>();
        for (ArrayList<String> pattern : patterns.get(predicate)) {
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
