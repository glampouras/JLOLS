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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import similarity_measures.Levenshtein;
import jdagger.JDAggerForBagel;

/**
 *
 * @author localadmin
 */
public class Bagel {

    HashMap<String, HashSet<String>> attributes = new HashMap<>();
    HashMap<String, HashSet<String>> attributeValuePairs = new HashMap<>();
    HashMap<String, HashMap<MeaningRepresentation, HashSet<String>>> meaningReprs = new HashMap<>();
    HashMap<String, ArrayList<DatasetInstance>> abstractDatasetInstances = new HashMap<>();
    HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments = new HashMap<>();
    HashMap<ArrayList<Action>, Action> punctPatterns = new HashMap<>();
    ArrayList<String> predicates = new ArrayList<>();
    final public static String TOKEN_START = "@start@";
    final public static String TOKEN_END = "@end@";
    final public static String TOKEN_PUNCT = "@punct@";
    final public static String TOKEN_X = "@x@";
    final public static String TOKEN_ATTR = "@attr@";
    public int rounds = 10;
    public int maxAttrRealizationSize = 0;
    public int maxWordRealizationSize = 0;
    public ArrayList<Double> crossAvgArgDistances = new ArrayList<>();
    public ArrayList<Double> crossNIST = new ArrayList<>();
    public ArrayList<Double> crossBLEU = new ArrayList<>();
    public ArrayList<Double> crossBLEUSmooth = new ArrayList<>();
    static final int seed = 13;
    public static Random r = new Random(seed);
    boolean useAlignments = false;
    double wordRefRolloutChance = 0.8;
    static int fold = 0;

    public static void main(String[] args) {
        boolean useDAggerArg = false;
        boolean useLolsWord = true;

        fold = Integer.parseInt(args[0]);
        JDAggerForBagel.earlyStopMaxFurtherSteps = Integer.parseInt(args[1]);
        JDAggerForBagel.p = Double.parseDouble(args[2]);
        
        Bagel bagel = new Bagel();
        bagel.runTestWithJAROW(useDAggerArg, useLolsWord);
    }

    public void runTestWithJAROW(boolean useDAggerArg, boolean useDAggerWord) {
        File dataFile = new File("bagel_data/ACL10-inform-training.txt");

        createLists(dataFile);
        //initializeEvaluation();

        //RANDOM DATA SPLIT
        //for (double fold = 0.0; fold < 1.0; fold += 0.1) {
            /*for (double fold = 0.0; fold < 0.1; fold += 0.1) {
        int from = ((int) Math.round(datasetInstances.size() * fold)); //+ 1;
        
        if (from < datasetInstances.size()) {
        int to = (int) Math.round(datasetInstances.size() * (fold + 0.1));
        if (to > datasetInstances.size()) {
        to = datasetInstances.size();
        }
        ArrayList<DatasetInstance> testingData = new ArrayList<>(datasetInstances.subList(from, to));
        ArrayList<DatasetInstance> trainingData = new ArrayList<>(datasetInstances);
        trainingData.removeAll(testingData);*/
        //Ondrej et al. data split
        //for (int fold = 0; fold < 10; fold += 1) {
        for (int f = fold; f < fold + 1; f += 1) {
            System.out.println("======================================================");
            System.out.println("======================================================");
            System.out.println("                      ===F=" + f + "===                      ");
            System.out.println("======================================================");
            System.out.println("======================================================");

            for (String predicate : predicates) {
                r = new Random(seed);

                ArrayList<DatasetInstance> datasetInstances = new ArrayList(abstractDatasetInstances.get(predicate));

                HashMap<String, JAROW> classifiersAttrs = new HashMap<>();
                HashMap<String, HashMap<String, JAROW>> classifiersWords = new HashMap<>();

                ArrayList<DatasetInstance> testingData = new ArrayList<>();
                ArrayList<DatasetInstance> trainingData = new ArrayList<>();
                File trainMRFile = new File("bagel_data/Data splits of Ondrej/cv0" + f + "/train-das.txt");
                File testMRFile = new File("bagel_data/Data splits of Ondrej/cv0" + f + "/test-das.txt");

                //ArrayList<HashMap<String, HashSet<String>>> trainMRs = new ArrayList<>();
                ArrayList<String> trainMRs = new ArrayList<>();
                try (BufferedReader br = new BufferedReader(new FileReader(trainMRFile))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        if (!line.trim().isEmpty()) {
                            String MRstr = line;
                            int x = 1;
                            int xInd = MRstr.indexOf("X-");
                            while (xInd != -1) {
                                String xStr = MRstr.substring(xInd, MRstr.indexOf(")", xInd));
                                MRstr = MRstr.replaceFirst(xStr, "\"X" + x + "\"");
                                x++;

                                xInd = MRstr.indexOf("X-");
                            }
                            MRstr = MRstr.replaceAll("inform\\(", "").replaceAll("&", ",").replaceAll("\\)", "");

                            /*HashMap<String, Integer> attrXIndeces = new HashMap<>();
                            HashMap<String, HashSet<String>> attrs = new HashMap<>();
                            String[] attrsToBe = line.trim().split("&");
                            for (String s : attrsToBe) {
                            String[] comps = s.trim().split("=");
                            String attr = comps[0].trim().replace("inform(", "");
                            if (!attrs.containsKey(attr)) {
                            attrs.put(attr, new HashSet<String>());
                            }
                            
                            String value = comps[1].trim().replace(")", "");
                            if (value.startsWith("X-")) {
                            int index = 0;
                            if (!attrXIndeces.containsKey(attr)) {
                            attrXIndeces.put(attr, 1);
                            } else {
                            index = attrXIndeces.get(attr);
                            attrXIndeces.put(attr, index + 1);
                            }
                            value = "X" + index;
                            } else if (value.startsWith("\"")) {
                            value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                            }
                            attrs.get(attr.toLowerCase()).add(value.toLowerCase());
                            }
                            trainMRs.add(attrs);*/
                            trainMRs.add(MRstr);
                        }
                    }
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
                }
                //ArrayList<HashMap<String, HashSet<String>>> testMRs = new ArrayList<>();
                ArrayList<String> testMRs = new ArrayList<>();
                try (BufferedReader br = new BufferedReader(new FileReader(testMRFile))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        if (!line.trim().isEmpty()) {
                            String MRstr = line;
                            int x = 1;
                            int xInd = MRstr.indexOf("X-");
                            while (xInd != -1) {
                                String xStr = MRstr.substring(xInd, MRstr.indexOf(")", xInd));
                                MRstr = MRstr.replaceFirst(xStr, "\"X" + x + "\"");
                                x++;

                                xInd = MRstr.indexOf("X-");
                            }
                            MRstr = MRstr.replaceAll("inform\\(", "").replaceAll("&", ",").replaceAll("\\)", "");

                            /*HashMap<String, Integer> attrXIndeces = new HashMap<>();
                            HashMap<String, HashSet<String>> attrs = new HashMap<>();
                            String[] attrsToBe = line.trim().split("&");
                            for (String s : attrsToBe) {
                            String[] comps = s.trim().split("=");
                            String attr = comps[0].trim().replace("inform(", "");
                            if (!attrs.containsKey(attr)) {
                            attrs.put(attr, new HashSet<String>());
                            }
                            String value = comps[1].trim().replace(")", "");
                            if (value.startsWith("X-")) {
                            int index = 0;
                            if (!attrXIndeces.containsKey(attr)) {
                            attrXIndeces.put(attr, 1);
                            } else {
                            index = attrXIndeces.get(attr);
                            attrXIndeces.put(attr, index + 1);
                            }
                            value = "X" + index + "";
                            } else if (value.startsWith("\"")) {
                            value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                            }
                            attrs.get(attr.toLowerCase()).add(value.toLowerCase());
                            }
                            testMRs.add(attrs);*/
                            testMRs.add(MRstr);
                        }
                    }
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
                }
                for (DatasetInstance di : datasetInstances) {
                    if (trainMRs.contains(di.getMeaningRepresentation().getMRstr())) {
                        trainingData.add(new DatasetInstance(di));
                    }
                }
                for (DatasetInstance di : datasetInstances) {
                    if (testMRs.contains(di.getMeaningRepresentation().getMRstr())) {
                        testingData.add(new DatasetInstance(di));
                    }
                }

                /*HashSet<DatasetInstance> allData = new HashSet<>();
                allData.addAll(testingData);
                allData.addAll(trainingData);
                System.out.println(abstractDatasetInstances.get("inform").size());
                System.out.println(trainingData.size());
                System.out.println(testingData.size());
                System.out.println(allData.size() + " == " + datasetInstances.size());                
                System.out.println("===============");
                }
                System.exit(0);
                for (int fold = 0; fold < 1; fold += 1) {
                ArrayList<DatasetInstance> testingData = new ArrayList<>();
                ArrayList<DatasetInstance> trainingData = new ArrayList<>();*/
                HashMap<Integer, HashSet<String>> nGrams = null;
                if (!useAlignments) {
                    nGrams = createRandomAlignments(trainingData);
                }

                HashMap<String, HashSet<Action>> availableWordActions = new HashMap<>();
                for (DatasetInstance DI : trainingData) {
                    HashSet<String> mentionedAttributes = new HashSet<>();
                    HashSet<String> mentionedWords = new HashSet<>();
                    for (ArrayList<Action> realization : DI.getEvalRealizations()) {
                        for (Action a : realization) {
                            if (!a.getAttribute().equals(Bagel.TOKEN_END)) {
                                String attr = a.getAttribute().substring(0, a.getAttribute().indexOf('='));
                                mentionedAttributes.add(attr);
                                if (!a.getWord().equals(Bagel.TOKEN_START)
                                        && !a.getWord().equals(Bagel.TOKEN_END)
                                        && !a.getWord().matches("([,.?!;:'])")) {
                                    mentionedWords.add(a.getWord());
                                }
                                if (attr.equals("[]")) {
                                    System.out.println("RR " + realization);
                                    System.out.println("RR " + a);
                                    System.exit(0);
                                }
                            }
                        }
                        for (String attribute : mentionedAttributes) {
                            if (!availableWordActions.containsKey(attribute)) {
                                availableWordActions.put(attribute, new HashSet<Action>());
                                availableWordActions.get(attribute).add(new Action(Bagel.TOKEN_END, attribute));
                            }
                            for (String word : mentionedWords) {
                                if (word.startsWith(Bagel.TOKEN_X)) {
                                    if (word.substring(3, word.lastIndexOf('_')).toLowerCase().trim().equals(attribute)) {
                                        availableWordActions.get(attribute).add(new Action(word, attribute));
                                    }
                                } else {
                                    availableWordActions.get(attribute).add(new Action(word, attribute));
                                }
                            }
                        }
                    }
                }
                Object[] results = createTrainingDatasets(trainingData, availableWordActions, nGrams);
                HashMap<String, ArrayList<Instance>> predicateAttrTrainingData = (HashMap<String, ArrayList<Instance>>) results[0];
                HashMap<String, HashMap<String, ArrayList<Instance>>> predicateWordTrainingData = (HashMap<String, HashMap<String, ArrayList<Instance>>>) results[1];

                boolean setToGo = true;
                if (predicateWordTrainingData.isEmpty() || predicateAttrTrainingData.isEmpty()) {
                    setToGo = false;
                }

                if (setToGo) {
                    if (useDAggerWord) {
                        JDAggerForBagel JDWords = new JDAggerForBagel(this);
                        Object[] LOLSResults = JDWords.runLOLS(predicate, attributes.get(predicate), trainingData, predicateAttrTrainingData.get(predicate), predicateWordTrainingData.get(predicate), availableWordActions, valueAlignments, wordRefRolloutChance, testingData, nGrams);

                        classifiersAttrs.put(predicate, (JAROW) LOLSResults[0]);
                        classifiersWords.put(predicate, (HashMap<String, JAROW>) LOLSResults[1]);
                    } else {
                        classifiersAttrs.put(predicate, train(predicate, predicateAttrTrainingData.get(predicate)));
                        for (String attribute : attributes.get(predicate)) {
                            if (!classifiersWords.containsKey(predicate)) {
                                classifiersWords.put(predicate, new HashMap<String, JAROW>());
                            }
                            if (!predicateWordTrainingData.get(predicate).get(attribute).isEmpty()) {
                                classifiersWords.get(predicate).put(attribute, train(predicate, predicateWordTrainingData.get(predicate).get(attribute)));
                            }
                        }
                    }

                    /*for (String a : classifiersWords.get(predicate).keySet()) {
                    System.out.print(" << " + a + " >> ");
                    ArrayList<String> keys = new ArrayList(classifiersWords.get(predicate).get(a).getCurrentWeightVectors().keySet());
                    Collections.sort(keys);
                    for (String key : keys) {
                    System.out.print(" | " + key + " | ");
                    ArrayList<String> keysB = new ArrayList(classifiersWords.get(predicate).get(a).getCurrentWeightVectors().get(key).keySet());
                    Collections.sort(keysB);
                    for (String kB : keysB) {
                    System.out.print(" (" + kB + " = " + classifiersWords.get(predicate).get(a).getCurrentWeightVectors().get(key).get(kB) + ") ");
                    }
                    }
                    System.out.println();
                    }*/
                    evaluateGeneration(classifiersAttrs.get(predicate), classifiersWords.get(predicate), trainingData, testingData, availableWordActions, nGrams, predicate, true, 10000);
                }
                //RANDOM DATA SPLIT
                //}
            }
            //System.exit(0);
        }
        double finalAvgArgDistance = 0.0;
        for (Double avg : crossAvgArgDistances) {
            finalAvgArgDistance += avg;
        }
        finalAvgArgDistance /= crossAvgArgDistances.size();

        System.out.println("==========================");
        System.out.println("==========================");
        System.out.println("Final avg arg distance: \t" + finalAvgArgDistance);

        double finalNISTDistance = 0.0;
        for (Double avg : crossNIST) {
            finalNISTDistance += avg;
        }
        finalNISTDistance /= crossNIST.size();

        System.out.println("Final NIST: \t" + finalNISTDistance);

        double finalBLEUDistance = 0.0;
        for (Double avg : crossBLEU) {
            finalBLEUDistance += avg;
        }
        finalBLEUDistance /= crossBLEU.size();

        System.out.println("Final BLEU: \t" + finalBLEUDistance);

        double finalBLEUSmoothDistance = 0.0;
        for (Double avg : crossBLEUSmooth) {
            finalBLEUSmoothDistance += avg;
        }
        finalBLEUSmoothDistance /= crossBLEUSmooth.size();

        System.out.println("Final BLEUSmooth: \t" + finalBLEUSmoothDistance);
        System.out.println("==========================");
        System.out.println("==========================");
        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);
    }

    public JAROW trainWords(String predicate, String attribute, ArrayList<DatasetInstance> trainingData) {
        JDAggerForBagel dagger = new JDAggerForBagel(this);
        System.out.println("Run dagger for property " + predicate);
        //JAROW classifier = dagger.runVDAggerForWords(predicate, attribute, trainingData, availableWordActions, 5, 0.7);

        //return classifier;
        return null;
    }

    public JAROW train(String predicate, ArrayList<Instance> trainingData) {
        //NO DAGGER USE
        JAROW classifier = new JAROW();
        //Collections.shuffle(wordInstances);
        //classifierWords.train(wordInstances, true, true, rounds, 0.1, true);
        Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
        //wordInstances = Instance.removeHapaxLegomena(wordInstances);
        //classifierArgs = JAROW.trainOpt(trainingData, rounds, params, 0.2, true, false, 0);
        classifier.train(trainingData, true, false, rounds, 100.0, true);

        return classifier;
    }

    public Double evaluateGeneration(JAROW classifierAttrs, HashMap<String, JAROW> classifierWords, ArrayList<DatasetInstance> trainingData, ArrayList<DatasetInstance> testingData, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams, String predicate) {
        return evaluateGeneration(classifierAttrs, classifierWords, trainingData, testingData, availableWordActions, nGrams, predicate, false, 0);
    }

    public Double evaluateGeneration(JAROW classifierAttrs, HashMap<String, JAROW> classifierWords, ArrayList<DatasetInstance> trainingData, ArrayList<DatasetInstance> testingData, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams, String predicate, boolean printResults, int epoch) {
        System.out.println("Evaluate argument generation " + predicate);

        int totalArgDistance = 0;
        ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
        ArrayList<ArrayList<Action>> generationActions = new ArrayList<>();
        ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
        ArrayList<String> predictedStrings = new ArrayList<>();
        ArrayList<String> predictedStringMRs = new ArrayList<>();
        HashSet<HashMap<String, HashSet<String>>> mentionedAttrs = new HashSet<HashMap<String, HashSet<String>>>();
        for (DatasetInstance di : testingData) {
            ArrayList<Action> predictedActionList = new ArrayList<>();
            ArrayList<Action> predictedWordList = new ArrayList<>();

            //PHRASE GENERATION EVALUATION
            String predictedAttr = "";
            ArrayList<String> predictedAttrValues = new ArrayList<>();
            ArrayList<String> predictedAttributes = new ArrayList<>();

            HashSet<String> attrValuesToBeMentioned = new HashSet<>();
            HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
            HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
            for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                int a = 0;
                for (String value : di.getMeaningRepresentation().getAttributes().get(attribute)) {
                    if (value.startsWith("\"x")) {
                        value = "x" + a;
                        a++;
                    } else if (value.startsWith("\"")) {
                        value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                    }
                    attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
                }
                valuesToBeMentioned.put(attribute, new ArrayList<>(di.getMeaningRepresentation().getAttributes().get(attribute)));
            }
            while (!predictedAttr.equals(Bagel.TOKEN_END) && predictedAttrValues.size() < maxAttrRealizationSize) {
                if (!predictedAttr.isEmpty()) {
                    attrValuesToBeMentioned.remove(predictedAttr);
                }
                Instance attrTrainingVector = Bagel.this.createAttrInstance(predicate, "@TOK@", predictedAttrValues, predictedActionList, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation());

                if (attrTrainingVector != null) {
                    Prediction predictAttr = classifierAttrs.predict(attrTrainingVector);
                    predictedAttr = predictAttr.getLabel().trim();
                    String predictedValue = "";
                    if (!predictedAttr.equals(SFX.TOKEN_END)) {
                        predictedValue = chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);

                        HashSet<String> rejectedAttrs = new HashSet<String>();
                        while (predictedValue.isEmpty() && !predictedAttr.equals(SFX.TOKEN_END)) {
                            rejectedAttrs.add(predictedAttr);

                            predictedAttr = SFX.TOKEN_END;
                            double maxScore = -Double.MAX_VALUE;
                            for (String attr : predictAttr.getLabel2Score().keySet()) {
                                if (!rejectedAttrs.contains(attr)
                                        && (Double.compare(predictAttr.getLabel2Score().get(attr), maxScore) > 0)) {
                                    maxScore = predictAttr.getLabel2Score().get(attr);
                                    predictedAttr = attr;
                                }
                            }
                            if (!predictedAttr.equals(SFX.TOKEN_END)) {
                                predictedValue = chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);
                            }
                        }
                        predictedAttr += "=" + predictedValue;
                    }
                    /*if (predictedAttr.endsWith("=")
                    && !attrValuesToBeMentioned.toString().startsWith("[postcode=x0")) {
                    System.out.println(predictAttr.getLabel2Score());
                    System.out.println(predictAttr.getMostInfluencialFeatures());
                    System.out.println(di.getEvalRealizations());
                    System.out.println(attrValuesToBeMentioned);
                    System.out.print("a: ");
                    String prevAttr = "";
                    for (Action a : predictedActionList) {
                    if (a.getWord().equals(TOKEN_START)
                    || !a.getAttribute().equals(prevAttr)) {
                    System.out.print("{" + a.getAttribute().toUpperCase() + "} ");
                    prevAttr = a.getAttribute();
                    }
                    if (!a.getWord().equals(TOKEN_END)
                    && !a.getWord().equals(TOKEN_START)) {
                    System.out.print(a.getWord().toLowerCase() + " ");
                    }
                    }
                    System.out.println();
                    System.out.println(predictedAttr);
                    System.out.println(attrTrainingVector.getGeneralFeatureVector());
                    System.out.println(attrTrainingVector.getValueSpecificFeatureVector());
                    System.exit(0);
                    }*/
                    /*double maxScore = -Double.MAX_VALUE;
                    String maxValidPrediction = "";
                    for (String attrValue : attrValuesToBeMentioned) {
                    String attr = attrValue.substring(0, attrValue.indexOf('='));
                    if (predictAttr.getLabel2Score().get(attr) > maxScore) {
                    maxScore = predictAttr.getLabel2Score().get(attr);
                    maxValidPrediction = attr;
                    }
                    }
                    if (maxValidPrediction.isEmpty()) {
                    predictedAttr = Bagel.TOKEN_END;
                    } else {
                    predictedAttr = maxValidPrediction.trim() + "=" + chooseNextValue(maxValidPrediction.trim(), attrValuesToBeMentioned, trainingData);
                    }*/
                    predictedAttrValues.add(predictedAttr);

                    String attribute = predictedAttrValues.get(predictedAttrValues.size() - 1).split("=")[0];
                    String attrValue = predictedAttrValues.get(predictedAttrValues.size() - 1);
                    predictedAttributes.add(attrValue);

                    //GENERATE PHRASES
                    if (!attribute.equals(Bagel.TOKEN_END)) {
                        if (classifierWords.containsKey(attribute)) {
                            String predictedWord = "";

                            boolean isValueMentioned = false;
                            String valueTBM = "";
                            if (attrValue.contains("=")) {
                                valueTBM = attrValue.substring(attrValue.indexOf('=') + 1);
                            }
                            if (valueTBM.isEmpty()) {
                                isValueMentioned = true;
                            }
                            ArrayList<String> subPhrase = new ArrayList<>();
                            while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordList.size() < maxWordRealizationSize) {
                                ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                                for (int i = 0; i < predictedAttributes.size() - 1; i++) {
                                    predictedAttributesForInstance.add(predictedAttributes.get(i));
                                }
                                if (!predictedAttributes.get(predictedAttributes.size() - 1).equals(attrValue)) {
                                    predictedAttributesForInstance.add(predictedAttributes.get(predictedAttributes.size() - 1));
                                }
                                Instance wordTrainingVector = createWordInstance(predicate, new Action("@TOK@", attrValue), predictedAttributesForInstance, predictedActionList, isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions, nGrams, false);

                                if (wordTrainingVector != null) {
                                    if (classifierWords.get(attribute) != null) {
                                        Prediction predictWord = classifierWords.get(attribute).predict(wordTrainingVector);
                                        if (predictWord.getLabel() != null) {
                                            predictedWord = predictWord.getLabel().trim();
                                            predictedActionList.add(new Action(predictedWord, attrValue));
                                            if (!predictedWord.equals(Bagel.TOKEN_END)) {
                                                subPhrase.add(predictedWord);
                                                predictedWordList.add(new Action(predictedWord, attrValue));
                                            }
                                        } else {
                                            predictedWord = Bagel.TOKEN_END;
                                            predictedActionList.add(new Action(predictedWord, attrValue));
                                        }
                                    } else {
                                        predictedWord = Bagel.TOKEN_END;
                                        predictedActionList.add(new Action(predictedWord, attrValue));
                                    }
                                }
                                if (!isValueMentioned) {
                                    if (!predictedWord.equals(Bagel.TOKEN_END)) {
                                        if (predictedWord.startsWith(Bagel.TOKEN_X)
                                                && (valueTBM.matches("\"[xX][0-9]+\"")
                                                || valueTBM.matches("[xX][0-9]+"))) {
                                            isValueMentioned = true;
                                        } else if (!predictedWord.startsWith(Bagel.TOKEN_X)
                                                && !(valueTBM.matches("\"[xX][0-9]+\"")
                                                || valueTBM.matches("[xX][0-9]+"))) {
                                            for (ArrayList<String> alignedStr : valueAlignments.get(valueTBM).keySet()) {
                                                if (endsWith(subPhrase, alignedStr)) {
                                                    isValueMentioned = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    if (isValueMentioned) {
                                        attrValuesAlreadyMentioned.add(attrValue);
                                        attrValuesToBeMentioned.remove(attrValue);
                                    }
                                }
                                String mentionedAttrValue = "";
                                if (!predictedWord.startsWith(Bagel.TOKEN_X)) {
                                    for (String attrValueTBM : attrValuesToBeMentioned) {
                                        if (attrValueTBM.contains("=")) {
                                            String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                            if (!(value.matches("\"[xX][0-9]+\"")
                                                    || value.matches("[xX][0-9]+"))) {
                                                for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                                    if (endsWith(subPhrase, alignedStr)) {
                                                        mentionedAttrValue = attrValueTBM;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if (!mentionedAttrValue.isEmpty()) {
                                    attrValuesAlreadyMentioned.add(attrValue);
                                    attrValuesToBeMentioned.remove(mentionedAttrValue);
                                }
                            }
                            if (predictedWordList.size() >= maxWordRealizationSize
                                    && !predictedActionList.get(predictedActionList.size() - 1).getWord().equals(Bagel.TOKEN_END)) {
                                predictedWord = Bagel.TOKEN_END;
                                predictedActionList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                            }
                        } else {
                            String predictedWord = Bagel.TOKEN_END;
                            predictedActionList.add(new Action(predictedWord, attrValue));
                        }
                    }
                }
            }
            ArrayList<String> predictedAttrs = new ArrayList<>();
            for (String attributeValuePair : predictedAttrValues) {
                predictedAttrs.add(attributeValuePair.split("=")[0]);
            }

            ArrayList<Action> cleanActionList = new ArrayList<Action>();
            for (Action action : predictedActionList) {
                if (!action.getWord().equals(Bagel.TOKEN_END)
                        && !action.getWord().equals(Bagel.TOKEN_START)) {
                    cleanActionList.add(action);
                }
            }
            for (int i = 0; i < cleanActionList.size(); i++) {
                for (ArrayList<Action> surrounds : punctPatterns.keySet()) {
                    boolean matches = true;
                    int m = 0;
                    for (int s = 0; s < surrounds.size(); s++) {
                        if (surrounds.get(s) != null) {
                            if (i + s < cleanActionList.size()) {
                                if (!cleanActionList.get(i + s).getWord().equals(surrounds.get(s).getWord()) /*|| !cleanActionList.get(i).getAttribute().equals(surrounds.get(s).getAttribute())*/) {
                                    matches = false;
                                    s = surrounds.size();
                                } else {
                                    m++;
                                }
                            } else {
                                matches = false;
                                s = surrounds.size();
                            }
                        }
                    }
                    if (matches && m > 0) {
                        /*System.out.println("\\o/ WEEEE HAAAVE A MAAATCH \\o/");
                        System.out.println(cleanActionList);
                        System.out.println(surrounds);
                        System.out.println(punctPatterns.get(surrounds));     */
                        cleanActionList.add(i + 2, punctPatterns.get(surrounds));
                        //System.out.println(cleanActionList);
                    }
                }
            }

            String predictedString = "";
            for (Action action : cleanActionList) {
                if (action.getWord().startsWith(Bagel.TOKEN_X)) {
                    predictedString += "x ";
                } else {
                    predictedString += action.getWord() + " ";
                }
            }
            predictedString = predictedString.replaceAll("\\p{Punct}|\\d", "");
            predictedString = predictedString.trim() + ".";
            predictedString = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();

            if (!mentionedAttrs.contains(di.getMeaningRepresentation().getAttributes())) {
                predictedStrings.add(predictedString);
                predictedStringMRs.add(di.getMeaningRepresentation().getMRstr());
                mentionedAttrs.add(di.getMeaningRepresentation().getAttributes());
            }

            Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedString.toLowerCase()));
            ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
            generations.add(tran);
            generationActions.add(predictedActionList);

            ArrayList<Sequence<IString>> references = new ArrayList<>();
            for (ArrayList<Action> realization : di.getEvalRealizations()) {
                String cleanedWords = "";
                for (Action nlWord : realization) {
                    if (!nlWord.equals(new Action(Bagel.TOKEN_START, ""))
                            && !nlWord.equals(new Action(Bagel.TOKEN_END, ""))) {
                        if (nlWord.getWord().startsWith(Bagel.TOKEN_X)) {
                            cleanedWords += "x ";
                        } else {
                            cleanedWords += nlWord.getWord() + " ";
                        }
                    }
                }
                //UNCOMMENTED THE REPLACES HERE
                //cleanedWords = cleanedWords.replaceAll("\\p{Punct}|\\d", "");
                cleanedWords = cleanedWords.trim();
                if (!cleanedWords.endsWith(".")) {
                    cleanedWords += ".";
                }
                cleanedWords = cleanedWords.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                references.add(IStrings.tokenize(NISTTokenizer.tokenize(cleanedWords)));
            }
            finalReferences.add(references);

            //EVALUATE ATTRIBUTE SEQUENCE
            HashSet<ArrayList<String>> goldAttributeSequences = new HashSet<>();
            for (DatasetInstance di2 : testingData) {
                if (di2.getMeaningRepresentation().getAttributes().equals(di.getMeaningRepresentation().getAttributes())) {
                    goldAttributeSequences.addAll(di2.getEvalMentionedAttributeSequences().values());
                }
            }

            //for (ArrayList<String> goldArgs : abstractMeaningReprs.get(predicate).get(mr).values()) {
            int minTotArgDistance = Integer.MAX_VALUE;
            ArrayList<String> minGoldArgs = null;
            for (ArrayList<String> goldArgs : goldAttributeSequences) {
                int totArgDistance = 0;
                HashSet<Integer> matchedPositions = new HashSet<>();
                for (int i = 0; i < predictedAttrs.size(); i++) {
                    if (!predictedAttrs.get(i).equals(Bagel.TOKEN_START)
                            && !predictedAttrs.get(i).equals(Bagel.TOKEN_END)) {
                        int minArgDistance = Integer.MAX_VALUE;
                        int minArgPos = -1;
                        for (int j = 0; j < goldArgs.size(); j++) {
                            if (!matchedPositions.contains(j)) {
                                if (goldArgs.get(j).equals(predictedAttrs.get(i))) {
                                    int argDistance = Math.abs(j - i);

                                    if (argDistance < minArgDistance) {
                                        minArgDistance = argDistance;
                                        minArgPos = j;
                                    }
                                }
                            }
                        }

                        if (minArgPos == -1) {
                            totArgDistance += 100;
                        } else {
                            matchedPositions.add(minArgPos);
                            totArgDistance += minArgDistance;
                        }
                    }
                }
                /*if (totArgDistance > 100) {
                System.out.println(predictedAttrs);
                System.out.println(goldAttributeSequences);
                System.out.println(totArgDistance);
                System.out.println(predictedActionList);
                System.exit(0);                    
                }*/
                ArrayList<String> predictedCopy = (ArrayList<String>) predictedAttrs.clone();
                for (String goldArg : goldArgs) {
                    if (!goldArg.equals(Bagel.TOKEN_END)) {
                        boolean contained = predictedCopy.remove(goldArg);
                        if (!contained) {
                            totArgDistance += 1000;
                        }
                    }
                }
                if (totArgDistance < minTotArgDistance) {
                    minTotArgDistance = totArgDistance;
                    minGoldArgs = goldArgs;
                }
            }
            /*System.out.println("PS: " + predictedString);
            System.out.println("R: " + references);
            System.out.println("M: " + di.getMeaningRepresentation().getAttributes());
            System.out.println("GGG: " + mentionedValueSequence);
            System.out.println("PV: " + predictedAttrValuePairs);
            System.out.println("P: " + predictedAttrs);
            System.out.println("G: " + minGoldArgs);
            System.out.println("Distance: " + minTotArgDistance);
            System.out.println("==============");*/
            totalArgDistance += minTotArgDistance;
        }
        crossAvgArgDistances.add(totalArgDistance / (double) testingData.size());

        NISTMetric NIST = new NISTMetric(finalReferences);
        BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
        BLEUMetric BLEUsmooth = new BLEUMetric(finalReferences, 4, true);
        Double nistScore = NIST.score(generations);
        Double bleuScore = BLEU.score(generations);
        Double bleuSmoothScore = BLEUsmooth.score(generations);

        crossNIST.add(nistScore);
        crossBLEU.add(bleuScore);
        crossBLEUSmooth.add(bleuSmoothScore);
        System.out.println("Avg arg distance: \t" + totalArgDistance / (double) testingData.size());
        System.out.println("NIST: \t" + nistScore);
        System.out.println("BLEU: \t" + bleuScore);
        System.out.println("g: " + generations);
        System.out.println("BLEU smooth: \t" + bleuSmoothScore);

        if (printResults) {
            BufferedWriter bw = null;
            File f = null;
            try {
                f = new File("BAGELTextsAt " + fold + "FoldAfter" + (epoch + 1) + "epochs" + "_" + JDAggerForBagel.earlyStopMaxFurtherSteps + "_" + JDAggerForBagel.p + ".txt");
            } catch (NullPointerException e) {
                System.err.println("File not found." + e);
            }

            try {
                bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
            } catch (FileNotFoundException e) {
                System.err.println("Error opening file for writing! " + e);
            }

            try {
                bw.write("BLEU:" + bleuScore);
                bw.write("\n");
            } catch (IOException e) {
                System.err.println("Write error!");
            }
            for (int i = 0; i < predictedStrings.size(); i++) {
                try {
                    //Grafoume to String sto arxeio
                    bw.write("MR," + predictedStringMRs.get(i)+",");
                    bw.write("BAGEL,");
                    bw.write(predictedStrings.get(i) + ",");

                    bw.write("\n");
                } catch (IOException e) {
                    System.err.println("Write error!");
                }
            }

            try {
                bw.close();
            } catch (IOException e) {
                System.err.println("Error closing file.");
            } catch (Exception e) {
            }
        }
        return bleuScore;
    }

    public void createLists(File dataFile) {
        predicates = new ArrayList<>();
        attributes = new HashMap<>();
        attributeValuePairs = new HashMap<>();
        valueAlignments = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {
            String line;

            String previousPredicate = null;
            MeaningRepresentation previousMR;
            MeaningRepresentation previousAMR = null;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("FULL_DA")) {
                    line = line.substring(10);

                    previousPredicate = line.substring(0, line.indexOf("("));
                    if (!predicates.contains(previousPredicate) && previousPredicate != null) {
                        predicates.add(previousPredicate);

                        /*if (!argDictionary.containsKey(predicate)) {
                        argDictionary.put(predicate, new ArrayList<>());
                        }*/
                        if (!attributes.containsKey(previousPredicate)) {
                            attributes.put(previousPredicate, new HashSet<String>());
                        }
                        if (!attributeValuePairs.containsKey(previousPredicate)) {
                            attributeValuePairs.put(previousPredicate, new HashSet<String>());
                        }
                        if (!meaningReprs.containsKey(previousPredicate)) {
                            meaningReprs.put(previousPredicate, new HashMap<MeaningRepresentation, HashSet<String>>());
                        }
                        if (!abstractDatasetInstances.containsKey(previousPredicate)) {
                            abstractDatasetInstances.put(previousPredicate, new ArrayList<DatasetInstance>());
                        }
                    }

                    line = line.substring(line.indexOf("(") + 1, line.lastIndexOf(")"));

                    String MRstr = new String(line);
                    HashMap<String, String> names = new HashMap<>();
                    int s = line.indexOf("\"");
                    int a = 0;
                    while (s != -1) {
                        int e = line.indexOf("\"", s + 1);

                        String name = line.substring(s, e + 1);
                        //line = line.replace(name, "@@$$" + a + "$$@@");
                        //names.put("@@$$" + a + "$$@@", name);
                        line = line.replace(name, "x" + a);
                        names.put("x" + a, name);
                        a++;

                        s = line.indexOf("\"");
                    }

                    HashMap<String, HashSet<String>> argumentValues = new HashMap<>();
                    String[] args = line.split(",");

                    for (String arg : args) {
                        String[] subArg = arg.split("=");
                        String value = subArg[1];
                        if (names.containsKey(value)) {
                            value = names.get(value);
                        }
                        attributes.get(previousPredicate).add(subArg[0].toLowerCase());
                        if (!argumentValues.containsKey(subArg[0])) {
                            argumentValues.put(subArg[0].toLowerCase(), new HashSet<String>());
                        }
                        argumentValues.get(subArg[0].toLowerCase()).add(value.toLowerCase());
                        attributeValuePairs.get(previousPredicate).add(arg.toLowerCase());
                    }

                    /*                    
                    for (String arg : args) {
                    if (!arguments.get(predicate).contains(arg)) {
                    arguments.get(predicate).add(arg);
                    }
                    if (!argDictionaryMap.containsKey(arg)) {
                    argDictionaryMap.put(arg, new HashMap<>());
                    }
                    }
                     */
                    previousMR = new MeaningRepresentation(previousPredicate, argumentValues, MRstr);
                    if (!meaningReprs.get(previousPredicate).containsKey(previousMR)) {
                        meaningReprs.get(previousPredicate).put(previousMR, new HashSet<String>());
                    }
                } else if (line.startsWith("ABSTRACT_DA")) {
                    line = line.substring(14);

                    previousPredicate = line.substring(0, line.indexOf("("));
                    if (!predicates.contains(previousPredicate) && previousPredicate != null) {
                        predicates.add(previousPredicate);

                        /*if (!argDictionary.containsKey(predicate)) {
                        argDictionary.put(predicate, new ArrayList<>());
                        }*/
                        if (!attributes.containsKey(previousPredicate)) {
                            attributes.put(previousPredicate, new HashSet<String>());
                        }
                        if (!meaningReprs.containsKey(previousPredicate)) {
                            meaningReprs.put(previousPredicate, new HashMap<MeaningRepresentation, HashSet<String>>());
                        }
                        if (!abstractDatasetInstances.containsKey(previousPredicate)) {
                            abstractDatasetInstances.put(previousPredicate, new ArrayList<DatasetInstance>());
                        }
                    }

                    line = line.substring(line.indexOf("(") + 1, line.lastIndexOf(")"));
                    String MRstr = new String(line);

                    HashMap<String, String> names = new HashMap<>();
                    int s = line.indexOf("\"");
                    int a = 0;
                    while (s != -1) {
                        int e = line.indexOf("\"", s + 1);

                        String name = line.substring(s, e + 1);
                        line = line.replace(name, "x" + a);
                        names.put("x" + a, name);
                        a++;

                        s = line.indexOf("\"");
                    }

                    HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
                    String[] args = line.split(",");

                    HashMap<String, Integer> attrXIndeces = new HashMap<>();
                    for (String arg : args) {
                        String[] subAttr = arg.split("=");
                        String value = subAttr[1];
                        if (names.containsKey(value)) {
                            value = names.get(value);
                        }
                        String attr = subAttr[0].toLowerCase();
                        if (!attributes.get(previousPredicate).contains(attr)) {
                            attributes.get(previousPredicate).add(attr);
                        }
                        if (!attributeValues.containsKey(attr)) {
                            attributeValues.put(attr, new HashSet<String>());
                        }
                        if (value.startsWith("\"")) {
                            value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                        }
                        if (value.toLowerCase().startsWith("x")) {
                            int index = 0;
                            if (!attrXIndeces.containsKey(attr)) {
                                attrXIndeces.put(attr, 1);
                            } else {
                                index = attrXIndeces.get(attr);
                                attrXIndeces.put(attr, index + 1);
                            }
                            value = "x" + index;
                        }
                        attributeValues.get(attr).add(value.toLowerCase());
                    }
                    /*
                    for (String arg : args) {
                    if (!arguments.get(predicate).contains(arg)) {
                    arguments.get(predicate).add(arg);
                    }
                    if (!argDictionaryMap.containsKey(arg)) {
                    argDictionaryMap.put(arg, new HashMap<>());
                    }
                    }
                     */
                    previousAMR = new MeaningRepresentation(previousPredicate, attributeValues, MRstr);
                } else if (line.startsWith("->")) {
                    line = line.substring(line.indexOf("\"") + 1, line.lastIndexOf("\""));

                    ArrayList<String> mentionedValueSequence = new ArrayList<>();
                    ArrayList<String> mentionedAttributeSequence = new ArrayList<>();

                    ArrayList<String> realization = new ArrayList<>();
                    ArrayList<String> alignedRealization = new ArrayList<>();

                    String[] words = line.replaceAll("([,.?!;:'])", " $1").split(" ");
                    HashMap<String, Integer> attributeXCount = new HashMap<>();
                    for (int i = 0; i < words.length; i++) {
                        boolean isEmptyAttr = false;
                        String mentionedAttribute = "";
                        if (!words[i].trim().isEmpty()) {
                            if (words[i].trim().startsWith("[]")) {
                                isEmptyAttr = true;
                            }
                            int s = words[i].indexOf("[");
                            if (s != -1) {
                                int e = words[i].indexOf("]", s + 1);

                                String mentionedValue = words[i].substring(s, e + 1);
                                words[i] = words[i].replace(mentionedValue, "");
                                if (mentionedValue.contains("+") && !words[i].trim().isEmpty()) {
                                    mentionedAttribute = mentionedValue.substring(1, mentionedValue.indexOf("+"));

                                    if (previousAMR.getAttributes().containsKey(mentionedAttribute)) {
                                        if (mentionedValueSequence.isEmpty()) {
                                            String v = mentionedValue.substring(1, mentionedValue.length() - 1).replaceAll("\\+", "=");
                                            if (v.endsWith("=X")) {
                                                //v = v.replace("=X", "=@@$$" + a + "$$@@");
                                                int a = 0;
                                                if (!attributeXCount.containsKey(mentionedAttribute)) {
                                                    attributeXCount.put(mentionedAttribute, 1);
                                                } else {
                                                    a = attributeXCount.get(mentionedAttribute);
                                                    attributeXCount.put(mentionedAttribute, attributeXCount.get(mentionedAttribute) + 1);
                                                }
                                                v = v.replace("=X", "=x" + a);
                                            }
                                            mentionedValueSequence.add(v.toLowerCase());
                                        } else if (!mentionedValueSequence.get(mentionedValueSequence.size() - 1).equals(mentionedValue)) {
                                            String v = mentionedValue.substring(1, mentionedValue.length() - 1).replaceAll("\\+", "=");
                                            if (v.endsWith("=X")) {
                                                //v = v.replace("=X", "=@@$$" + +a + "$$@@");
                                                int a = 0;
                                                if (!attributeXCount.containsKey(mentionedAttribute)) {
                                                    attributeXCount.put(mentionedAttribute, 1);
                                                } else {
                                                    a = attributeXCount.get(mentionedAttribute);
                                                    attributeXCount.put(mentionedAttribute, attributeXCount.get(mentionedAttribute) + 1);
                                                }
                                                v = v.replace("=X", "=x" + a);
                                            }
                                            mentionedValueSequence.add(v.toLowerCase());
                                        }

                                        if (mentionedAttributeSequence.isEmpty()) {
                                            mentionedAttributeSequence.add(mentionedAttribute.toLowerCase());
                                        } else if (!mentionedAttributeSequence.get(mentionedAttributeSequence.size() - 1).equals(mentionedAttribute)) {
                                            mentionedAttributeSequence.add(mentionedAttribute.toLowerCase());
                                        }
                                    }
                                } else if (!words[i].trim().isEmpty()) {
                                    mentionedAttribute = mentionedValue.substring(1, mentionedValue.length() - 1);

                                    if (!previousAMR.getAttributes().containsKey(mentionedAttribute)) {
                                        mentionedAttribute = "";
                                    }
                                }

                                //s = line.indexOf("[");
                            }
                            if (!words[i].trim().isEmpty()) {
                                if (useAlignments) {
                                    if (words[i].trim().matches("[,.?!;:']")) {
                                        alignedRealization.add(Bagel.TOKEN_PUNCT);
                                    } else if (isEmptyAttr) {
                                        alignedRealization.add("[]");
                                    } else {
                                        alignedRealization.add(mentionedAttribute);
                                    }
                                }
                                //if (!words[i].trim().matches("[,.?!;:']")) {
                                if (words[i].trim().equalsIgnoreCase("x")) {
                                    //realization.add(Bagel.TOKEN_X + mentionedAttribute + "_" + (attributeXCount.get(mentionedAttribute) - 1));
                                    realization.add(Bagel.TOKEN_X + mentionedAttribute + "_" + (attributeXCount.get(mentionedAttribute) - 1));
                                } else {
                                    realization.add(words[i].trim().toLowerCase());
                                }
                            }
                            //}
                        }
                    }

                    /*ArrayList<String> bAl = new ArrayList<>();
                    String[] bWs = bestAlignment[1].split(" ");
                    for (int i = 0; i < bWs.length; i++) {
                    bAl.add(bWs[i]);
                    }
                    if (!valueAlignments.containsKey(value)) {
                    valueAlignments.put(value, new HashSet());
                    }
                    valueAlignments.get(value).add(bAl);*/
                    for (String attr : previousAMR.getAttributes().keySet()) {
                        for (String value : previousAMR.getAttributes().get(attr)) {
                            if (attr.equals("name") && value.equals("none")) {
                                mentionedValueSequence.add(0, attr.toLowerCase() + "=" + value.toLowerCase());
                                mentionedAttributeSequence.add(0, attr.toLowerCase());

                                if (useAlignments) {
                                    for (int i = 0; i < alignedRealization.size(); i++) {
                                        if (alignedRealization.get(i).isEmpty() || alignedRealization.get(i).equals("[]")) {
                                            alignedRealization.set(i, "name");
                                        } else {
                                            i = alignedRealization.size();
                                        }
                                    }
                                }
                            }
                            /*if (value.matches("\"X\\d+\"")) {
                            value = "X";
                            }
                            value = value.replaceAll("\"", "").replaceAll(" ", "_");
                            if (!(mentionedValues.contains("[" + arg + "+" + value + "]"))) {
                            if (!value.equals("placetoeat")) 
                            System.out.println("missing value: " + arg + "+"+ value);
                            }*/
                        }
                    }

                    mentionedValueSequence.add(Bagel.TOKEN_END);
                    mentionedAttributeSequence.add(Bagel.TOKEN_END);

                    if (realization.size() > maxWordRealizationSize) {
                        maxWordRealizationSize = realization.size();
                    }

                    if (useAlignments) {
                        String previousAttr = "";
                        for (int i = 0; i < alignedRealization.size(); i++) {
                            if (alignedRealization.get(i).isEmpty()) {
                                if (!previousAttr.isEmpty()) {
                                    alignedRealization.set(i, previousAttr);
                                }
                            } else if (!alignedRealization.get(i).equals(Bagel.TOKEN_PUNCT)) {
                                previousAttr = alignedRealization.get(i);
                            } else {
                                previousAttr = "";
                            }
                        }
                    } else {
                        for (String word : realization) {
                            if (word.trim().matches("[,.?!;:']")) {
                                alignedRealization.add(Bagel.TOKEN_PUNCT);
                            } else {
                                alignedRealization.add("[]");
                            }
                        }
                    }

                    //Calculate alignments
                    HashMap<String, HashMap<String, Double>> alignments = new HashMap<>();
                    for (String attr : previousAMR.getAttributes().keySet()) {
                        for (String value : previousAMR.getAttributes().get(attr)) {
                            if (!value.equals("name=none") && !(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+"))) {
                                alignments.put(value, new HashMap<String, Double>());
                                //For all ngrams
                                for (int n = 1; n < realization.size(); n++) {
                                    //Calculate all alignment similarities
                                    for (int i = 0; i <= realization.size() - n; i++) {
                                        boolean pass = true;
                                        for (int j = 0; j < n; j++) {
                                            if (realization.get(i + j).startsWith(Bagel.TOKEN_X)
                                                    || alignedRealization.get(i + j).equals(Bagel.TOKEN_PUNCT)
                                                    || StringNLPUtilities.isArticle(realization.get(i + j))
                                                    || StringNLPUtilities.isPreposition(realization.get(i + j))
                                                    || realization.get(i + j).equalsIgnoreCase("and")
                                                    || realization.get(i + j).equalsIgnoreCase("or")
                                                    || (useAlignments && !alignedRealization.get(i + j).equals(attr))) {
                                                pass = false;
                                            }
                                        }
                                        if (pass) {
                                            String align = "";
                                            String compare = "";
                                            for (int j = 0; j < n; j++) {
                                                align += (i + j) + " ";
                                                compare += realization.get(i + j);
                                            }
                                            align = align.trim();

                                            Double distance = Levenshtein.getSimilarity(value.toLowerCase(), compare.toLowerCase(), true, false);
                                            if (distance > 0.3) {
                                                alignments.get(value).put(align, distance);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    HashSet<String> toRemove = new HashSet<>();
                    for (String value : alignments.keySet()) {
                        if (alignments.get(value).isEmpty()) {
                            toRemove.add(value);
                        }
                    }
                    for (String value : toRemove) {
                        alignments.remove(value);
                    }

                    while (!alignments.keySet().isEmpty()) {
                        Double max = Double.NEGATIVE_INFINITY;
                        String[] bestAlignment = new String[2];
                        for (String value : alignments.keySet()) {
                            for (String alignment : alignments.get(value).keySet()) {
                                if (alignments.get(value).get(alignment) > max) {
                                    max = alignments.get(value).get(alignment);
                                    bestAlignment[0] = value;
                                    bestAlignment[1] = alignment;
                                }
                            }
                        }

                        ArrayList<String> alignedStr = new ArrayList<>();
                        String[] coords = bestAlignment[1].split(" ");

                        if (coords.length == 1) {
                            alignedStr.add(realization.get(Integer.parseInt(coords[0].trim())));
                        } else {
                            for (int i = Integer.parseInt(coords[0].trim()); i <= Integer.parseInt(coords[coords.length - 1].trim()); i++) {
                                alignedStr.add(realization.get(i));
                            }
                        }

                        if (!valueAlignments.containsKey(bestAlignment[0])) {
                            valueAlignments.put(bestAlignment[0], new HashMap<ArrayList<String>, Double>());
                        }
                        valueAlignments.get(bestAlignment[0]).put(alignedStr, max);

                        alignments.remove(bestAlignment[0]);
                        for (String value : alignments.keySet()) {
                            HashSet<String> alignmentsToBeRemoved = new HashSet<>();
                            for (String alignment : alignments.get(value).keySet()) {
                                String[] othCoords = alignment.split(" ");
                                if (Integer.parseInt(coords[0].trim()) <= Integer.parseInt(othCoords[0].trim()) && (Integer.parseInt(coords[coords.length - 1].trim()) >= Integer.parseInt(othCoords[0].trim()))
                                        || (Integer.parseInt(othCoords[0].trim()) <= Integer.parseInt(coords[0].trim()) && Integer.parseInt(othCoords[othCoords.length - 1].trim()) >= Integer.parseInt(coords[0].trim()))) {
                                    alignmentsToBeRemoved.add(alignment);
                                }
                            }
                            for (String alignment : alignmentsToBeRemoved) {
                                alignments.get(value).remove(alignment);
                            }
                        }
                        toRemove = new HashSet<>();
                        for (String value : alignments.keySet()) {
                            if (alignments.get(value).isEmpty()) {
                                toRemove.add(value);
                            }
                        }
                        for (String value : toRemove) {
                            alignments.remove(value);
                        }
                    }
                    String previousAttr = "";
                    for (int i = alignedRealization.size() - 1; i >= 0; i--) {
                        if (alignedRealization.get(i).isEmpty() || alignedRealization.get(i).equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                alignedRealization.set(i, previousAttr);
                            }
                        } else if (!alignedRealization.get(i).equals(Bagel.TOKEN_PUNCT)) {
                            previousAttr = alignedRealization.get(i);
                        } else {
                            previousAttr = "";
                        }
                    }
                    previousAttr = "";
                    for (int i = 0; i < alignedRealization.size(); i++) {
                        if (alignedRealization.get(i).isEmpty() || alignedRealization.get(i).equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                alignedRealization.set(i, previousAttr);
                            }
                        } else if (!alignedRealization.get(i).equals(Bagel.TOKEN_PUNCT)) {
                            previousAttr = alignedRealization.get(i);
                        } else {
                            previousAttr = "";
                        }
                    }
                    /*
                    System.out.println("===============");
                    System.out.println(realization);
                    System.out.println(alignedRealization);
                    String previousAlignment = "";
                    for (int i = 0; i < realization.size(); i++) {
                    String currentAlignment = alignedRealization.get(i);
                    if (!previousAlignment.isEmpty() && !currentAlignment.equals(previousAlignment) && !previousAlignment.equals(Bagel.TOKEN_PUNCT)) {
                    realization.add(i, Bagel.TOKEN_END);
                    alignedRealization.add(i, previousAlignment);                            
                    i++;                            
                    }
                    previousAlignment = currentAlignment;
                    }
                    if (!previousAlignment.equals(Bagel.TOKEN_PUNCT)) {
                    realization.add(Bagel.TOKEN_END);
                    alignedRealization.add(previousAlignment);
                    }
                    System.out.println("====");
                    System.out.println(realization);
                    System.out.println(alignedRealization);
                    System.out.println("===============");*/

                    /*for (int i = 0; i < alignedRealization.size(); i++) {
                    if (alignedRealization.get(i).equals(Bagel.TOKEN_PUNCT)) {
                    alignedRealization.set(i, previousAlignment);
                    }
                    
                    previousAlignment = alignedRealization.get(i);
                    }*/

                    /*HashSet<String> dupliCheck = new HashSet(mentionedAttributeSequence);
                    //if (dupliCheck.size() < mentionedAttributeSequence.size()) {
                    System.out.println(previousAMR.getAttributes());
                    System.out.println(pureLine);
                    System.out.println(realization);
                    System.out.println(alignedRealization);
                    System.out.println(mentionedAttributeSequence);
                    System.out.println("================");*/
                    //}
                    ArrayList<Action> realizationActions = new ArrayList<>();
                    for (int i = 0; i < realization.size(); i++) {
                        realizationActions.add(new Action(realization.get(i), alignedRealization.get(i)));
                    }

                    boolean existing = false;
                    for (DatasetInstance existingDI : abstractDatasetInstances.get(previousPredicate)) {
                        //if (existingDI.getMeaningRepresentation().equals(previousAMR)) {
                        if (existingDI.getMeaningRepresentation().getMRstr().equals(previousAMR.getMRstr())) {
                            existing = true;
                            existingDI.mergeDatasetInstance(mentionedValueSequence, mentionedAttributeSequence, realizationActions);
                        }
                    }
                    if (!existing) {
                        DatasetInstance DI = new DatasetInstance(previousAMR, mentionedValueSequence, mentionedAttributeSequence, realizationActions);
                        abstractDatasetInstances.get(previousPredicate).add(DI);
                    }
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /*public HashMap<String, ArrayList<Instance>> createAttrTrainingDatasets(ArrayList<DatasetInstance> trainingData, HashMap<String, HashSet<Action>> availableWordActions) {
    
    if (!availableWordActions.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*//*) {
    /*   for (String predicate : predicates) {
    predicateArgTrainingData.put(predicate, new ArrayList<Instance>());
    for (DatasetInstance di : trainingData) {
    for (ArrayList<String> mentionedAttrValueSequence : di.getEvalMentionedValueSequences().values()) {
    //For every mentioned argument in realization
    HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
    HashSet<String> attrValuesToBeMentioned = new HashSet<>();
    for (String attr : di.getMeaningRepresentation().getAttributes().keySet()) {
    int a = 0;
    for (String value : di.getMeaningRepresentation().getAttributes().get(attr)) {
    if (value.startsWith("\"x")
    || value.startsWith("\"X")) {
    value = "x" + a;
    a++;
    } else if (value.startsWith("\"")) {
    value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
    }
    attrValuesToBeMentioned.add(attr + "=" + value);
    }
    }
    
    for (int w = 0; w < mentionedAttrValueSequence.size(); w++) {
    }
    }
    }
    }
    }
    return predicateArgTrainingData;
    }*/

    public Object[] createTrainingDatasets(ArrayList<DatasetInstance> trainingData, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams) {
        HashMap<String, ArrayList<Instance>> predicateAttrTrainingData = new HashMap<>();
        HashMap<String, HashMap<String, ArrayList<Instance>>> predicateWordTrainingData = new HashMap<>();

        if (!availableWordActions.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*/) {
            for (String predicate : predicates) {
                predicateAttrTrainingData.put(predicate, new ArrayList<Instance>());
                predicateWordTrainingData.put(predicate, new HashMap<String, ArrayList<Instance>>());

                for (DatasetInstance di : trainingData) {
                    for (ArrayList<Action> realization : di.getEvalRealizations()) {
                        HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
                        HashSet<String> attrValuesToBeMentioned = new HashSet<>();
                        for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                            int a = 0;
                            for (String value : di.getMeaningRepresentation().getAttributes().get(attribute)) {
                                if (value.startsWith("\"x")) {
                                    value = "x" + a;
                                    a++;
                                } else if (value.startsWith("\"")) {
                                    value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                                }
                                attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
                            }
                        }

                        ArrayList<String> attrs = new ArrayList<>();
                        boolean isValueMentioned = false;
                        String valueTBM = "";
                        String attrValue = "";
                        ArrayList<String> subPhrase = new ArrayList<>();
                        for (int w = 0; w < realization.size(); w++) {
                            if (!realization.get(w).getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                                if (!realization.get(w).getAttribute().equals(attrValue)) {
                                    if (!attrValue.isEmpty()) {
                                        attrValuesToBeMentioned.remove(attrValue);
                                    }
                                    Instance attrTrainingVector = Bagel.this.createAttrInstance(predicate, realization.get(w).getAttribute(), attrs, new ArrayList<Action>(realization.subList(0, w)), attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation());
                                    if (attrTrainingVector != null) {
                                        predicateAttrTrainingData.get(predicate).add(attrTrainingVector);
                                    }
                                    attrs.add(realization.get(w).getAttribute());

                                    attrValue = realization.get(w).getAttribute();
                                    subPhrase = new ArrayList<>();
                                    isValueMentioned = false;
                                    valueTBM = "";
                                    if (attrValue.contains("=")) {
                                        valueTBM = attrValue.substring(attrValue.indexOf('=') + 1);
                                    }
                                    if (valueTBM.isEmpty()) {
                                        isValueMentioned = true;
                                    }
                                }
                                if (!attrValue.equals(Bagel.TOKEN_END)) {
                                    ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                                    for (int i = 0; i < attrs.size() - 1; i++) {
                                        predictedAttributesForInstance.add(attrs.get(i));
                                    }
                                    if (!attrs.get(attrs.size() - 1).equals(attrValue)) {
                                        predictedAttributesForInstance.add(attrs.get(attrs.size() - 1));
                                    }
                                    Instance wordTrainingVector = createWordInstance(predicate, realization.get(w), predictedAttributesForInstance, new ArrayList<Action>(realization.subList(0, w)), isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions, nGrams, false);

                                    if (wordTrainingVector != null) {
                                        String attribute = attrValue.substring(0, attrValue.indexOf('='));
                                        if (!predicateWordTrainingData.get(predicate).containsKey(attribute)) {
                                            predicateWordTrainingData.get(predicate).put(attribute, new ArrayList<Instance>());
                                        }
                                        predicateWordTrainingData.get(predicate).get(attribute).add(wordTrainingVector);
                                        if (!realization.get(w).getWord().equals(Bagel.TOKEN_START)
                                                && !realization.get(w).getWord().equals(Bagel.TOKEN_END)) {
                                            subPhrase.add(realization.get(w).getWord());
                                        }
                                    }
                                    if (!isValueMentioned) {
                                        if (realization.get(w).getWord().startsWith(Bagel.TOKEN_X)
                                                && (valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\""))) {
                                            isValueMentioned = true;
                                        } else if (!realization.get(w).getWord().startsWith(Bagel.TOKEN_X)
                                                && !(valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\""))) {
                                            for (ArrayList<String> alignedStr : valueAlignments.get(valueTBM).keySet()) {
                                                if (endsWith(subPhrase, alignedStr)) {
                                                    isValueMentioned = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if (isValueMentioned) {
                                            attrValuesAlreadyMentioned.add(attrValue);
                                            attrValuesToBeMentioned.remove(attrValue);
                                        }
                                    }
                                    String mentionedAttrValue = "";
                                    if (!realization.get(w).getWord().startsWith(Bagel.TOKEN_X)) {
                                        for (String attrValueTBM : attrValuesToBeMentioned) {
                                            if (attrValueTBM.contains("=")) {
                                                String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                                if (!(value.matches("\"[xX][0-9]+\"")
                                                        || value.matches("[xX][0-9]+"))) {
                                                    for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                                        if (endsWith(subPhrase, alignedStr)) {
                                                            mentionedAttrValue = attrValueTBM;
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if (!mentionedAttrValue.isEmpty()) {
                                        attrValuesAlreadyMentioned.add(mentionedAttrValue);
                                        attrValuesToBeMentioned.remove(mentionedAttrValue);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Object[] results = new Object[2];
        results[0] = predicateAttrTrainingData;
        results[1] = predicateWordTrainingData;
        return results;
    }

    /*public HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> createRandomWordTrainingDatasets() {
    HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData = new HashMap<>();
    
    if (!availableWordActions.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*//*) {
    for (String predicate : predicates) {
    predicateWordTrainingData.put(predicate, new HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>());
    
    ArrayList<String> attrs = new ArrayList<>();
    for (DatasetInstance di : abstractDatasetInstances.get(predicate)) {
    HashMap<String, ArrayList<Instance>> instances = new HashMap<>();
    for (ArrayList<Action> real : di.getEvalRealizations()) {
    ArrayList<String> attributes = new ArrayList<String>();
    for (Action act : real) {
    if (attributes.isEmpty()) {
    attributes.add(act.getAttribute());
    } else if (!attributes.get(attributes.size() - 1).equals(act.getAttribute())) {
    attributes.add(act.getAttribute());
    }
    }
    ArrayList<Action> realization = new ArrayList<Action>();
    ArrayList<Action> availableActions = new ArrayList(availableWordActions.get(Bagel.TOKEN_ATTR));
    for (String at : attributes) {
    if (!at.equals(Bagel.TOKEN_PUNCT)) {
    Action act = new Action("", "");
    int c = 0;
    while (!act.getWord().equals(Bagel.TOKEN_START)
    && !act.getWord().equals(Bagel.TOKEN_END)) {
    c++;
    
    if (c < 15) {
    act = availableActions.get(r.nextInt(availableActions.size()));
    
    if (!realization.isEmpty()) {
    if (act.getWord().equals(realization.get(realization.size() - 1).getWord())) {
    while (act.getWord().equals(realization.get(realization.size() - 1).getWord())) {
    act = availableActions.get(r.nextInt(availableActions.size()));
    }
    }
    }
    realization.add(new Action(act.getWord(), at));
    } else {
    act = new Action(Bagel.TOKEN_END, at);
    realization.add(new Action(Bagel.TOKEN_END, at));
    }
    }
    }
    }
    
    HashMap<String, HashSet<String>> values = new HashMap();
    for (String attr : di.getMeaningRepresentation().getAttributes().keySet()) {
    values.put(attr, new HashSet(di.getMeaningRepresentation().getAttributes().get(attr)));
    }
    boolean isValueMentioned = false;
    String valueTBM = "";
    String previousAlignment = "";
    ArrayList<String> subPhrase = new ArrayList<>();
    for (int w = 0; w < realization.size(); w++) {
    if (!realization.get(w).getAttribute().equals(Bagel.TOKEN_PUNCT)) {
    if (!realization.get(w).getAttribute().equals(previousAlignment)) {
    attrs.add(realization.get(w).getAttribute());
    
    previousAlignment = realization.get(w).getAttribute();
    subPhrase = new ArrayList<>();
    isValueMentioned = false;
    valuesTF = new ArrayList<>();
    valueTBM = "";
    if (previousAlignment.contains("=")) {
    valueTBM = previousAlignment.substring(previousAlignment.indexOf("=") + 1);
    } else {
    isValueMentioned = true;
    }
    }
    ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
    for (int i = 0; i < attrs.size() - 1; i++) {
    predictedAttributesForInstance.add(attrs.get(i));
    }
    if (!attrs.get(attrs.size() - 1).equals(previousAlignment)) {
    predictedAttributesForInstance.add(attrs.get(attrs.size() - 1));
    }
    Instance wordTrainingVector = createWordInstance(predicate, predictedAttributesForInstance, realization, w, valueTBM, isValueMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), true);
    
    if (wordTrainingVector != null) {
    if (!instances.containsKey(previousAlignment)) {
    instances.put(previousAlignment, new ArrayList<Instance>());
    }
    instances.get(previousAlignment).add(wordTrainingVector);
    if (!realization.get(w).getWord().equals(Bagel.TOKEN_START)
    && !realization.get(w).getWord().equals(Bagel.TOKEN_END)) {
    subPhrase.add(realization.get(w).getWord());
    }
    }
    if (!isValueMentioned) {
    if (realization.get(w).getWord().startsWith(Bagel.TOKEN_X)
    && (valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+"))) {
    isValueMentioned = true;
    } else if (!realization.get(w).getWord().startsWith(Bagel.TOKEN_X)
    && !(valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+"))) {
    for (ArrayList<String> alignedStr : valueAlignments.get(valueTBM).keySet()) {
    if (endsWith(subPhrase, alignedStr)) {
    isValueMentioned = true;
    break;
    }
    }
    }
    }
    }
    }
    }
    predicateWordTrainingData.get(predicate).put(di, instances);
    }
    }
    }
    return predicateWordTrainingData;
    }
    
    public HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> createAllWithAllWordTrainingDatasets() {
    HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData = new HashMap<>();
    
    if (!availableWordActions.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*//*) {
    for (String predicate : predicates) {
    predicateWordTrainingData.put(predicate, new HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>());
    
    ArrayList<String> attrs = new ArrayList<>();
    for (DatasetInstance di : abstractDatasetInstances.get(predicate)) {
    HashMap<String, ArrayList<Instance>> instances = new HashMap<>();
    for (ArrayList<Action> real : di.getEvalRealizations()) {
    ArrayList<String> mentionedAttributes = di.getEvalMentionedAttributeSequences().get(real);
    for (Action act : real) {
    if (mentionedAttributes.isEmpty()) {
    mentionedAttributes.add(act.getAttribute());
    } else if (!mentionedAttributes.get(mentionedAttributes.size() - 1).equals(act.getAttribute())) {
    mentionedAttributes.add(act.getAttribute());
    }
    }
    ArrayList<Action> realization = new ArrayList<Action>();
    for (String at : mentionedAttributes) {
    if (!at.equals(Bagel.TOKEN_PUNCT)
    && !at.equals(Bagel.TOKEN_START)
    && !at.equals(Bagel.TOKEN_END)
    && !at.equals("[]")) {
    for (Action a : real) {
    if (!a.getWord().equals(Bagel.TOKEN_START)
    && !a.getWord().equals(Bagel.TOKEN_END)) {
    realization.add(new Action(a.getWord(), at));
    }
    }
    realization.add(new Action(Bagel.TOKEN_END, at));
    }
    }
    
    HashMap<String, HashSet<String>> values = new HashMap();
    for (String attr : di.getMeaningRepresentation().getAttributes().keySet()) {
    values.put(attr, new HashSet(di.getMeaningRepresentation().getAttributes().get(attr)));
    }
    boolean isValueMentioned = false;
    ArrayList<String> valuesTF = null;
    String valueTBM = "";
    String previousAlignment = "";
    ArrayList<String> subPhrase = new ArrayList<>();
    for (int w = 0; w < realization.size(); w++) {
    ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
    for (int i = 0; i < attrs.size() - 1; i++) {
    predictedAttributesForInstance.add(attrs.get(i));
    }
    if (!attrs.get(attrs.size() - 1).equals(previousAlignment)) {
    predictedAttributesForInstance.add(attrs.get(attrs.size() - 1));
    }
    if (!realization.get(w).getAttribute().equals(Bagel.TOKEN_PUNCT)) {
    if (!realization.get(w).getAttribute().equals(previousAlignment)) {
    attrs.add(realization.get(w).getAttribute());
    
    previousAlignment = realization.get(w).getAttribute();
    subPhrase = new ArrayList<>();
    isValueMentioned = false;
    valuesTF = new ArrayList<>();
    valueTBM = "";
    if (previousAlignment.contains("=")) {
    valueTBM = previousAlignment.substring(previousAlignment.indexOf("=") + 1);
    } else {
    isValueMentioned = true;
    }
    }
    Instance wordTrainingVector = createWordInstance(predicate, predictedAttributesForInstance, realization, w, valueTBM, isValueMentioned, valuesTF, di.getMeaningRepresentation(), false);
    
    if (wordTrainingVector != null) {
    if (!instances.containsKey(previousAlignment)) {
    instances.put(previousAlignment, new ArrayList<Instance>());
    }
    instances.get(previousAlignment).add(wordTrainingVector);
    if (!realization.get(w).getWord().equals(Bagel.TOKEN_START)
    && !realization.get(w).getWord().equals(Bagel.TOKEN_END)) {
    subPhrase.add(realization.get(w).getWord());
    }
    }
    if (!isValueMentioned) {
    if (realization.get(w).getWord().startsWith(Bagel.TOKEN_X)
    && (valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+"))) {
    isValueMentioned = true;
    } else if (!realization.get(w).getWord().startsWith(Bagel.TOKEN_X)
    && !(valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+"))) {
    for (ArrayList<String> alignedStr : valueAlignments.get(valueTBM).keySet()) {
    if (endsWith(subPhrase, alignedStr)) {
    isValueMentioned = true;
    break;
    }
    }
    }
    }
    }
    }
    }
    predicateWordTrainingData.get(predicate).put(di, instances);
    }
    }
    }
    return predicateWordTrainingData;
    }*/


    public HashMap<Integer, HashSet<String>> createRandomAlignments(ArrayList<DatasetInstance> trainingData) {
        punctPatterns = new HashMap<>();
        HashMap<Integer, HashSet<String>> nGrams = new HashMap<>();
        for (DatasetInstance di : trainingData) {
            HashSet<ArrayList<Action>> randomRealizations = new HashSet<>();
            for (ArrayList<Action> realization : di.getEvalRealizations()) {
                HashMap<String, HashSet<String>> values = new HashMap();
                for (String attr : di.getMeaningRepresentation().getAttributes().keySet()) {
                    values.put(attr, new HashSet(di.getMeaningRepresentation().getAttributes().get(attr)));
                }

                ArrayList<Action> randomRealization = new ArrayList<Action>();
                for (Action a : realization) {
                    if (a.getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                        randomRealization.add(new Action(a.getWord(), a.getAttribute()));
                    } else {
                        randomRealization.add(new Action(a.getWord(), ""));
                    }
                }

                HashMap<Double, HashMap<String, ArrayList<Integer>>> indexAlignments = new HashMap<>();
                for (String attr : values.keySet()) {
                    if (!attr.equals("type")) {
                        for (String value : values.get(attr)) {
                            if (!(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+"))) {
                                for (ArrayList<String> align : valueAlignments.get(value).keySet()) {
                                    int n = align.size();
                                    for (int i = 0; i <= randomRealization.size() - n; i++) {
                                        ArrayList<String> compare = new ArrayList<String>();
                                        ArrayList<Integer> indexAlignment = new ArrayList<Integer>();
                                        for (int j = 0; j < n; j++) {
                                            compare.add(randomRealization.get(i + j).getWord());
                                            indexAlignment.add(i + j);
                                        }
                                        if (compare.equals(align)) {
                                            if (!indexAlignments.containsKey(valueAlignments.get(value).get(align))) {
                                                indexAlignments.put(valueAlignments.get(value).get(align), new HashMap());
                                            }
                                            indexAlignments.get(valueAlignments.get(value).get(align)).put(attr + "=" + value, indexAlignment);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                ArrayList<Double> similarities = new ArrayList<>(indexAlignments.keySet());
                Collections.sort(similarities);
                HashSet<String> assignedAttrValues = new HashSet<String>();
                HashSet<Integer> assignedIntegers = new HashSet<Integer>();
                for (int i = similarities.size() - 1; i >= 0; i--) {
                    for (String attrValue : indexAlignments.get(similarities.get(i)).keySet()) {
                        if (!assignedAttrValues.contains(attrValue)) {
                            boolean isUnassigned = true;
                            for (Integer index : indexAlignments.get(similarities.get(i)).get(attrValue)) {
                                if (assignedIntegers.contains(index)) {
                                    isUnassigned = false;
                                }
                            }
                            if (isUnassigned) {
                                assignedAttrValues.add(attrValue);
                                for (Integer index : indexAlignments.get(similarities.get(i)).get(attrValue)) {
                                    assignedIntegers.add(index);
                                    randomRealization.get(index).setAttribute(attrValue.toLowerCase().trim());
                                }
                            }
                        }
                    }
                }

                HashMap<String, Integer> attrXIndeces = new HashMap<>();
                for (Action a : randomRealization) {
                    if (a.getWord().startsWith(Bagel.TOKEN_X)) {
                        String attr = a.getWord().substring(3, a.getWord().lastIndexOf('_')).toLowerCase().trim();
                        int index = 0;
                        if (!attrXIndeces.containsKey(attr)) {
                            attrXIndeces.put(attr, 1);
                        } else {
                            index = attrXIndeces.get(attr);
                            attrXIndeces.put(attr, index + 1);
                        }
                        a.setAttribute(attr + "=x" + index);
                    }
                }

                String previousAttr = "";
                for (int i = 0; i < randomRealization.size(); i++) {
                    if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                        }
                    } else if (!randomRealization.get(i).getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                    } else {
                        previousAttr = "";
                    }
                }

                previousAttr = "";
                for (int i = randomRealization.size() - 1; i >= 0; i--) {
                    if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                        }
                    } else if (!randomRealization.get(i).getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                    } else {
                        previousAttr = "";
                    }
                }

                previousAttr = "";
                for (int i = 0; i < randomRealization.size(); i++) {
                    if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                        }
                    } else {
                        previousAttr = randomRealization.get(i).getAttribute();
                    }
                }

                previousAttr = "";
                for (int i = randomRealization.size() - 1; i >= 0; i--) {
                    if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                        }
                    } else {
                        previousAttr = randomRealization.get(i).getAttribute();
                    }
                }

                //FIX WRONG @PUNCT@                
                previousAttr = "";
                for (int i = randomRealization.size() - 1; i >= 0; i--) {
                    if (randomRealization.get(i).getAttribute().equals(TOKEN_PUNCT) && !randomRealization.get(i).getWord().matches("[,.?!;:']")) {
                        if (!previousAttr.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                        }
                    } else if (!randomRealization.get(i).getAttribute().equals(TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                    }
                }
                ArrayList<Action> cleanRandomRealization = new ArrayList<>();
                for (Action a : randomRealization) {
                    if (!a.getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                        cleanRandomRealization.add(a);
                    }
                }
                //ADD END TOKENS
                ArrayList<Action> endRandomRealization = new ArrayList<>();
                previousAttr = "";
                for (int i = 0; i < cleanRandomRealization.size(); i++) {
                    Action a = cleanRandomRealization.get(i);
                    if (!previousAttr.isEmpty()) {
                        if (!a.getAttribute().equals(previousAttr)) {
                            endRandomRealization.add(new Action(Bagel.TOKEN_END, previousAttr));
                        }
                    }
                    endRandomRealization.add(a);
                    previousAttr = a.getAttribute();
                }
                endRandomRealization.add(new Action(Bagel.TOKEN_END, previousAttr));
                endRandomRealization.add(new Action(Bagel.TOKEN_END, Bagel.TOKEN_END));
                randomRealizations.add(endRandomRealization);

                ArrayList<String> attrValues = new ArrayList<String>();
                for (Action a : endRandomRealization) {
                    if (attrValues.isEmpty()) {
                        attrValues.add(a.getAttribute());
                    } else if (!attrValues.get(attrValues.size() - 1).equals(a.getAttribute())) {
                        attrValues.add(a.getAttribute());
                    }
                }
                if (attrValues.size() > maxAttrRealizationSize) {
                    maxAttrRealizationSize = attrValues.size();
                }

                for (int i = 0; i < randomRealization.size(); i++) {
                    Action a = randomRealization.get(i);
                    if (a.getAttribute().equals(Bagel.TOKEN_PUNCT)
                            && !a.getWord().equals(".")) {
                        boolean legal = true;
                        ArrayList<Action> surroundingActions = new ArrayList<>();
                        if (i - 2 >= 0) {
                            surroundingActions.add(randomRealization.get(i - 2));
                        } else {
                            surroundingActions.add(null);
                        }
                        if (i - 1 >= 0) {
                            surroundingActions.add(randomRealization.get(i - 1));
                        } else {
                            legal = false;
                        }
                        if (i + 1 < randomRealization.size()) {
                            surroundingActions.add(randomRealization.get(i + 1));
                        } else {
                            legal = false;
                        }
                        if (i + 2 < randomRealization.size()) {
                            surroundingActions.add(randomRealization.get(i + 2));
                        } else {
                            surroundingActions.add(null);
                        }
                        if (legal) {
                            punctPatterns.put(surroundingActions, a);
                        }
                    }
                }
                /*for (int i = 2; i <= 6; i++) {
                for (int j = 0; j < cleanRandomRealization.size() - i; j++) {
                String ngram = "";
                for (int r = 0; r < i; r++) {
                ngram += cleanRandomRealization.get(j + r).getWord() + "|";
                }
                ngram = ngram.substring(0, ngram.length() - 1);
                if (!nGrams.containsKey(i)) {
                nGrams.put(i, new HashSet<String>());
                }
                nGrams.get(i).add(ngram);
                }
                }*/
            }
            for (ArrayList<Action> rr : randomRealizations) {
                for (Action key : rr) {
                    if (key.getWord().trim().isEmpty()) {
                        System.out.println("RR " + rr);
                        System.out.println("RR " + key);
                        System.exit(0);
                    }
                    if (key.getAttribute().equals("[]")) {
                        System.out.println("RR " + rr);
                        System.out.println("RR " + key);
                        System.exit(0);
                    }
                }
            }
            di.setRealizations(randomRealizations);
        }
        return nGrams;
    }

    public Instance createAttrInstance(String predicate, String bestAction, ArrayList<String> previousGeneratedAttrs, ArrayList<Action> previousGeneratedWords, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

        HashSet<String> attrsToBeMentioned = new HashSet<>();
        for (String attrValue : attrValuesToBeMentioned) {
            String attr = attrValue;
            if (attr.contains("=")) {
                attr = attrValue.substring(0, attrValue.indexOf('='));
            }
            attrsToBeMentioned.add(attr);
        }
        if (!bestAction.isEmpty()) {
            //COSTS
            if (bestAction.equals(Bagel.TOKEN_END)) {
                costs.put(Bagel.TOKEN_END, 0.0);
                for (String action : attributes.get(predicate)) {
                    costs.put(action, 1.0);
                }
            } else if (!bestAction.equals("@TOK@")) {
                costs.put(Bagel.TOKEN_END, 1.0);
                for (String action : attributes.get(predicate)) {
                    String attr = bestAction;
                    if (bestAction.contains("=")) {
                        attr = bestAction.substring(0, bestAction.indexOf('=')).toLowerCase().trim();
                    }
                    if (action.equals(attr)) {
                        costs.put(action, 0.0);
                    } else {
                        costs.put(action, 1.0);
                    }
                }
            }
        }
        return Bagel.this.createAttrInstance(predicate, previousGeneratedAttrs, previousGeneratedWords, costs, attrValuesAlreadyMentioned, attrValuesToBeMentioned, MR);
    }

    public Instance createAttrInstance(String predicate, ArrayList<String> previousGeneratedAttrs, ArrayList<Action> previousGeneratedWords, TObjectDoubleHashMap<String> costs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        for (String action : attributes.get(predicate)) {
            valueSpecificFeatures.put(action, new TObjectDoubleHashMap<String>());
        }

        ArrayList<String> mentionedAttrValues = new ArrayList<>();
        for (String attrValue : previousGeneratedAttrs) {
            if (!attrValue.equals(Bagel.TOKEN_END)
                    && !attrValue.equals(Bagel.TOKEN_START)) {
                mentionedAttrValues.add(attrValue);
            }
        }

        for (int j = 1; j <= 1; j++) {
            String previousAttrValue = "@@";
            if (mentionedAttrValues.size() - j >= 0) {
                previousAttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - j).trim();
            }
            generalFeatures.put("feature_attrValue_" + j + "_" + previousAttrValue, 1.0);
        }
        //Word N-Grams            
        String prevAttrValue = "@@";
        if (mentionedAttrValues.size() - 1 >= 0) {
            prevAttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 1).trim();
        }
        String prev2AttrValue = "@@";
        if (mentionedAttrValues.size() - 2 >= 0) {
            prev2AttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 2).trim();
        }
        String prev3AttrValue = "@@";
        if (mentionedAttrValues.size() - 3 >= 0) {
            prev3AttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 3).trim();
        }
        String prev4AttrValue = "@@";
        if (mentionedAttrValues.size() - 4 >= 0) {
            prev4AttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 4).trim();
        }
        String prev5AttrValue = "@@";
        if (mentionedAttrValues.size() - 5 >= 0) {
            prev5AttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 5).trim();
        }

        String prevBigramAttrValue = prev2AttrValue + "|" + prevAttrValue;
        String prevTrigramAttrValue = prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        String prev4gramAttrValue = prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        String prev5gramAttrValue = prev5AttrValue + "|" + prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        generalFeatures.put("feature_attrValue_bigram_" + prevBigramAttrValue, 1.0);
        generalFeatures.put("feature_attrValue_trigram_" + prevTrigramAttrValue, 1.0);
        generalFeatures.put("feature_attrValue_4gram_" + prev4gramAttrValue, 1.0);
        generalFeatures.put("feature_attrValue_5gram_" + prev5gramAttrValue, 1.0);

        /*String bigramAttrValue54 = prev5AttrValue + "|" + prev4AttrValue;
        String bigramAttrValue43 = prev4AttrValue + "|" + prev3AttrValue;
        String bigramAttrValue32 = prev3AttrValue + "|" + prev2AttrValue;
        generalFeatures.put("feature_attrValue_bigramAttrValue54_" + bigramAttrValue54, 1.0);
        generalFeatures.put("feature_attrValue_bigramAttrValue43_" + bigramAttrValue43, 1.0);
        generalFeatures.put("feature_attrValue_bigramAttrValue32_" + bigramAttrValue32, 1.0);
        
        String bigramAttrValueSkip53 = prev5AttrValue + "|" + prev3AttrValue;
        String bigramAttrValueSkip42 = prev4AttrValue + "|" + prev2AttrValue;
        String bigramAttrValueSkip31 = prev3AttrValue + "|" + prevAttrValue;
        generalFeatures.put("feature_attrValue_bigramAttrValueSkip53_" + bigramAttrValueSkip53, 1.0);
        generalFeatures.put("feature_attrValue_bigramAttrValueSkip42_" + bigramAttrValueSkip42, 1.0);
        generalFeatures.put("feature_attrValue_bigramAttrValueSkip31_" + bigramAttrValueSkip31, 1.0);
        
        String trigramAttrValue543 = prev5AttrValue + "|" + prev4AttrValue + "|" + prev3AttrValue;
        String trigramAttrValue432 = prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue;
        generalFeatures.put("feature_attrValue_trigramAttrValue543_" + trigramAttrValue543, 1.0);
        generalFeatures.put("feature_attrValue_trigramAttrValue432_" + trigramAttrValue432, 1.0);
        
        String trigramAttrValueSkip542 = prev5AttrValue + "|" + prev4AttrValue + "|" + prev2AttrValue;
        String trigramAttrValueSkip532 = prev5AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue;
        String trigramAttrValueSkip431 = prev4AttrValue + "|" + prev3AttrValue + "|" + prevAttrValue;
        String trigramAttrValueSkip421 = prev4AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        generalFeatures.put("feature_attrValue_trigramAttrValueSkip542_" + trigramAttrValueSkip542, 1.0);
        generalFeatures.put("feature_attrValue_trigramAttrValueSkip532_" + trigramAttrValueSkip532, 1.0);
        generalFeatures.put("feature_attrValue_trigramAttrValueSkip431_" + trigramAttrValueSkip431, 1.0);
        generalFeatures.put("feature_attrValue_trigramAttrValueSkip421_" + trigramAttrValueSkip421, 1.0);*/
        //Word Positions
        //features.put("feature_pos:", w);
        //If arguments have been generated or not
        for (int i = 0; i < mentionedAttrValues.size(); i++) {
            generalFeatures.put("feature_attrValue_allreadyMentioned_" + mentionedAttrValues.get(i), 1.0);
        }
        //If arguments should still be generated or not
        for (String attrValue : attrValuesToBeMentioned) {
            generalFeatures.put("feature_attrValue_toBeMentioned_" + attrValue, 1.0);
        }
        //Which attrs are in the MR and which are not
        for (String attribute : attributes.get(predicate)) {
            if (MR.getAttributes().keySet().contains(attribute)) {
                generalFeatures.put("feature_attr_inMR_" + attribute, 1.0);
            } else {
                generalFeatures.put("feature_attr_notInMR_" + attribute, 1.0);
            }
        }

        ArrayList<String> mentionedAttrs = new ArrayList<>();
        for (int i = 0; i < mentionedAttrValues.size(); i++) {
            String attr = mentionedAttrValues.get(i);
            if (attr.contains("=")) {
                attr = mentionedAttrValues.get(i).substring(0, mentionedAttrValues.get(i).indexOf('='));
            }
            //if (mentionedAttrs.isEmpty()) {
            //mentionedAttrs.add(attr);
            //} else if (!mentionedAttrs.get(mentionedAttrs.size() - 1).equals(attr)) {
            mentionedAttrs.add(attr);
            //}
        }
        HashSet<String> attrsToBeMentioned = new HashSet<>();
        for (String attrValue : attrValuesToBeMentioned) {
            String attr = attrValue;
            if (attr.contains("=")) {
                attr = attrValue.substring(0, attrValue.indexOf('='));
            }
            attrsToBeMentioned.add(attr);
        }

        for (int j = 1; j <= 1; j++) {
            String previousAttr = "";
            if (mentionedAttrs.size() - j >= 0) {
                previousAttr = mentionedAttrs.get(mentionedAttrs.size() - j).trim();
            }
            if (!previousAttr.isEmpty()) {
                generalFeatures.put("feature_attr_" + j + "_" + previousAttr, 1.0);
            } else {
                generalFeatures.put("feature_attr_" + j + "_@@", 1.0);
            }
        }
        //Word N-Grams            
        String prevAttr = "@@";
        if (mentionedAttrs.size() - 1 >= 0) {
            prevAttr = mentionedAttrs.get(mentionedAttrs.size() - 1).trim();
        }
        String prev2Attr = "@@";
        if (mentionedAttrs.size() - 2 >= 0) {
            prev2Attr = mentionedAttrs.get(mentionedAttrs.size() - 2).trim();
        }
        String prev3Attr = "@@";
        if (mentionedAttrs.size() - 3 >= 0) {
            prev3Attr = mentionedAttrs.get(mentionedAttrs.size() - 3).trim();
        }
        String prev4Attr = "@@";
        if (mentionedAttrs.size() - 4 >= 0) {
            prev4Attr = mentionedAttrs.get(mentionedAttrs.size() - 4).trim();
        }
        String prev5Attr = "@@";
        if (mentionedAttrs.size() - 5 >= 0) {
            prev5Attr = mentionedAttrs.get(mentionedAttrs.size() - 5).trim();
        }

        String prevBigramAttr = prev2Attr + "|" + prevAttr;
        String prevTrigramAttr = prev3Attr + "|" + prev2Attr + "|" + prevAttr;
        String prev4gramAttr = prev4Attr + "|" + prev3Attr + "|" + prev2Attr + "|" + prevAttr;
        String prev5gramAttr = prev5Attr + "|" + prev4Attr + "|" + prev3Attr + "|" + prev2Attr + "|" + prevAttr;

        generalFeatures.put("feature_attr_bigram_" + prevBigramAttr, 1.0);
        generalFeatures.put("feature_attr_trigram_" + prevTrigramAttr, 1.0);
        generalFeatures.put("feature_attr_4gram_" + prev4gramAttr, 1.0);
        generalFeatures.put("feature_attr_5gram_" + prev5gramAttr, 1.0);

        /*String bigramAttr54 = prev5Attr + "|" + prev4Attr;
        String bigramAttr43 = prev4Attr + "|" + prev3Attr;
        String bigramAttr32 = prev3Attr + "|" + prev2Attr;
        generalFeatures.put("feature_attr_bigramAttr54_" + bigramAttr54, 1.0);
        generalFeatures.put("feature_attr_bigramAttr43_" + bigramAttr43, 1.0);
        generalFeatures.put("feature_attr_bigramAttr32_" + bigramAttr32, 1.0);
        
        String bigramAttrSkip53 = prev5Attr + "|" + prev3Attr;
        String bigramAttrSkip42 = prev4Attr + "|" + prev2Attr;
        String bigramAttrSkip31 = prev3Attr + "|" + prevAttr;
        generalFeatures.put("feature_attr_bigramAttrSkip53_" + bigramAttrSkip53, 1.0);
        generalFeatures.put("feature_attr_bigramAttrSkip42_" + bigramAttrSkip42, 1.0);
        generalFeatures.put("feature_attr_bigramAttrSkip31_" + bigramAttrSkip31, 1.0);
        
        String trigramAttr543 = prev5Attr + "|" + prev4Attr + "|" + prev3Attr;
        String trigramAttr432 = prev4Attr + "|" + prev3Attr + "|" + prev2Attr;
        generalFeatures.put("feature_attr_trigramAttr543_" + trigramAttr543, 1.0);
        generalFeatures.put("feature_attr_trigramAttr432_" + trigramAttr432, 1.0);
        
        String trigramAttrSkip542 = prev5Attr + "|" + prev4Attr + "|" + prev2Attr;
        String trigramAttrSkip532 = prev5Attr + "|" + prev3Attr + "|" + prev2Attr;
        String trigramAttrSkip431 = prev4Attr + "|" + prev3Attr + "|" + prevAttr;
        String trigramAttrSkip421 = prev4Attr + "|" + prev2Attr + "|" + prevAttr;
        generalFeatures.put("feature_attr_trigramAttrSkip542_" + trigramAttrSkip542, 1.0);
        generalFeatures.put("feature_attr_trigramAttrSkip532_" + trigramAttrSkip532, 1.0);
        generalFeatures.put("feature_attr_trigramAttrSkip431_" + trigramAttrSkip431, 1.0);
        generalFeatures.put("feature_attr_trigramAttrSkip421_" + trigramAttrSkip421, 1.0);*/
        ArrayList<Action> generatedWords = new ArrayList<>();
        ArrayList<Action> generatedWordsInPreviousAttrValue = new ArrayList<>();
        if (!mentionedAttrValues.isEmpty()) {
            String previousAttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 1);
            for (int i = 0; i < previousGeneratedWords.size(); i++) {
                Action a = previousGeneratedWords.get(i);
                if (!a.getWord().equals(Bagel.TOKEN_START)
                        && !a.getWord().equals(Bagel.TOKEN_END)) {
                    generatedWords.add(a);
                    if (a.getAttribute().equals(previousAttrValue)) {
                        generatedWordsInPreviousAttrValue.add(a);
                    }
                }
            }
        }
        //Previous word features
        for (int j = 1; j <= 1; j++) {
            String previousWord = "@@";
            if (generatedWords.size() - j >= 0) {
                previousWord = generatedWords.get(generatedWords.size() - j).getWord().trim();
            }
            generalFeatures.put("feature_word_" + j + "_" + previousWord.toLowerCase(), 1.0);
        }
        String prevWord = "@@";
        if (generatedWords.size() - 1 >= 0) {
            prevWord = generatedWords.get(generatedWords.size() - 1).getWord().trim();
        }
        String prev2Word = "@@";
        if (generatedWords.size() - 2 >= 0) {
            prev2Word = generatedWords.get(generatedWords.size() - 2).getWord().trim();
        }
        String prev3Word = "@@";
        if (generatedWords.size() - 3 >= 0) {
            prev3Word = generatedWords.get(generatedWords.size() - 3).getWord().trim();
        }
        String prev4Word = "@@";
        if (generatedWords.size() - 4 >= 0) {
            prev4Word = generatedWords.get(generatedWords.size() - 4).getWord().trim();
        }
        String prev5Word = "@@";
        if (generatedWords.size() - 5 >= 0) {
            prev5Word = generatedWords.get(generatedWords.size() - 5).getWord().trim();
        }

        String prevBigram = prev2Word + "|" + prevWord;
        String prevTrigram = prev3Word + "|" + prev2Word + "|" + prevWord;
        String prev4gram = prev4Word + "|" + prev3Word + "|" + prev2Word + "|" + prevWord;
        String prev5gram = prev5Word + "|" + prev4Word + "|" + prev3Word + "|" + prev2Word + "|" + prevWord;

        generalFeatures.put("feature_word_bigram_" + prevBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_word_trigram_" + prevTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_word_4gram_" + prev4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_word_5gram_" + prev5gram.toLowerCase(), 1.0);

        /*String bigramWord54 = prev5Word + "|" + prev4Word;
        String bigramWord43 = prev4Word + "|" + prev3Word;
        String bigramWord32 = prev3Word + "|" + prev2Word;
        generalFeatures.put("feature_word_bigramWord54_" + bigramWord54, 1.0);
        generalFeatures.put("feature_word_bigramWord43_" + bigramWord43, 1.0);
        generalFeatures.put("feature_word_bigramWord32_" + bigramWord32, 1.0);
        
        String bigramWordSkip53 = prev5Word + "|" + prev3Word;
        String bigramWordSkip42 = prev4Word + "|" + prev2Word;
        String bigramWordSkip31 = prev3Word + "|" + prevWord;
        generalFeatures.put("feature_word_bigramWordSkip53_" + bigramWordSkip53, 1.0);
        generalFeatures.put("feature_word_bigramWordSkip42_" + bigramWordSkip42, 1.0);
        generalFeatures.put("feature_word_bigramWordSkip31_" + bigramWordSkip31, 1.0);
        
        String trigramWord543 = prev5Word + "|" + prev4Word + "|" + prev3Word;
        String trigramWord432 = prev4Word + "|" + prev3Word + "|" + prev2Word;
        generalFeatures.put("feature_word_trigramWord543_" + trigramWord543, 1.0);
        generalFeatures.put("feature_word_trigramWord432_" + trigramWord432, 1.0);
        
        String trigramWordSkip542 = prev5Word + "|" + prev4Word + "|" + prev2Word;
        String trigramWordSkip532 = prev5Word + "|" + prev3Word + "|" + prev2Word;
        String trigramWordSkip431 = prev4Word + "|" + prev3Word + "|" + prevWord;
        String trigramWordSkip421 = prev4Word + "|" + prev2Word + "|" + prevWord;
        generalFeatures.put("feature_word_trigramWordSkip542_" + trigramWordSkip542, 1.0);
        generalFeatures.put("feature_word_trigramWordSkip532_" + trigramWordSkip532, 1.0);
        generalFeatures.put("feature_word_trigramWordSkip431_" + trigramWordSkip431, 1.0);
        generalFeatures.put("feature_word_trigramWordSkip421_" + trigramWordSkip421, 1.0);*/
        //Previous word in same as current attrValue features
        for (int j = 1; j <= 1; j++) {
            String previousCurrentAttrValueWord = "@@";
            if (generatedWordsInPreviousAttrValue.size() - j >= 0) {
                previousCurrentAttrValueWord = generatedWordsInPreviousAttrValue.get(generatedWordsInPreviousAttrValue.size() - j).getWord().trim();
            }
            generalFeatures.put("feature_currentAttrValueWord_" + j + "_" + previousCurrentAttrValueWord.toLowerCase(), 1.0);
        }
        String prevCurrentAttrValueWord = "@@";
        if (generatedWordsInPreviousAttrValue.size() - 1 >= 0) {
            prevCurrentAttrValueWord = generatedWordsInPreviousAttrValue.get(generatedWordsInPreviousAttrValue.size() - 1).getWord().trim();
        }
        String prev2CurrentAttrValueWord = "@@";
        if (generatedWordsInPreviousAttrValue.size() - 2 >= 0) {
            prev2CurrentAttrValueWord = generatedWordsInPreviousAttrValue.get(generatedWordsInPreviousAttrValue.size() - 2).getWord().trim();
        }
        String prev3CurrentAttrValueWord = "@@";
        if (generatedWordsInPreviousAttrValue.size() - 3 >= 0) {
            prev3CurrentAttrValueWord = generatedWordsInPreviousAttrValue.get(generatedWordsInPreviousAttrValue.size() - 3).getWord().trim();
        }
        String prev4CurrentAttrValueWord = "@@";
        if (generatedWordsInPreviousAttrValue.size() - 4 >= 0) {
            prev4CurrentAttrValueWord = generatedWordsInPreviousAttrValue.get(generatedWordsInPreviousAttrValue.size() - 4).getWord().trim();
        }
        String prev5CurrentAttrValueWord = "@@";
        if (generatedWordsInPreviousAttrValue.size() - 5 >= 0) {
            prev5CurrentAttrValueWord = generatedWordsInPreviousAttrValue.get(generatedWordsInPreviousAttrValue.size() - 5).getWord().trim();
        }

        String prevCurrentAttrValueBigram = prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        String prevCurrentAttrValueTrigram = prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        String prevCurrentAttrValue4gram = prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        String prevCurrentAttrValue5gram = prev5CurrentAttrValueWord + "|" + prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;

        generalFeatures.put("feature_previousAttrValueWord_bigram_" + prevCurrentAttrValueBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_previousAttrValueWord_trigram_" + prevCurrentAttrValueTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_previousAttrValueWord_4gram_" + prevCurrentAttrValue4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_previousAttrValueWord_5gram_" + prevCurrentAttrValue5gram.toLowerCase(), 1.0);

        /*String bigramCurrentAttrValueWord54 = prev5CurrentAttrValueWord + "|" + prev4CurrentAttrValueWord;
        String bigramCurrentAttrValueWord43 = prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord;
        String bigramCurrentAttrValueWord32 = prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        generalFeatures.put("feature_previousAttrValueWord_bigramCurrentAttrValueWord54_" + bigramCurrentAttrValueWord54, 1.0);
        generalFeatures.put("feature_previousAttrValueWord_bigramCurrentAttrValueWord43_" + bigramCurrentAttrValueWord43, 1.0);
        generalFeatures.put("feature_previousAttrValueWord_bigramCurrentAttrValueWord32_" + bigramCurrentAttrValueWord32, 1.0);
        
        String bigramCurrentAttrValueWordSkip53 = prev5CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord;
        String bigramCurrentAttrValueWordSkip42 = prev4CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        String bigramCurrentAttrValueWordSkip31 = prev3CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        generalFeatures.put("feature_previousAttrValueWord_bigramCurrentAttrValueWordSkip53_" + bigramCurrentAttrValueWordSkip53, 1.0);
        generalFeatures.put("feature_previousAttrValueWord_bigramCurrentAttrValueWordSkip42_" + bigramCurrentAttrValueWordSkip42, 1.0);
        generalFeatures.put("feature_previousAttrValueWord_bigramCurrentAttrValueWordSkip31_" + bigramCurrentAttrValueWordSkip31, 1.0);
        
        String trigramCurrentAttrValueWord543 = prev5CurrentAttrValueWord + "|" + prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord;
        String trigramCurrentAttrValueWord432 = prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        generalFeatures.put("feature_previousAttrValueWord_trigramCurrentAttrValueWord543_" + trigramCurrentAttrValueWord543, 1.0);
        generalFeatures.put("feature_previousAttrValueWord_trigramCurrentAttrValueWord432_" + trigramCurrentAttrValueWord432, 1.0);
        
        String trigramCurrentAttrValueWordSkip542 = prev5CurrentAttrValueWord + "|" + prev4CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        String trigramCurrentAttrValueWordSkip532 = prev5CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        String trigramCurrentAttrValueWordSkip431 = prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        String trigramCurrentAttrValueWordSkip421 = prev4CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        generalFeatures.put("feature_previousAttrValueWord_trigramCurrentAttrValueWordSkip542_" + trigramCurrentAttrValueWordSkip542, 1.0);
        generalFeatures.put("feature_previousAttrValueWord_trigramCurrentAttrValueWordSkip532_" + trigramCurrentAttrValueWordSkip532, 1.0);
        generalFeatures.put("feature_previousAttrValueWord_trigramCurrentAttrValueWordSkip431_" + trigramCurrentAttrValueWordSkip431, 1.0);
        generalFeatures.put("feature_previousAttrValueWord_trigramCurrentAttrValueWordSkip421_" + trigramCurrentAttrValueWordSkip421, 1.0);*/
        //Previous Attr|Word features
        for (int j = 1; j <= 1; j++) {
            String previousAttrWord = "@@";
            if (generatedWords.size() - j >= 0) {
                if (generatedWords.get(generatedWords.size() - j).getAttribute().contains("=")) {
                    previousAttrWord = generatedWords.get(generatedWords.size() - j).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - j).getAttribute().indexOf('=')) + "|" + generatedWords.get(generatedWords.size() - j).getWord().trim();
                } else {
                    previousAttrWord = generatedWords.get(generatedWords.size() - j).getAttribute().trim() + "|" + generatedWords.get(generatedWords.size() - j).getWord().trim();
                }
            }
            generalFeatures.put("feature_attrWord_" + j + "_" + previousAttrWord.toLowerCase(), 1.0);
        }
        String prevAttrWord = "@@";
        if (generatedWords.size() - 1 >= 0) {
            if (generatedWords.get(generatedWords.size() - 1).getAttribute().contains("=")) {
                prevAttrWord = generatedWords.get(generatedWords.size() - 1).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 1).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 1).getWord().trim();
            } else {
                prevAttrWord = generatedWords.get(generatedWords.size() - 1).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 1).getWord().trim();

            }
        }
        String prev2AttrWord = "@@";
        if (generatedWords.size() - 2 >= 0) {
            if (generatedWords.get(generatedWords.size() - 2).getAttribute().contains("=")) {
                prev2AttrWord = generatedWords.get(generatedWords.size() - 2).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 2).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 2).getWord().trim();
            } else {
                prev2AttrWord = generatedWords.get(generatedWords.size() - 2).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 2).getWord().trim();
            }
        }
        String prev3AttrWord = "@@";
        if (generatedWords.size() - 3 >= 0) {
            if (generatedWords.get(generatedWords.size() - 3).getAttribute().contains("=")) {
                prev3AttrWord = generatedWords.get(generatedWords.size() - 3).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 3).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 3).getWord().trim();
            } else {
                prev3AttrWord = generatedWords.get(generatedWords.size() - 3).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 3).getWord().trim();
            }
        }
        String prev4AttrWord = "@@";
        if (generatedWords.size() - 4 >= 0) {
            if (generatedWords.get(generatedWords.size() - 4).getAttribute().contains("=")) {
                prev4AttrWord = generatedWords.get(generatedWords.size() - 4).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 4).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 4).getWord().trim();
            } else {
                prev4AttrWord = generatedWords.get(generatedWords.size() - 4).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 4).getWord().trim();
            }
        }
        String prev5AttrWord = "@@";
        if (generatedWords.size() - 5 >= 0) {
            if (generatedWords.get(generatedWords.size() - 5).getAttribute().contains("=")) {
                prev5AttrWord = generatedWords.get(generatedWords.size() - 5).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 5).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 5).getWord().trim();
            } else {
                prev5AttrWord = generatedWords.get(generatedWords.size() - 5).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 5).getWord().trim();
            }
        }

        String prevAttrWordBigram = prev2AttrWord + "|" + prevAttrWord;
        String prevAttrWordTrigram = prev3AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;
        String prevAttrWord4gram = prev4AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;
        String prevAttrWord5gram = prev5AttrWord + "|" + prev4AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;

        generalFeatures.put("feature_attrWord_bigram_" + prevAttrWordBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrWord_trigram_" + prevAttrWordTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrWord_4gram_" + prevAttrWord4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrWord_5gram_" + prevAttrWord5gram.toLowerCase(), 1.0);

        /*String bigramAttrWord54 = prev5AttrWord + "|" + prev4AttrWord;
        String bigramAttrWord43 = prev4AttrWord + "|" + prev3AttrWord;
        String bigramAttrWord32 = prev3AttrWord + "|" + prev2AttrWord;
        generalFeatures.put("feature_attrWord_bigramAttrWord54_" + bigramAttrWord54, 1.0);
        generalFeatures.put("feature_attrWord_bigramAttrWord43_" + bigramAttrWord43, 1.0);
        generalFeatures.put("feature_attrWord_bigramAttrWord32_" + bigramAttrWord32, 1.0);
        
        String bigramAttrWordSkip53 = prev5AttrWord + "|" + prev3AttrWord;
        String bigramAttrWordSkip42 = prev4AttrWord + "|" + prev2AttrWord;
        String bigramAttrWordSkip31 = prev3AttrWord + "|" + prevAttrWord;
        generalFeatures.put("feature_attrWord_bigramAttrWordSkip53_" + bigramAttrWordSkip53, 1.0);
        generalFeatures.put("feature_attrWord_bigramAttrWordSkip42_" + bigramAttrWordSkip42, 1.0);
        generalFeatures.put("feature_attrWord_bigramAttrWordSkip31_" + bigramAttrWordSkip31, 1.0);
        
        String trigramAttrWord543 = prev5AttrWord + "|" + prev4AttrWord + "|" + prev3AttrWord;
        String trigramAttrWord432 = prev4AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord;
        generalFeatures.put("feature_attrWord_trigramAttrWord543_" + trigramAttrWord543, 1.0);
        generalFeatures.put("feature_attrWord_trigramAttrWord432_" + trigramAttrWord432, 1.0);
        
        String trigramAttrWordSkip542 = prev5AttrWord + "|" + prev4AttrWord + "|" + prev2AttrWord;
        String trigramAttrWordSkip532 = prev5AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord;
        String trigramAttrWordSkip431 = prev4AttrWord + "|" + prev3AttrWord + "|" + prevAttrWord;
        String trigramAttrWordSkip421 = prev4AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;
        generalFeatures.put("feature_attrWord_trigramAttrWordSkip542_" + trigramAttrWordSkip542, 1.0);
        generalFeatures.put("feature_attrWord_trigramAttrWordSkip532_" + trigramAttrWordSkip532, 1.0);
        generalFeatures.put("feature_attrWord_trigramAttrWordSkip431_" + trigramAttrWordSkip431, 1.0);
        generalFeatures.put("feature_attrWord_trigramAttrWordSkip421_" + trigramAttrWordSkip421, 1.0);*/
        //Previous AttrValue|Word features
        for (int j = 1; j <= 1; j++) {
            String previousAttrWord = "@@";
            if (generatedWords.size() - j >= 0) {
                previousAttrWord = generatedWords.get(generatedWords.size() - j).getAttribute().trim() + "|" + generatedWords.get(generatedWords.size() - j).getWord().trim();
            }
            generalFeatures.put("feature_attrValueWord_" + j + "_" + previousAttrWord.toLowerCase(), 1.0);
        }
        String prevAttrValueWord = "@@";
        if (generatedWords.size() - 1 >= 0) {
            prevAttrValueWord = generatedWords.get(generatedWords.size() - 1).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 1).getWord().trim();
        }
        String prev2AttrValueWord = "@@";
        if (generatedWords.size() - 2 >= 0) {
            prev2AttrValueWord = generatedWords.get(generatedWords.size() - 2).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 2).getWord().trim();
        }
        String prev3AttrValueWord = "@@";
        if (generatedWords.size() - 3 >= 0) {
            prev3AttrValueWord = generatedWords.get(generatedWords.size() - 3).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 3).getWord().trim();
        }
        String prev4AttrValueWord = "@@";
        if (generatedWords.size() - 4 >= 0) {
            prev4AttrValueWord = generatedWords.get(generatedWords.size() - 4).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 4).getWord().trim();
        }
        String prev5AttrValueWord = "@@";
        if (generatedWords.size() - 5 >= 0) {
            prev5AttrValueWord = generatedWords.get(generatedWords.size() - 5).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 5).getWord().trim();
        }

        String prevAttrValueWordBigram = prev2AttrValueWord + "|" + prevAttrValueWord;
        String prevAttrValueWordTrigram = prev3AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;
        String prevAttrValueWord4gram = prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;
        String prevAttrValueWord5gram = prev5AttrValueWord + "|" + prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;

        generalFeatures.put("feature_attrValueWord_bigram_" + prevAttrValueWordBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValueWord_trigram_" + prevAttrValueWordTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValueWord_4gram_" + prevAttrValueWord4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValueWord_5gram_" + prevAttrValueWord5gram.toLowerCase(), 1.0);

        /*String bigramAttrValueWord54 = prev5AttrValueWord + "|" + prev4AttrValueWord;
        String bigramAttrValueWord43 = prev4AttrValueWord + "|" + prev3AttrValueWord;
        String bigramAttrValueWord32 = prev3AttrValueWord + "|" + prev2AttrValueWord;
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWord54_" + bigramAttrValueWord54, 1.0);
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWord43_" + bigramAttrValueWord43, 1.0);
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWord32_" + bigramAttrValueWord32, 1.0);
        
        String bigramAttrValueWordSkip53 = prev5AttrValueWord + "|" + prev3AttrValueWord;
        String bigramAttrValueWordSkip42 = prev4AttrValueWord + "|" + prev2AttrValueWord;
        String bigramAttrValueWordSkip31 = prev3AttrValueWord + "|" + prevAttrValueWord;
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWordSkip53_" + bigramAttrValueWordSkip53, 1.0);
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWordSkip42_" + bigramAttrValueWordSkip42, 1.0);
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWordSkip31_" + bigramAttrValueWordSkip31, 1.0);
        
        String trigramAttrValueWord543 = prev5AttrValueWord + "|" + prev4AttrValueWord + "|" + prev3AttrValueWord;
        String trigramAttrValueWord432 = prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord;
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWord543_" + trigramAttrValueWord543, 1.0);
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWord432_" + trigramAttrValueWord432, 1.0);
        
        String trigramAttrValueWordSkip542 = prev5AttrValueWord + "|" + prev4AttrValueWord + "|" + prev2AttrValueWord;
        String trigramAttrValueWordSkip532 = prev5AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord;
        String trigramAttrValueWordSkip431 = prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prevAttrValueWord;
        String trigramAttrValueWordSkip421 = prev4AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWordSkip542_" + trigramAttrValueWordSkip542, 1.0);
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWordSkip532_" + trigramAttrValueWordSkip532, 1.0);
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWordSkip431_" + trigramAttrValueWordSkip431, 1.0);
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWordSkip421_" + trigramAttrValueWordSkip421, 1.0);*/

        /*
        System.out.println("5AV: " + prev5gramAttrValue);
        System.out.println("5A: " + prev5gramAttr);
        System.out.println("MA: " + mentionedAttrs);
        System.out.println("A_TBM: " + attrsToBeMentioned);
        System.out.println("MAV: " + mentionedAttrValues.subList(0, w));
        System.out.println("AV_TBM: " + attrValuesToBeMentioned);
        System.out.println(costs);
        System.out.println("==============================");*/
        //If arguments have been generated or not
        for (String attr : attrValuesAlreadyMentioned) {
            generalFeatures.put("feature_attr_alreadyMentioned_" + attr, 1.0);
        }
        //If arguments should still be generated or not
        for (String attr : attrsToBeMentioned) {
            generalFeatures.put("feature_attr_toBeMentioned_" + attr, 1.0);
        }

        //Attr specific features (and global features)
        for (String action : attributes.get(predicate)) {
            if (action.equals(Bagel.TOKEN_END)) {
                if (attrsToBeMentioned.isEmpty()) {
                    //valueSpecificFeatures.get(action).put("feature_specific_allAttrValuesMentioned", 1.0);
                    valueSpecificFeatures.get(action).put("global_feature_specific_allAttrValuesMentioned", 1.0);
                } else {
                    //valueSpecificFeatures.get(action).put("feature_specific_allAttrValuesNotMentioned", 1.0);
                    valueSpecificFeatures.get(action).put("global_feature_specific_allAttrValuesNotMentioned", 1.0);
                }
            } else {
                //Is attr in MR?
                if (MR.getAttributes().get(action) != null) {
                    //valueSpecificFeatures.get(action).put("feature_specific_isInMR", 1.0);
                    valueSpecificFeatures.get(action).put("global_feature_specific_isInMR", 1.0);
                } else {
                    //valueSpecificFeatures.get(action).put("feature_specific_isNotInMR", 1.0);
                    valueSpecificFeatures.get(action).put("global_feature_specific_isNotInMR", 1.0);
                }
                //Is attr already mentioned right before
                if (prevAttr.equals(action)) {
                    //valueSpecificFeatures.get(action).put("feature_specific_attrFollowingSameAttr", 1.0);
                    valueSpecificFeatures.get(action).put("global_feature_specific_attrFollowingSameAttr", 1.0);
                } else {
                    //valueSpecificFeatures.get(action).put("feature_specific_attrNotFollowingSameAttr", 1.0);
                    valueSpecificFeatures.get(action).put("global_feature_specific_attrNotFollowingSameAttr", 1.0);
                }
                //Is attr already mentioned
                for (String attrValue : attrValuesAlreadyMentioned) {
                    if (attrValue.substring(0, attrValue.indexOf('=')).equals(action)) {
                        //valueSpecificFeatures.get(action).put("feature_specific_attrAlreadyMentioned", 1.0);
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrAlreadyMentioned", 1.0);
                    }
                }
                //Is attr to be mentioned
                boolean toBeMentioned = false;
                for (String attrValue : attrValuesToBeMentioned) {
                        if (attrValue.substring(0, attrValue.indexOf('=')).equals(action)) {
                            toBeMentioned = true;
                            valueSpecificFeatures.get(action).put("global_feature_specific_attrToBeMentioned", 1.0);
                        }
                }
                if (!toBeMentioned) {
                    valueSpecificFeatures.get(action).put("global_feature_specific_attrNotToBeMentioned", 1.0);
                }
            }
            HashSet<String> keys = new HashSet<>(valueSpecificFeatures.get(action).keySet());
            for (String feature1 : keys) {
                for (String feature2 : keys) {
                    if (valueSpecificFeatures.get(action).get(feature1) == 1.0
                            && valueSpecificFeatures.get(action).get(feature2) == 1.0
                            && feature1.compareTo(feature2) < 0) {
                        valueSpecificFeatures.get(action).put(feature1 + "&&" + feature2, 1.0);
                    }
                }
            }
        }
        return new Instance(generalFeatures, valueSpecificFeatures, costs);
    }

    public Instance createWordInstance(String predicate, Action bestAction, ArrayList<String> previousGeneratedAttributes, ArrayList<Action> previousGeneratedWords, boolean wasValueMentioned, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesThatFollow, MeaningRepresentation MR, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams, boolean isRandom) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        if (!bestAction.getWord().trim().isEmpty()) {
            //COSTS
            String attr = bestAction.getAttribute().substring(0, bestAction.getAttribute().indexOf('='));
            if (bestAction.getAttribute().contains("=")) {
                attr = bestAction.getAttribute().substring(0, bestAction.getAttribute().indexOf('='));
            }
            for (Action action : availableWordActions.get(attr)) {
                if (action.getWord().equalsIgnoreCase(bestAction.getWord().trim())) {
                    costs.put(action.getWord().toLowerCase(), 0.0);
                } else {
                    costs.put(action.getWord().toLowerCase(), 1.0);
                }
            }

            if (bestAction.getWord().trim().equalsIgnoreCase(Bagel.TOKEN_END)) {
                costs.put(Bagel.TOKEN_END, 0.0);
            } else {
                costs.put(Bagel.TOKEN_END, 1.0);
            }
        }
        return createWordInstance(predicate, bestAction.getAttribute(), previousGeneratedAttributes, previousGeneratedWords, costs, wasValueMentioned, attrValuesAlreadyMentioned, attrValuesThatFollow, MR, availableWordActions, nGrams);
    }

    public Instance createWordInstance(String predicate, String currentAttrValue, ArrayList<String> generatedAttributes, ArrayList<Action> previousGeneratedWords, TObjectDoubleHashMap<String> costs, boolean wasValueMentioned, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesThatFollow, MeaningRepresentation MR, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams) {
        String currentAttr = currentAttrValue;
        String currentValue = "";
        if (currentAttr.contains("=")) {
            currentAttr = currentAttrValue.substring(0, currentAttrValue.indexOf('='));
            currentValue = currentAttrValue.substring(currentAttrValue.indexOf('=') + 1);
        }
        
        TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        for (Action action : availableWordActions.get(currentAttr)) {
            valueSpecificFeatures.put(action.getWord(), new TObjectDoubleHashMap<String>());
        }

        /*if (gWords.get(wIndex).getWord().equals(Bagel.TOKEN_END)) {
        System.out.println("!!! "+ gWords.subList(0, wIndex + 1));
        }*/
        ArrayList<Action> generatedWords = new ArrayList<>();
        ArrayList<Action> generatedWordsInSameAttrValue = new ArrayList<>();
        ArrayList<String> generatedPhrase = new ArrayList<>();
        for (int i = 0; i < previousGeneratedWords.size(); i++) {
            Action a = previousGeneratedWords.get(i);
            if (!a.getWord().equals(Bagel.TOKEN_START)
                    && !a.getWord().equals(Bagel.TOKEN_END)) {
                generatedWords.add(a);
                generatedPhrase.add(a.getWord());
                if (a.getAttribute().equals(currentAttrValue)) {
                    generatedWordsInSameAttrValue.add(a);
                }
            }
        }

        //Previous word features
        for (int j = 1; j <= 1; j++) {
            String previousWord = "@@";
            if (generatedWords.size() - j >= 0) {
                previousWord = generatedWords.get(generatedWords.size() - j).getWord().trim();
            }
            generalFeatures.put("feature_word_" + j + "_" + previousWord.toLowerCase(), 1.0);
        }
        String prevWord = "@@";
        if (generatedWords.size() - 1 >= 0) {
            prevWord = generatedWords.get(generatedWords.size() - 1).getWord().trim();
        }
        String prev2Word = "@@";
        if (generatedWords.size() - 2 >= 0) {
            prev2Word = generatedWords.get(generatedWords.size() - 2).getWord().trim();
        }
        String prev3Word = "@@";
        if (generatedWords.size() - 3 >= 0) {
            prev3Word = generatedWords.get(generatedWords.size() - 3).getWord().trim();
        }
        String prev4Word = "@@";
        if (generatedWords.size() - 4 >= 0) {
            prev4Word = generatedWords.get(generatedWords.size() - 4).getWord().trim();
        }
        String prev5Word = "@@";
        if (generatedWords.size() - 5 >= 0) {
            prev5Word = generatedWords.get(generatedWords.size() - 5).getWord().trim();
        }

        String prevBigram = prev2Word + "|" + prevWord;
        String prevTrigram = prev3Word + "|" + prev2Word + "|" + prevWord;
        String prev4gram = prev4Word + "|" + prev3Word + "|" + prev2Word + "|" + prevWord;
        String prev5gram = prev5Word + "|" + prev4Word + "|" + prev3Word + "|" + prev2Word + "|" + prevWord;

        generalFeatures.put("feature_word_bigram_" + prevBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_word_trigram_" + prevTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_word_4gram_" + prev4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_word_5gram_" + prev5gram.toLowerCase(), 1.0);

        /*String bigramWord54 = prev5Word + "|" + prev4Word;
        String bigramWord43 = prev4Word + "|" + prev3Word;
        String bigramWord32 = prev3Word + "|" + prev2Word;
        generalFeatures.put("feature_word_bigramWord54_" + bigramWord54, 1.0);
        generalFeatures.put("feature_word_bigramWord43_" + bigramWord43, 1.0);
        generalFeatures.put("feature_word_bigramWord32_" + bigramWord32, 1.0);
        
        String bigramWordSkip53 = prev5Word + "|" + prev3Word;
        String bigramWordSkip42 = prev4Word + "|" + prev2Word;
        String bigramWordSkip31 = prev3Word + "|" + prevWord;
        generalFeatures.put("feature_word_bigramWordSkip53_" + bigramWordSkip53, 1.0);
        generalFeatures.put("feature_word_bigramWordSkip42_" + bigramWordSkip42, 1.0);
        generalFeatures.put("feature_word_bigramWordSkip31_" + bigramWordSkip31, 1.0);
        
        String trigramWord543 = prev5Word + "|" + prev4Word + "|" + prev3Word;
        String trigramWord432 = prev4Word + "|" + prev3Word + "|" + prev2Word;
        generalFeatures.put("feature_word_trigramWord543_" + trigramWord543, 1.0);
        generalFeatures.put("feature_word_trigramWord432_" + trigramWord432, 1.0);
        
        String trigramWordSkip542 = prev5Word + "|" + prev4Word + "|" + prev2Word;
        String trigramWordSkip532 = prev5Word + "|" + prev3Word + "|" + prev2Word;
        String trigramWordSkip431 = prev4Word + "|" + prev3Word + "|" + prevWord;
        String trigramWordSkip421 = prev4Word + "|" + prev2Word + "|" + prevWord;
        generalFeatures.put("feature_word_trigramWordSkip542_" + trigramWordSkip542, 1.0);
        generalFeatures.put("feature_word_trigramWordSkip532_" + trigramWordSkip532, 1.0);
        generalFeatures.put("feature_word_trigramWordSkip431_" + trigramWordSkip431, 1.0);
        generalFeatures.put("feature_word_trigramWordSkip421_" + trigramWordSkip421, 1.0);*/
        //Previous words in same as current attrValue features
        if (generatedWordsInSameAttrValue.isEmpty()) {
            generalFeatures.put("feature_currentAttrValueWord_isEmpty", 1.0);
        }

        for (int j = 1; j <= 1; j++) {
            String previousCurrentAttrValueWord = "@@";
            if (generatedWordsInSameAttrValue.size() - j >= 0) {
                previousCurrentAttrValueWord = generatedWordsInSameAttrValue.get(generatedWordsInSameAttrValue.size() - j).getWord().trim();
            }
            generalFeatures.put("feature_currentAttrValueWord_" + j + "_" + previousCurrentAttrValueWord.toLowerCase(), 1.0);
        }
        String prevCurrentAttrValueWord = "@@";
        if (generatedWordsInSameAttrValue.size() - 1 >= 0) {
            prevCurrentAttrValueWord = generatedWordsInSameAttrValue.get(generatedWordsInSameAttrValue.size() - 1).getWord().trim();
        }
        String prev2CurrentAttrValueWord = "@@";
        if (generatedWordsInSameAttrValue.size() - 2 >= 0) {
            prev2CurrentAttrValueWord = generatedWordsInSameAttrValue.get(generatedWordsInSameAttrValue.size() - 2).getWord().trim();
        }
        String prev3CurrentAttrValueWord = "@@";
        if (generatedWordsInSameAttrValue.size() - 3 >= 0) {
            prev3CurrentAttrValueWord = generatedWordsInSameAttrValue.get(generatedWordsInSameAttrValue.size() - 3).getWord().trim();
        }
        String prev4CurrentAttrValueWord = "@@";
        if (generatedWordsInSameAttrValue.size() - 4 >= 0) {
            prev4CurrentAttrValueWord = generatedWordsInSameAttrValue.get(generatedWordsInSameAttrValue.size() - 4).getWord().trim();
        }
        String prev5CurrentAttrValueWord = "@@";
        if (generatedWordsInSameAttrValue.size() - 5 >= 0) {
            prev5CurrentAttrValueWord = generatedWordsInSameAttrValue.get(generatedWordsInSameAttrValue.size() - 5).getWord().trim();
        }

        String prevCurrentAttrValueBigram = prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        String prevCurrentAttrValueTrigram = prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        String prevCurrentAttrValue4gram = prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        String prevCurrentAttrValue5gram = prev5CurrentAttrValueWord + "|" + prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;

        generalFeatures.put("feature_currentAttrValueWord_bigram_" + prevCurrentAttrValueBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_currentAttrValueWord_trigram_" + prevCurrentAttrValueTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_currentAttrValueWord_4gram_" + prevCurrentAttrValue4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_currentAttrValueWord_5gram_" + prevCurrentAttrValue5gram.toLowerCase(), 1.0);

        /*String bigramCurrentAttrValueWord54 = prev5CurrentAttrValueWord + "|" + prev4CurrentAttrValueWord;
        String bigramCurrentAttrValueWord43 = prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord;
        String bigramCurrentAttrValueWord32 = prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        generalFeatures.put("feature_currentAttrValueWord_bigramCurrentAttrValueWord54_" + bigramCurrentAttrValueWord54, 1.0);
        generalFeatures.put("feature_currentAttrValueWord_bigramCurrentAttrValueWord43_" + bigramCurrentAttrValueWord43, 1.0);
        generalFeatures.put("feature_currentAttrValueWord_bigramCurrentAttrValueWord32_" + bigramCurrentAttrValueWord32, 1.0);
        
        String bigramCurrentAttrValueWordSkip53 = prev5CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord;
        String bigramCurrentAttrValueWordSkip42 = prev4CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        String bigramCurrentAttrValueWordSkip31 = prev3CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        generalFeatures.put("feature_currentAttrValueWord_bigramCurrentAttrValueWordSkip53_" + bigramCurrentAttrValueWordSkip53, 1.0);
        generalFeatures.put("feature_currentAttrValueWord_bigramCurrentAttrValueWordSkip42_" + bigramCurrentAttrValueWordSkip42, 1.0);
        generalFeatures.put("feature_currentAttrValueWord_bigramCurrentAttrValueWordSkip31_" + bigramCurrentAttrValueWordSkip31, 1.0);
        
        String trigramCurrentAttrValueWord543 = prev5CurrentAttrValueWord + "|" + prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord;
        String trigramCurrentAttrValueWord432 = prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        generalFeatures.put("feature_currentAttrValueWord_trigramCurrentAttrValueWord543_" + trigramCurrentAttrValueWord543, 1.0);
        generalFeatures.put("feature_currentAttrValueWord_trigramCurrentAttrValueWord432_" + trigramCurrentAttrValueWord432, 1.0);
        
        String trigramCurrentAttrValueWordSkip542 = prev5CurrentAttrValueWord + "|" + prev4CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        String trigramCurrentAttrValueWordSkip532 = prev5CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord;
        String trigramCurrentAttrValueWordSkip431 = prev4CurrentAttrValueWord + "|" + prev3CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        String trigramCurrentAttrValueWordSkip421 = prev4CurrentAttrValueWord + "|" + prev2CurrentAttrValueWord + "|" + prevCurrentAttrValueWord;
        generalFeatures.put("feature_currentAttrValueWord_trigramCurrentAttrValueWordSkip542_" + trigramCurrentAttrValueWordSkip542, 1.0);
        generalFeatures.put("feature_currentAttrValueWord_trigramCurrentAttrValueWordSkip532_" + trigramCurrentAttrValueWordSkip532, 1.0);
        generalFeatures.put("feature_currentAttrValueWord_trigramCurrentAttrValueWordSkip431_" + trigramCurrentAttrValueWordSkip431, 1.0);
        generalFeatures.put("feature_currentAttrValueWord_trigramCurrentAttrValueWordSkip421_" + trigramCurrentAttrValueWordSkip421, 1.0);*/
        //Previous Attr|Word features
        for (int j = 1; j <= 1; j++) {
            String previousAttrWord = "@@";
            if (generatedWords.size() - j >= 0) {
                if (generatedWords.get(generatedWords.size() - j).getAttribute().contains("=")) {
                    previousAttrWord = generatedWords.get(generatedWords.size() - j).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - j).getAttribute().indexOf('=')) + "|" + generatedWords.get(generatedWords.size() - j).getWord().trim();
                } else {
                    previousAttrWord = generatedWords.get(generatedWords.size() - j).getAttribute().trim() + "|" + generatedWords.get(generatedWords.size() - j).getWord().trim();
                }
            }
            generalFeatures.put("feature_attrWord_" + j + "_" + previousAttrWord.toLowerCase(), 1.0);
        }
        String prevAttrWord = "@@";
        if (generatedWords.size() - 1 >= 0) {
            if (generatedWords.get(generatedWords.size() - 1).getAttribute().contains("=")) {
                prevAttrWord = generatedWords.get(generatedWords.size() - 1).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 1).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 1).getWord().trim();
            } else {
                prevAttrWord = generatedWords.get(generatedWords.size() - 1).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 1).getWord().trim();

            }
        }
        String prev2AttrWord = "@@";
        if (generatedWords.size() - 2 >= 0) {
            if (generatedWords.get(generatedWords.size() - 2).getAttribute().contains("=")) {
                prev2AttrWord = generatedWords.get(generatedWords.size() - 2).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 2).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 2).getWord().trim();
            } else {
                prev2AttrWord = generatedWords.get(generatedWords.size() - 2).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 2).getWord().trim();
            }
        }
        String prev3AttrWord = "@@";
        if (generatedWords.size() - 3 >= 0) {
            if (generatedWords.get(generatedWords.size() - 3).getAttribute().contains("=")) {
                prev3AttrWord = generatedWords.get(generatedWords.size() - 3).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 3).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 3).getWord().trim();
            } else {
                prev3AttrWord = generatedWords.get(generatedWords.size() - 3).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 3).getWord().trim();
            }
        }
        String prev4AttrWord = "@@";
        if (generatedWords.size() - 4 >= 0) {
            if (generatedWords.get(generatedWords.size() - 4).getAttribute().contains("=")) {
                prev4AttrWord = generatedWords.get(generatedWords.size() - 4).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 4).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 4).getWord().trim();
            } else {
                prev4AttrWord = generatedWords.get(generatedWords.size() - 4).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 4).getWord().trim();
            }
        }
        String prev5AttrWord = "@@";
        if (generatedWords.size() - 5 >= 0) {
            if (generatedWords.get(generatedWords.size() - 5).getAttribute().contains("=")) {
                prev5AttrWord = generatedWords.get(generatedWords.size() - 5).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 5).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 5).getWord().trim();
            } else {
                prev5AttrWord = generatedWords.get(generatedWords.size() - 5).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 5).getWord().trim();
            }
        }

        String prevAttrWordBigram = prev2AttrWord + "|" + prevAttrWord;
        String prevAttrWordTrigram = prev3AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;
        String prevAttrWord4gram = prev4AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;
        String prevAttrWord5gram = prev5AttrWord + "|" + prev4AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;

        generalFeatures.put("feature_attrWord_bigram_" + prevAttrWordBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrWord_trigram_" + prevAttrWordTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrWord_4gram_" + prevAttrWord4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrWord_5gram_" + prevAttrWord5gram.toLowerCase(), 1.0);

        /*String bigramAttrWord54 = prev5AttrWord + "|" + prev4AttrWord;
        String bigramAttrWord43 = prev4AttrWord + "|" + prev3AttrWord;
        String bigramAttrWord32 = prev3AttrWord + "|" + prev2AttrWord;
        generalFeatures.put("feature_attrWord_bigramAttrWord54_" + bigramAttrWord54, 1.0);
        generalFeatures.put("feature_attrWord_bigramAttrWord43_" + bigramAttrWord43, 1.0);
        generalFeatures.put("feature_attrWord_bigramAttrWord32_" + bigramAttrWord32, 1.0);
        
        String bigramAttrWordSkip53 = prev5AttrWord + "|" + prev3AttrWord;
        String bigramAttrWordSkip42 = prev4AttrWord + "|" + prev2AttrWord;
        String bigramAttrWordSkip31 = prev3AttrWord + "|" + prevAttrWord;
        generalFeatures.put("feature_attrWord_bigramAttrWordSkip53_" + bigramAttrWordSkip53, 1.0);
        generalFeatures.put("feature_attrWord_bigramAttrWordSkip42_" + bigramAttrWordSkip42, 1.0);
        generalFeatures.put("feature_attrWord_bigramAttrWordSkip31_" + bigramAttrWordSkip31, 1.0);
        
        String trigramAttrWord543 = prev5AttrWord + "|" + prev4AttrWord + "|" + prev3AttrWord;
        String trigramAttrWord432 = prev4AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord;
        generalFeatures.put("feature_attrWord_trigramAttrWord543_" + trigramAttrWord543, 1.0);
        generalFeatures.put("feature_attrWord_trigramAttrWord432_" + trigramAttrWord432, 1.0);
        
        String trigramAttrWordSkip542 = prev5AttrWord + "|" + prev4AttrWord + "|" + prev2AttrWord;
        String trigramAttrWordSkip532 = prev5AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord;
        String trigramAttrWordSkip431 = prev4AttrWord + "|" + prev3AttrWord + "|" + prevAttrWord;
        String trigramAttrWordSkip421 = prev4AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;
        generalFeatures.put("feature_attrWord_trigramAttrWordSkip542_" + trigramAttrWordSkip542, 1.0);
        generalFeatures.put("feature_attrWord_trigramAttrWordSkip532_" + trigramAttrWordSkip532, 1.0);
        generalFeatures.put("feature_attrWord_trigramAttrWordSkip431_" + trigramAttrWordSkip431, 1.0);
        generalFeatures.put("feature_attrWord_trigramAttrWordSkip421_" + trigramAttrWordSkip421, 1.0);*/
        //Previous AttrValue|Word features
        for (int j = 1; j <= 1; j++) {
            String previousAttrWord = "@@";
            if (generatedWords.size() - j >= 0) {
                previousAttrWord = generatedWords.get(generatedWords.size() - j).getAttribute().trim() + "|" + generatedWords.get(generatedWords.size() - j).getWord().trim();
            }
            generalFeatures.put("feature_attrValueWord_" + j + "_" + previousAttrWord.toLowerCase(), 1.0);
        }
        String prevAttrValueWord = "@@";
        if (generatedWords.size() - 1 >= 0) {
            prevAttrValueWord = generatedWords.get(generatedWords.size() - 1).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 1).getWord().trim();
        }
        String prev2AttrValueWord = "@@";
        if (generatedWords.size() - 2 >= 0) {
            prev2AttrValueWord = generatedWords.get(generatedWords.size() - 2).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 2).getWord().trim();
        }
        String prev3AttrValueWord = "@@";
        if (generatedWords.size() - 3 >= 0) {
            prev3AttrValueWord = generatedWords.get(generatedWords.size() - 3).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 3).getWord().trim();
        }
        String prev4AttrValueWord = "@@";
        if (generatedWords.size() - 4 >= 0) {
            prev4AttrValueWord = generatedWords.get(generatedWords.size() - 4).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 4).getWord().trim();
        }
        String prev5AttrValueWord = "@@";
        if (generatedWords.size() - 5 >= 0) {
            prev5AttrValueWord = generatedWords.get(generatedWords.size() - 5).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 5).getWord().trim();
        }

        String prevAttrValueWordBigram = prev2AttrValueWord + "|" + prevAttrValueWord;
        String prevAttrValueWordTrigram = prev3AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;
        String prevAttrValueWord4gram = prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;
        String prevAttrValueWord5gram = prev5AttrValueWord + "|" + prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;

        generalFeatures.put("feature_attrValueWord_bigram_" + prevAttrValueWordBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValueWord_trigram_" + prevAttrValueWordTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValueWord_4gram_" + prevAttrValueWord4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValueWord_5gram_" + prevAttrValueWord5gram.toLowerCase(), 1.0);

        /*String bigramAttrValueWord54 = prev5AttrValueWord + "|" + prev4AttrValueWord;
        String bigramAttrValueWord43 = prev4AttrValueWord + "|" + prev3AttrValueWord;
        String bigramAttrValueWord32 = prev3AttrValueWord + "|" + prev2AttrValueWord;
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWord54_" + bigramAttrValueWord54, 1.0);
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWord43_" + bigramAttrValueWord43, 1.0);
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWord32_" + bigramAttrValueWord32, 1.0);
        
        String bigramAttrValueWordSkip53 = prev5AttrValueWord + "|" + prev3AttrValueWord;
        String bigramAttrValueWordSkip42 = prev4AttrValueWord + "|" + prev2AttrValueWord;
        String bigramAttrValueWordSkip31 = prev3AttrValueWord + "|" + prevAttrValueWord;
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWordSkip53_" + bigramAttrValueWordSkip53, 1.0);
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWordSkip42_" + bigramAttrValueWordSkip42, 1.0);
        generalFeatures.put("feature_attrValueWord_bigramAttrValueWordSkip31_" + bigramAttrValueWordSkip31, 1.0);
        
        String trigramAttrValueWord543 = prev5AttrValueWord + "|" + prev4AttrValueWord + "|" + prev3AttrValueWord;
        String trigramAttrValueWord432 = prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord;
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWord543_" + trigramAttrValueWord543, 1.0);
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWord432_" + trigramAttrValueWord432, 1.0);
        
        String trigramAttrValueWordSkip542 = prev5AttrValueWord + "|" + prev4AttrValueWord + "|" + prev2AttrValueWord;
        String trigramAttrValueWordSkip532 = prev5AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord;
        String trigramAttrValueWordSkip431 = prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prevAttrValueWord;
        String trigramAttrValueWordSkip421 = prev4AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWordSkip542_" + trigramAttrValueWordSkip542, 1.0);
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWordSkip532_" + trigramAttrValueWordSkip532, 1.0);
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWordSkip431_" + trigramAttrValueWordSkip431, 1.0);
        generalFeatures.put("feature_attrValueWord_trigramAttrValueWordSkip421_" + trigramAttrValueWordSkip421, 1.0);*/
        //Previous attrValue features
        int attributeSize = generatedAttributes.size();
        for (int j = 1; j <= 1; j++) {
            String previousAttrValue = "@@";
            if (attributeSize - j >= 0) {
                previousAttrValue = generatedAttributes.get(attributeSize - j).trim();
            }
            generalFeatures.put("feature_attrValue_" + j + "_" + previousAttrValue, 1.0);
        }
        String prevAttrValue = "@@";
        if (attributeSize - 1 >= 0) {
            prevAttrValue = generatedAttributes.get(attributeSize - 1).trim();
        }
        String prev2AttrValue = "@@";
        if (attributeSize - 2 >= 0) {
            prev2AttrValue = generatedAttributes.get(attributeSize - 2).trim();
        }
        String prev3AttrValue = "@@";
        if (attributeSize - 3 >= 0) {
            prev3AttrValue = generatedAttributes.get(attributeSize - 3).trim();
        }
        String prev4AttrValue = "@@";
        if (attributeSize - 4 >= 0) {
            prev4AttrValue = generatedAttributes.get(attributeSize - 4).trim();
        }
        String prev5AttrValue = "@@";
        if (attributeSize - 5 >= 0) {
            prev5AttrValue = generatedAttributes.get(attributeSize - 5).trim();
        }

        String prevAttrBigramValue = prev2AttrValue + "|" + prevAttrValue;
        String prevAttrTrigramValue = prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        String prevAttr4gramValue = prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        String prevAttr5gramValue = prev5AttrValue + "|" + prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;

        generalFeatures.put("feature_attrValue_bigram_" + prevAttrBigramValue.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValue_trigram_" + prevAttrTrigramValue.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValue_4gram_" + prevAttr4gramValue.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValue_5gram_" + prevAttr5gramValue.toLowerCase(), 1.0);

        /*String bigramAttrValue54 = prev5AttrValue + "|" + prev4AttrValue;
        String bigramAttrValue43 = prev4AttrValue + "|" + prev3AttrValue;
        String bigramAttrValue32 = prev3AttrValue + "|" + prev2AttrValue;
        generalFeatures.put("feature_attrValue_bigramAttrValue54_" + bigramAttrValue54, 1.0);
        generalFeatures.put("feature_attrValue_bigramAttrValue43_" + bigramAttrValue43, 1.0);
        generalFeatures.put("feature_attrValue_bigramAttrValue32_" + bigramAttrValue32, 1.0);
        
        String bigramAttrValueSkip53 = prev5AttrValue + "|" + prev3AttrValue;
        String bigramAttrValueSkip42 = prev4AttrValue + "|" + prev2AttrValue;
        String bigramAttrValueSkip31 = prev3AttrValue + "|" + prevAttrValue;
        generalFeatures.put("feature_attrValue_bigramAttrValueSkip53_" + bigramAttrValueSkip53, 1.0);
        generalFeatures.put("feature_attrValue_bigramAttrValueSkip42_" + bigramAttrValueSkip42, 1.0);
        generalFeatures.put("feature_attrValue_bigramAttrValueSkip31_" + bigramAttrValueSkip31, 1.0);
        
        String trigramAttrValue543 = prev5AttrValue + "|" + prev4AttrValue + "|" + prev3AttrValue;
        String trigramAttrValue432 = prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue;
        generalFeatures.put("feature_attrValue_trigramAttrValue543_" + trigramAttrValue543, 1.0);
        generalFeatures.put("feature_attrValue_trigramAttrValue432_" + trigramAttrValue432, 1.0);
        
        String trigramAttrValueSkip542 = prev5AttrValue + "|" + prev4AttrValue + "|" + prev2AttrValue;
        String trigramAttrValueSkip532 = prev5AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue;
        String trigramAttrValueSkip431 = prev4AttrValue + "|" + prev3AttrValue + "|" + prevAttrValue;
        String trigramAttrValueSkip421 = prev4AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        generalFeatures.put("feature_attrValue_trigramAttrValueSkip542_" + trigramAttrValueSkip542, 1.0);
        generalFeatures.put("feature_attrValue_trigramAttrValueSkip532_" + trigramAttrValueSkip532, 1.0);
        generalFeatures.put("feature_attrValue_trigramAttrValueSkip431_" + trigramAttrValueSkip431, 1.0);
        generalFeatures.put("feature_attrValue_trigramAttrValueSkip421_" + trigramAttrValueSkip421, 1.0);*/
        //Previous attr features
        for (int j = 1; j <= 1; j++) {
            String previousAttr = "@@";
            if (attributeSize - j >= 0) {
                if (generatedAttributes.get(attributeSize - j).contains("=")) {
                    previousAttr = generatedAttributes.get(attributeSize - j).trim().substring(0, generatedAttributes.get(attributeSize - j).indexOf('='));
                } else {
                    previousAttr = generatedAttributes.get(attributeSize - j).trim();
                }
            }
            generalFeatures.put("feature_attr_" + j + "_" + previousAttr, 1.0);
        }
        String prevAttr = "@@";
        if (attributeSize - 1 >= 0) {
            if (generatedAttributes.get(attributeSize - 1).contains("=")) {
                prevAttr = generatedAttributes.get(attributeSize - 1).trim().substring(0, generatedAttributes.get(attributeSize - 1).indexOf('='));
            } else {
                prevAttr = generatedAttributes.get(attributeSize - 1).trim();
            }
        }
        String prev2Attr = "@@";
        if (attributeSize - 2 >= 0) {
            if (generatedAttributes.get(attributeSize - 2).contains("=")) {
                prev2Attr = generatedAttributes.get(attributeSize - 2).trim().substring(0, generatedAttributes.get(attributeSize - 2).indexOf('='));
            } else {
                prev2Attr = generatedAttributes.get(attributeSize - 2).trim();
            }
        }
        String prev3Attr = "@@";
        if (attributeSize - 3 >= 0) {
            if (generatedAttributes.get(attributeSize - 3).contains("=")) {
                prev3Attr = generatedAttributes.get(attributeSize - 3).trim().substring(0, generatedAttributes.get(attributeSize - 3).indexOf('='));
            } else {
                prev3Attr = generatedAttributes.get(attributeSize - 3).trim();
            }
        }
        String prev4Attr = "@@";
        if (attributeSize - 4 >= 0) {
            if (generatedAttributes.get(attributeSize - 4).contains("=")) {
                prev4Attr = generatedAttributes.get(attributeSize - 4).trim().substring(0, generatedAttributes.get(attributeSize - 4).indexOf('='));
            } else {
                prev4Attr = generatedAttributes.get(attributeSize - 4).trim();
            }
        }
        String prev5Attr = "@@";
        if (attributeSize - 5 >= 0) {
            if (generatedAttributes.get(attributeSize - 5).contains("=")) {
                prev5Attr = generatedAttributes.get(attributeSize - 5).trim().substring(0, generatedAttributes.get(attributeSize - 5).indexOf('='));
            } else {
                prev5Attr = generatedAttributes.get(attributeSize - 5).trim();
            }
        }

        String prevAttrBigram = prev2Attr + "|" + prevAttr;
        String prevAttrTrigram = prev3Attr + "|" + prev2Attr + "|" + prevAttr;
        String prevAttr4gram = prev4Attr + "|" + prev3Attr + "|" + prev2Attr + "|" + prevAttr;
        String prevAttr5gram = prev5Attr + "|" + prev4Attr + "|" + prev3Attr + "|" + prev2Attr + "|" + prevAttr;

        generalFeatures.put("feature_attr_bigram_" + prevAttrBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attr_trigram_" + prevAttrTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attr_4gram_" + prevAttr4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attr_5gram_" + prevAttr5gram.toLowerCase(), 1.0);

        /*String bigramAttr54 = prev5Attr + "|" + prev4Attr;
        String bigramAttr43 = prev4Attr + "|" + prev3Attr;
        String bigramAttr32 = prev3Attr + "|" + prev2Attr;
        generalFeatures.put("feature_attr_bigramAttr54_" + bigramAttr54, 1.0);
        generalFeatures.put("feature_attr_bigramAttr43_" + bigramAttr43, 1.0);
        generalFeatures.put("feature_attr_bigramAttr32_" + bigramAttr32, 1.0);
        
        String bigramAttrSkip53 = prev5Attr + "|" + prev3Attr;
        String bigramAttrSkip42 = prev4Attr + "|" + prev2Attr;
        String bigramAttrSkip31 = prev3Attr + "|" + prevAttr;
        generalFeatures.put("feature_attr_bigramAttrSkip53_" + bigramAttrSkip53, 1.0);
        generalFeatures.put("feature_attr_bigramAttrSkip42_" + bigramAttrSkip42, 1.0);
        generalFeatures.put("feature_attr_bigramAttrSkip31_" + bigramAttrSkip31, 1.0);
        
        String trigramAttr543 = prev5Attr + "|" + prev4Attr + "|" + prev3Attr;
        String trigramAttr432 = prev4Attr + "|" + prev3Attr + "|" + prev2Attr;
        generalFeatures.put("feature_attr_trigramAttr543_" + trigramAttr543, 1.0);
        generalFeatures.put("feature_attr_trigramAttr432_" + trigramAttr432, 1.0);
        
        String trigramAttrSkip542 = prev5Attr + "|" + prev4Attr + "|" + prev2Attr;
        String trigramAttrSkip532 = prev5Attr + "|" + prev3Attr + "|" + prev2Attr;
        String trigramAttrSkip431 = prev4Attr + "|" + prev3Attr + "|" + prevAttr;
        String trigramAttrSkip421 = prev4Attr + "|" + prev2Attr + "|" + prevAttr;
        generalFeatures.put("feature_attr_trigramAttrSkip542_" + trigramAttrSkip542, 1.0);
        generalFeatures.put("feature_attr_trigramAttrSkip532_" + trigramAttrSkip532, 1.0);
        generalFeatures.put("feature_attr_trigramAttrSkip431_" + trigramAttrSkip431, 1.0);
        generalFeatures.put("feature_attr_trigramAttrSkip421_" + trigramAttrSkip421, 1.0);*/
        //If values have already been generated or not
        generalFeatures.put("feature_valueToBeMentioned_" + currentValue.toLowerCase(), 1.0);
        if (wasValueMentioned) {
            generalFeatures.put("feature_wasValueMentioned_true", 1.0);
        } else {
            //generalFeatures.put("feature_wasValueMentioned_false", 1.0);
        }
        HashSet<String> valuesThatFollow = new HashSet<>();
        for (String attrValue : attrValuesThatFollow) {
            generalFeatures.put("feature_attrValuesThatFollow_" + attrValue.toLowerCase(), 1.0);
            if (attrValue.contains("=")) {
                String v = attrValue.substring(attrValue.indexOf('=') + 1);
                if (v.matches("[xX][0-9]+")) {
                    String attr = attrValue.substring(0, attrValue.indexOf('='));
                    valuesThatFollow.add(Bagel.TOKEN_X + attr + "_" + v.substring(1));
                } else {
                    valuesThatFollow.add(v);
                }
                generalFeatures.put("feature_attrsThatFollow_" + attrValue.substring(0, attrValue.indexOf('=')).toLowerCase(), 1.0);
            } else {
                generalFeatures.put("feature_attrsThatFollow_" + attrValue.toLowerCase(), 1.0);
            }
        }
        HashSet<String> mentionedValues = new HashSet<>();
        for (String attrValue : attrValuesAlreadyMentioned) {
            generalFeatures.put("feature_attrValuesAlreadyMentioned_" + attrValue.toLowerCase(), 1.0);
            if (attrValue.contains("=")) {
                generalFeatures.put("feature_attrsAlreadyMentioned_" + attrValue.substring(0, attrValue.indexOf('=')).toLowerCase(), 1.0);
                String v = attrValue.substring(attrValue.indexOf('=') + 1);
                if (v.matches("[xX][0-9]+")) {
                    String attr = attrValue.substring(0, attrValue.indexOf('='));
                    mentionedValues.add(Bagel.TOKEN_X + attr + "_" + v.substring(1));
                } else {
                    mentionedValues.add(v);
                }
            } else {
                generalFeatures.put("feature_attrsAlreadyMentioned_" + attrValue.toLowerCase(), 1.0);
            }
        }

        /*System.out.println("5W: " + prev5gram);
        //System.out.println("5AW: " + prevAttrWord5gram);
        System.out.println("5A: " + prevAttr5gram);
        System.out.println("v_TBM: " + valueToBeMentioned);
        System.out.println("VM: " + wasValueMentioned);
        System.out.println("A_TF: " + attrValuesThatFollow);
        System.out.println("==============================");*/
        //Word specific features (and global features)
        for (Action action : availableWordActions.get(currentAttr)) {
            //Is word same as previous word
            if (prevWord.equals(action.getWord())) {
                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_sameAsPreviousWord", 1.0);
                valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_sameAsPreviousWord", 1.0);
            } else {
                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_notSameAsPreviousWord", 1.0);
                valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_notSameAsPreviousWord", 1.0);
            }
            //Has word appeared in the same attrValue before
            for (Action previousAction : generatedWords) {
                if (previousAction.getWord().equals(action.getWord())
                        && previousAction.getAttribute().equals(currentAttrValue)) {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_appearedInSameAttrValue", 1.0);
                    valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_appearedInSameAttrValue", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_notAppearedInSameAttrValue", 1.0);
                }
            }
            //Has word appeared before
            for (Action previousAction : generatedWords) {
                if (previousAction.getWord().equals(action.getWord())) {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_appeared", 1.0);
                    valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_appeared", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_notAppeared", 1.0);
                }
            }
            if (!action.getWord().startsWith(SFX.TOKEN_X)) {
                for (String value : valueAlignments.keySet()) {
                    for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                        if (alignedStr.get(0).equals(action.getWord())) {
                            if (mentionedValues.contains(value)) {
                                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_beginsValue_alreadyMentioned", 1.0);
                                valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_beginsValue_alreadyMentioned", 1.0);

                            } else if (currentValue.equals(value)) {
                                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_beginsValue_current", 1.0);
                                valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_beginsValue_current", 1.0);

                            } else if (valuesThatFollow.contains(value)) {
                                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_beginsValue_thatFollows", 1.0);
                                valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_beginsValue_thatFollows", 1.0);

                            } else {
                                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_beginsValue_notInMR", 1.0);
                                valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_beginsValue_notInMR", 1.0);

                            }
                        } else {
                            for (int i = 1; i < alignedStr.size(); i++) {
                                if (alignedStr.get(i).equals(action.getWord())) {
                                    if (endsWith(generatedPhrase, new ArrayList<String>(alignedStr.subList(0, i + 1)))) {
                                        if (mentionedValues.contains(value)) {
                                            //valueSpecificFeatures.get(action.getWord()).put("feature_specific_inValue_alreadyMentioned", 1.0);
                                            valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_inValue_alreadyMentioned", 1.0);

                                        } else if (currentValue.equals(value)) {
                                            //valueSpecificFeatures.get(action.getWord()).put("feature_specific_inValue_current", 1.0);
                                            valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_inValue_current", 1.0);

                                        } else if (valuesThatFollow.contains(value)) {
                                            //valueSpecificFeatures.get(action.getWord()).put("feature_specific_inValue_thatFollows", 1.0);
                                            valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_inValue_thatFollows", 1.0);

                                        } else {
                                            //valueSpecificFeatures.get(action.getWord()).put("feature_specific_inValue_notInMR", 1.0);
                                            valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_inValue_notInMR", 1.0);

                                        }
                                    } else {
                                        /*if (mentionedValues.contains(value)) {
                                        valueSpecificFeatures.get(action.getWord()).put("feature_specific_outOfValue_alreadyMentioned", 1.0);
                                        } else if (currentValue.equals(value)) {
                                        valueSpecificFeatures.get(action.getWord()).put("feature_specific_outOfValue_current", 1.0);
                                        } else if (valuesThatFollow.contains(value)) {
                                        valueSpecificFeatures.get(action.getWord()).put("feature_specific_outOfValue_thatFollows", 1.0);
                                        } else {
                                        valueSpecificFeatures.get(action.getWord()).put("feature_specific_outOfValue_notInMR", 1.0);
                                        }*/
                                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_outOfValue", 1.0);
                                        valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_outOfValue", 1.0);
                                    }
                                }
                            }
                        }
                    }
                }
                if (action.getWord().equals(SFX.TOKEN_END)) {
                    if (generatedWordsInSameAttrValue.isEmpty()) {
                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_closingEmptyAttr", 1.0);
                        valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_closingEmptyAttr", 1.0);
                    }
                    if (!wasValueMentioned) {
                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_closingAttrWithValueNotMentioned", 1.0);
                        valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_closingAttrWithValueNotMentioned", 1.0);
                    }

                    if (!prevCurrentAttrValueWord.equals("@@")) {
                        boolean alignmentIsOpen = false;
                        for (String value : valueAlignments.keySet()) {
                            for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                for (int i = 0; i < alignedStr.size() - 1; i++) {
                                    if (alignedStr.get(i).equals(prevCurrentAttrValueWord)) {
                                        if (endsWith(generatedPhrase, new ArrayList<String>(alignedStr.subList(0, i + 1)))) {
                                            alignmentIsOpen = true;
                                        }
                                    }
                                }
                            }
                        }
                        if (alignmentIsOpen) {
                            // valueSpecificFeatures.get(action.getWord()).put("feature_specific_closingAttrWhileValueIsNotConcluded", 1.0);
                            valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_closingAttrWhileValueIsNotConcluded", 1.0);
                        }
                    }
                }
            } else {
                if (currentValue.equals("no")
                    || currentValue.equals("yes")
                                        || currentValue.equals("yes or no")
                    || currentValue.equals("none")
                    || currentValue.equals("empty")
                    || currentValue.equals("dont_care")) {                    
                    valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_XValue_notInMR", 1.0);
                } else {
                    String currentValueVariant = "";
                    if (currentValue.matches("[xX][0-9]+")) {
                        currentValueVariant = SFX.TOKEN_X + currentAttr + "_" + currentValue.substring(1);
                    }

                    if (mentionedValues.contains(action.getWord())) {
                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_XValue_alreadyMentioned", 1.0);
                        valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_XValue_alreadyMentioned", 1.0);
                    } else if (currentValueVariant.equals(action.getWord())
                            && !currentValueVariant.isEmpty()) {
                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_XValue_current", 1.0);
                        valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_XValue_current", 1.0);

                    } else if (valuesThatFollow.contains(action.getWord())) {
                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_XValue_thatFollows", 1.0);
                        valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_XValue_thatFollows", 1.0);
                    } else {
                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_XValue_notInMR", 1.0);
                        valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_XValue_notInMR", 1.0);
                    }
                }
            }
            /*for (int i : nGrams.keySet()) {
            for (String nGram : nGrams.get(i)) {
            if (i == 2) {
            if (nGram.startsWith(prevWord + "|")
            && nGram.endsWith("|" + action.getWord())) {
            valueSpecificFeatures.get(action.getWord()).put("feature_specific_valuesFollowsPreviousWord", 1.0);
            }
            } else if (i == 3) {
            if (nGram.startsWith(prevBigram + "|")
            && nGram.endsWith("|" + action.getWord())) {
            valueSpecificFeatures.get(action.getWord()).put("feature_specific_valuesFollowsPreviousBigram", 1.0);
            }
            } else if (i == 4) {
            if (nGram.startsWith(prevTrigram + "|")
            && nGram.endsWith("|" + action.getWord())) {
            valueSpecificFeatures.get(action.getWord()).put("feature_specific_valuesFollowsPreviousTrigram", 1.0);
            }
            } else if (i == 5) {
            if (nGram.startsWith(prev4gram + "|")
            && nGram.endsWith("|" + action.getWord())) {
            valueSpecificFeatures.get(action.getWord()).put("feature_specific_valuesFollowsPrevious4gram", 1.0);
            }
            } else if (i == 6) {
            if (nGram.startsWith(prev5gram + "|")
            && nGram.endsWith("|" + action.getWord())) {
            valueSpecificFeatures.get(action.getWord()).put("feature_specific_valuesFollowsPrevious5gram", 1.0);
            }
            }
            }
            }*/
            HashSet<String> keys = new HashSet<>(valueSpecificFeatures.get(action.getWord()).keySet());
            for (String feature1 : keys) {
                for (String feature2 : keys) {
                    if (valueSpecificFeatures.get(action.getWord()).get(feature1) == 1.0
                            && valueSpecificFeatures.get(action.getWord()).get(feature2) == 1.0
                            && feature1.compareTo(feature2) < 0) {
                        valueSpecificFeatures.get(action.getWord()).put(feature1 + "&&" + feature2, 1.0);
                    }
                }
            }
        }

        return new Instance(generalFeatures, valueSpecificFeatures, costs);
    }

    public boolean endsWith(ArrayList<String> phrase, ArrayList<String> subPhrase) {
        if (subPhrase.size() > phrase.size()) {
            return false;
        }
        for (int i = 0; i < subPhrase.size(); i++) {
            if (!subPhrase.get(subPhrase.size() - 1 - i).equals(phrase.get(phrase.size() - 1 - i))) {
                return false;
            }
        }
        return true;
    }

    public String chooseNextValue(String attribute, HashSet<String> attrValuesToBeMentioned, ArrayList<DatasetInstance> trainingData) {
        HashMap<String, Integer> relevantValues = new HashMap<>();
        for (String attrValue : attrValuesToBeMentioned) {
            String attr = attrValue.substring(0, attrValue.indexOf('='));
            String value = attrValue.substring(attrValue.indexOf('=') + 1);
            if (attr.equals(attribute)) {
                relevantValues.put(value, 0);
            }
        }
        if (!relevantValues.isEmpty()) {
            if (relevantValues.keySet().size() == 1) {
                for (String value : relevantValues.keySet()) {
                    return value;
                }
            } else {
                String bestValue = "";
                int minIndex = Integer.MAX_VALUE;
                for (String value : relevantValues.keySet()) {
                    if (value.startsWith("x")) {
                        int vI = Integer.parseInt(value.substring(1));
                        if (vI < minIndex) {
                            minIndex = vI;
                            bestValue = value;
                        }
                    }
                }
                if (!bestValue.isEmpty()) {
                    return bestValue;
                }
                for (DatasetInstance di : trainingData) {
                    for (ArrayList<Action> ac : di.getEvalMentionedValueSequences().keySet()) {
                        ArrayList<String> mentionedValueSeq = di.getEvalMentionedValueSequences().get(ac);
                        boolean doesSeqContainValues = true;
                        minIndex = Integer.MAX_VALUE;
                        for (String value : relevantValues.keySet()) {
                            int index = mentionedValueSeq.indexOf(attribute + "=" + value);
                            if (index != -1
                                    && index < minIndex) {
                                minIndex = index;
                                bestValue = value;
                            } else if (index == -1) {
                                doesSeqContainValues = false;
                            }
                        }
                        if (doesSeqContainValues) {
                            relevantValues.put(bestValue, relevantValues.get(bestValue) + 1);
                        }
                    }
                }
                int max = -1;
                for (String value : relevantValues.keySet()) {
                    if (relevantValues.get(value) > max) {
                        max = relevantValues.get(value);
                        bestValue = value;
                    }
                }
                return bestValue;
            }
        }
        return "";
    }
}
