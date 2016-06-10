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
import jdagger.JDAggerForSFX;
import org.json.JSONArray;
import org.json.JSONException;
import similarity_measures.Levenshtein;

public class SFX {

    HashMap<String, HashSet<String>> attributes = new HashMap<>();
    HashMap<String, HashSet<String>> attributeValuePairs = new HashMap<>();
    HashMap<String, ArrayList<DatasetInstance>> datasetInstances = new HashMap<>();
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
    public HashMap<String, String> wenDaToGen = new HashMap<>();
    public ArrayList<Double> crossAvgArgDistances = new ArrayList<>();
    public ArrayList<Double> crossNIST = new ArrayList<>();
    public ArrayList<Double> crossBLEU = new ArrayList<>();
    public ArrayList<Double> crossBLEUSmooth = new ArrayList<>();
    static final int seed = 13;
    public static Random r = new Random(seed);
    double wordRefRolloutChance = 0.8;

    public static void main(String[] args) {
        boolean useDAggerArg = false;
        boolean useLolsWord = true;

        JDAggerForSFX.earlyStopMaxFurtherSteps = Integer.parseInt(args[0]);
        JDAggerForSFX.p = Double.parseDouble(args[1]);

        SFX sfx = new SFX();
        sfx.runTestWithJAROW(useDAggerArg, useLolsWord);
    }
    public static String dataset = "hotel";
    //public static String dataset = "restaurant";

    public void runTestWithJAROW(boolean useDAggerArg, boolean useDAggerWord) {
        File dataFile = new File("sfx_data/sfx" + dataset + "/train+valid+test.json");

        String wenFile = "results/wenResults/sfxhotel.log";
        if (dataset.equals("restaurant")) {
            wenFile = "results/wenResults/sfxrest.log";
        }

        boolean useValidation = true;
        boolean useRandomAlignments = false;
        createLists(dataFile);

        //RANDOM DATA SPLIT
        /*int to = (int) Math.round(datasetInstances.size() * 0.1);
         if (to > datasetInstances.size()) {
         to = datasetInstances.size();
         }
         ArrayList<DatasetInstance> trainingData = new ArrayList<>();
         ArrayList<DatasetInstance> validationData = new ArrayList<>();
         ArrayList<DatasetInstance> testingData = new ArrayList<>();

         for (String predicate : predicates) {
         trainingData.addAll(datasetInstances.get(predicate).subList(0, ((int) Math.round(datasetInstances.get(predicate).size() * 0.3))));
         validationData.addAll(datasetInstances.get(predicate).subList(((int) Math.round(datasetInstances.get(predicate).size() * 0.3)), ((int) Math.round(datasetInstances.get(predicate).size() * 0.4))));
         testingData.addAll(datasetInstances.get(predicate).subList(((int) Math.round(datasetInstances.get(predicate).size() * 0.4)), datasetInstances.get(predicate).size()));
         }*/
        //WEN DATA SPLIT
        //PRINT RESULTS
        ArrayList<String> mrs = new ArrayList<String>();
        try (BufferedReader br = new BufferedReader(new FileReader(wenFile))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    mrs.add(s.trim());
                }
            }

        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        ArrayList<String> testWenMRs = new ArrayList<>();
        wenDaToGen = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(wenFile))) {
            String s;
            boolean inGen = false;
            boolean inRef = false;
            String da = "";
            while ((s = br.readLine()) != null) {
                if (s.startsWith("DA")) {
                    inGen = false;
                    inRef = false;
                    da = s.substring(s.indexOf(":") + 1).replaceAll(",", ";").replaceAll("no or yes", "yes or no").replaceAll("ave ; presidio", "ave and presidio").replaceAll("point ; ste", "point and ste").trim();
                    testWenMRs.add(da);
                } else if (s.startsWith("Gen")) {
                    inGen = true;
                } else if (s.startsWith("Ref")) {
                    inRef = true;
                } else if (inGen) {
                    inGen = false;

                    if (!wenDaToGen.containsKey(da)) {
                        wenDaToGen.put(da.toLowerCase(), s.trim());
                    }
                    da = "";
                } else if (inRef) {
                    if (s.trim().isEmpty()) {
                        inRef = false;
                        da = "";
                    }
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }

        ArrayList<DatasetInstance> trainingData = new ArrayList<>();
        ArrayList<DatasetInstance> validationData = new ArrayList<>();
        ArrayList<DatasetInstance> testingData = new ArrayList<>();

        System.out.println(testWenMRs.size());
        ArrayList<DatasetInstance> restData = new ArrayList<>();
        for (String predicate : predicates) {
            for (DatasetInstance in : datasetInstances.get(predicate)) {
                if (testWenMRs.contains(in.getMeaningRepresentation().getMRstr())) {
                    testingData.add(in);
                    testWenMRs.remove(in.getMeaningRepresentation().getMRstr());
                } else {
                    restData.add(in);
                }
            }
        }
        Collections.shuffle(restData);
        for (int i = 0; i < restData.size(); i++) {
            if (i < testingData.size()) {
                validationData.add(restData.get(i));
            } else {
                trainingData.add(restData.get(i));
            }
        }
        System.out.println(trainingData.size());
        System.out.println(validationData.size());
        System.out.println(testingData.size());
        System.out.println(testWenMRs);

        r = new Random(seed);

        HashMap<String, JAROW> classifiersAttrs = new HashMap<>();
        HashMap<String, HashMap<String, JAROW>> classifiersWords = new HashMap<>();

        HashMap<Integer, HashSet<String>> nGrams;
        if (!useRandomAlignments) {
            nGrams = createNaiveAlignments(trainingData);
        } else {
            nGrams = createRandomAlignments(trainingData);
        }

        HashMap<String, HashSet<String>> availableAttributeActions = new HashMap<>();
        HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions = new HashMap<>();
        for (DatasetInstance DI : trainingData) {
            String predicate = DI.getMeaningRepresentation().getPredicate();
            if (!availableAttributeActions.containsKey(predicate)) {
                availableAttributeActions.put(predicate, new HashSet<String>());
            }
            if (!availableWordActions.containsKey(predicate)) {
                availableWordActions.put(predicate, new HashMap<String, HashSet<Action>>());
            }

            ArrayList<Action> realization = DI.getTrainRealization();
            for (Action a : realization) {
                if (!a.getAttribute().equals(SFX.TOKEN_END)) {
                    String attr = "";
                    if (a.getAttribute().contains("=")) {
                        attr = a.getAttribute().substring(0, a.getAttribute().indexOf('='));
                    } else {
                        attr = a.getAttribute();
                    }
                    availableAttributeActions.get(predicate).add(attr);
                    if (!availableWordActions.get(predicate).containsKey(attr)) {
                        availableWordActions.get(predicate).put(attr, new HashSet<Action>());
                        availableWordActions.get(predicate).get(attr).add(new Action(SFX.TOKEN_END, attr));
                    }
                    if (!a.getWord().equals(SFX.TOKEN_START)
                            && !a.getWord().equals(SFX.TOKEN_END)
                            && !a.getWord().matches("([,.?!;:'])")) {
                        if (a.getWord().startsWith(SFX.TOKEN_X)) {
                            if (a.getWord().substring(3, a.getWord().lastIndexOf('_')).toLowerCase().trim().equals(attr)) {
                                availableWordActions.get(predicate).get(attr).add(new Action(a.getWord(), attr));
                            }
                        } else {
                            availableWordActions.get(predicate).get(attr).add(new Action(a.getWord(), attr));
                        }
                    }
                    if (attr.equals("[]")) {
                        System.out.println("RR " + realization);
                        System.out.println("RR " + a);
                    }
                }
            }
        }
        //ONLY WHEN RANDOM ALIGNMENTS
        if (useRandomAlignments) {
            valueAlignments = new HashMap<>();
        }

        Object[] results = createTrainingDatasets(trainingData, availableAttributeActions, availableWordActions, nGrams);
        HashMap<String, ArrayList<Instance>> predicateAttrTrainingData = (HashMap<String, ArrayList<Instance>>) results[0];
        HashMap<String, HashMap<String, ArrayList<Instance>>> predicateWordTrainingData = (HashMap<String, HashMap<String, ArrayList<Instance>>>) results[1];

        boolean setToGo = true;
        if (predicateWordTrainingData.isEmpty() || predicateAttrTrainingData.isEmpty()) {
            setToGo = false;
        }
        if (setToGo) {
            JDAggerForSFX JDWords = new JDAggerForSFX(this);
            Object[] LOLSResults = null;
            if (useValidation) {
                LOLSResults = JDWords.runLOLS(availableAttributeActions, trainingData, predicateAttrTrainingData, predicateWordTrainingData, availableWordActions, valueAlignments, wordRefRolloutChance, validationData, nGrams);
            } else {
                LOLSResults = JDWords.runLOLS(availableAttributeActions, trainingData, predicateAttrTrainingData, predicateWordTrainingData, availableWordActions, valueAlignments, wordRefRolloutChance, testingData, nGrams);
            }

            classifiersAttrs = (HashMap<String, JAROW>) LOLSResults[0];
            classifiersWords = (HashMap<String, HashMap<String, JAROW>>) LOLSResults[1];

            if (useValidation) {
                evaluateGeneration(classifiersAttrs, classifiersWords, trainingData, validationData, availableAttributeActions, availableWordActions, nGrams, true, 100000);
            } else {
                evaluateGeneration(classifiersAttrs, classifiersWords, trainingData, testingData, availableAttributeActions, availableWordActions, nGrams, true, 100000);
            }
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
    }

    public JAROW train(String predicate, ArrayList<Instance> trainingData) {
        //NO DAGGER USE
        JAROW classifier = new JAROW();
        Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
        classifier.train(trainingData, false, false, rounds, 100.0, true);

        return classifier;
    }

    public Double evaluateGeneration(HashMap<String, JAROW> classifierAttrs, HashMap<String, HashMap<String, JAROW>> classifierWords, ArrayList<DatasetInstance> trainingData, ArrayList<DatasetInstance> testingData, HashMap<String, HashSet<String>> availableAttributeActions, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams) {
        return evaluateGeneration(classifierAttrs, classifierWords, trainingData, testingData, availableAttributeActions, availableWordActions, nGrams, false, 0);
    }
    double previousBLEU = 0.0;
    ArrayList<ArrayList<Action>> previousResults = null;

    public Double evaluateGeneration(HashMap<String, JAROW> classifierAttrs, HashMap<String, HashMap<String, JAROW>> classifierWords, ArrayList<DatasetInstance> trainingData, ArrayList<DatasetInstance> testingData, HashMap<String, HashSet<String>> availableAttributeActions, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams, boolean printResults, int epoch) {
        System.out.println("Evaluate argument generation ");

        int totalArgDistance = 0;
        ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
        ArrayList<ArrayList<Action>> generationActions = new ArrayList<>();
        HashMap<ArrayList<Action>, DatasetInstance> generationActionsMap = new HashMap<>();
        ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
        ArrayList<String> predictedStrings = new ArrayList<>();
        ArrayList<String> predictedStringMRs = new ArrayList<>();
        ArrayList<Double> attrCoverage = new ArrayList<>();
        ArrayList<ArrayList<String>> predictedAttrLists = new ArrayList<>();
        HashSet<HashMap<String, HashSet<String>>> mentionedAttrs = new HashSet<HashMap<String, HashSet<String>>>();
        for (DatasetInstance di : testingData) {
            String predicate = di.getMeaningRepresentation().getPredicate();
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
                for (String value : di.getMeaningRepresentation().getAttributes().get(attribute)) {
                    attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
                }
                valuesToBeMentioned.put(attribute, new ArrayList<>(di.getMeaningRepresentation().getAttributes().get(attribute)));
            }
            if (attrValuesToBeMentioned.isEmpty()) {
                attrValuesToBeMentioned.add("empty=empty");
            }
            HashSet<String> attrValuesToBeMentionedCopy = new HashSet<>(attrValuesToBeMentioned);
            while (!predictedAttr.equals(SFX.TOKEN_END) && predictedAttrValues.size() < maxAttrRealizationSize) {
                if (!predictedAttr.isEmpty()) {
                    attrValuesToBeMentioned.remove(predictedAttr);
                }
                Instance attrTrainingVector = SFX.this.createAttrInstance(predicate, "@TOK@", predictedAttrValues, predictedActionList, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableAttributeActions);

                if (attrTrainingVector != null) {
                    Prediction predictAttr = classifierAttrs.get(predicate).predict(attrTrainingVector);
                    if (predictAttr.getLabel() != null) {
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
                        }
                        if (!predictedAttr.equals(SFX.TOKEN_END)) {
                            predictedAttr += "=" + predictedValue;
                        }
                        predictedAttrValues.add(predictedAttr);

                        String attribute = predictedAttrValues.get(predictedAttrValues.size() - 1).split("=")[0];
                        String attrValue = predictedAttrValues.get(predictedAttrValues.size() - 1);
                        predictedAttributes.add(attrValue);

                        //GENERATE PHRASES
                        if (!attribute.equals(SFX.TOKEN_END)) {
                            if (classifierWords.get(predicate).containsKey(attribute)) {
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
                                    Instance wordTrainingVector = createWordInstance(predicate, new Action("@TOK@", attrValue), predictedAttributesForInstance, predictedActionList, isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions.get(predicate), nGrams, false);

                                    if (wordTrainingVector != null) {
                                        if (classifierWords.get(predicate) != null) {
                                            if (classifierWords.get(predicate).get(attribute) != null) {
                                                Prediction predictWord = classifierWords.get(predicate).get(attribute).predict(wordTrainingVector);
                                                if (predictWord.getLabel() != null) {
                                                    predictedWord = predictWord.getLabel().trim();
                                                    predictedActionList.add(new Action(predictedWord, attrValue));
                                                    if (!predictedWord.equals(SFX.TOKEN_END)) {
                                                        subPhrase.add(predictedWord);
                                                        predictedWordList.add(new Action(predictedWord, attrValue));
                                                    }
                                                } else {
                                                    predictedWord = SFX.TOKEN_END;
                                                    predictedActionList.add(new Action(predictedWord, attrValue));
                                                }
                                            } else {
                                                predictedWord = SFX.TOKEN_END;
                                                predictedActionList.add(new Action(predictedWord, attrValue));
                                            }
                                        }
                                    }
                                    if (!isValueMentioned) {
                                        if (!predictedWord.equals(SFX.TOKEN_END)) {
                                            if (predictedWord.startsWith(SFX.TOKEN_X)
                                                    && (valueTBM.matches("\"[xX][0-9]+\"")
                                                    || valueTBM.matches("[xX][0-9]+")
                                                    || valueTBM.startsWith(SFX.TOKEN_X))) {
                                                isValueMentioned = true;
                                            } else if (!predictedWord.startsWith(SFX.TOKEN_X)
                                                    && !(valueTBM.matches("\"[xX][0-9]+\"")
                                                    || valueTBM.matches("[xX][0-9]+")
                                                    || valueTBM.startsWith(SFX.TOKEN_X))) {
                                                String valueToCheck = valueTBM;
                                                if (valueToCheck.equals("no")
                                                        || valueToCheck.equals("yes")
                                                        || valueToCheck.equals("yes or no")
                                                        || valueToCheck.equals("none")
                                                        || valueToCheck.equals("dont_care")
                                                        || valueToCheck.equals("empty")) {
                                                    if (attribute.contains("=")) {
                                                        valueToCheck = attribute.replace("=", ":");
                                                    } else {
                                                        valueToCheck = attribute + ":" + valueTBM;
                                                    }
                                                }
                                                if (!valueToCheck.equals("empty:empty")
                                                        && valueAlignments.containsKey(valueToCheck)) {
                                                    for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                                        if (endsWith(subPhrase, alignedStr)) {
                                                            isValueMentioned = true;
                                                            break;
                                                        }
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
                                    if (!predictedWord.startsWith(SFX.TOKEN_X)) {
                                        for (String attrValueTBM : attrValuesToBeMentioned) {
                                            if (attrValueTBM.contains("=")) {
                                                String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                                if (!(value.matches("\"[xX][0-9]+\"")
                                                        || value.matches("[xX][0-9]+")
                                                        || value.startsWith(SFX.TOKEN_X))) {
                                                    String valueToCheck = value;
                                                    if (valueToCheck.equals("no")
                                                            || valueToCheck.equals("yes")
                                                            || valueToCheck.equals("yes or no")
                                                            || valueToCheck.equals("none")
                                                            || valueToCheck.equals("dont_care")
                                                            || valueToCheck.equals("empty")) {
                                                        valueToCheck = attrValueTBM.replace("=", ":");
                                                    }
                                                    if (!valueToCheck.equals("empty:empty")
                                                            && valueAlignments.containsKey(valueToCheck)) {
                                                        for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                                            if (endsWith(subPhrase, alignedStr)) {
                                                                mentionedAttrValue = attrValueTBM;
                                                                break;
                                                            }
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
                                        && !predictedActionList.get(predictedActionList.size() - 1).getWord().equals(SFX.TOKEN_END)) {
                                    predictedWord = SFX.TOKEN_END;
                                    predictedActionList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                }
                            } else {
                                String predictedWord = SFX.TOKEN_END;
                                predictedActionList.add(new Action(predictedWord, attrValue));
                            }
                        }
                    } else {
                        predictedAttr = SFX.TOKEN_END;
                    }
                }
            }
            ArrayList<String> predictedAttrs = new ArrayList<>();
            for (String attributeValuePair : predictedAttrValues) {
                predictedAttrs.add(attributeValuePair.split("=")[0]);
            }

            ArrayList<Action> cleanActionList = new ArrayList<Action>();
            for (Action action : predictedActionList) {
                if (!action.getWord().equals(SFX.TOKEN_END)
                        && !action.getWord().equals(SFX.TOKEN_START)) {
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
                        cleanActionList.add(i + 2, punctPatterns.get(surrounds));
                    }
                }
            }

            String predictedString = "";
            ArrayList<String> predictedAttrList = new ArrayList<String>();
            HashSet<String> redundants = new HashSet<String>();
            for (Action action : cleanActionList) {
                if (action.getWord().startsWith(SFX.TOKEN_X)) {
                    predictedString += di.getMeaningRepresentation().getDelexMap().get(action.getWord()) + " ";
                    //predictedString += "x ";
                    if (di.getMeaningRepresentation().getDelexMap().get(action.getWord()) == null
                            || di.getMeaningRepresentation().getDelexMap().get(action.getWord()).equals("null")) {
                        redundants.add(action.getWord());
                    }
                } else {
                    predictedString += action.getWord() + " ";
                }
                if (predictedAttrList.isEmpty()) {
                    predictedAttrList.add(action.getAttribute());
                } else if (!predictedAttrList.get(predictedAttrList.size() - 1).equals(action.getAttribute())) {
                    predictedAttrList.add(action.getAttribute());
                }
            }
            predictedAttrLists.add(predictedAttrList);
            if (attrValuesToBeMentionedCopy.size() != 0.0) {
                double redundAttrs = 0.0;
                double missingAttrs = 0.0;
                for (String attr : predictedAttrList) {
                    if (!attrValuesToBeMentionedCopy.contains(attr)) {
                        redundAttrs += 1.0;
                    }
                }
                for (String attr : attrValuesToBeMentionedCopy) {
                    if (!predictedAttrList.contains(attr)) {
                        missingAttrs += 1.0;
                    }
                }
                double attrSize = (double) attrValuesToBeMentionedCopy.size();
                attrCoverage.add((redundAttrs + missingAttrs) / attrSize);
            }

            if (predicate.startsWith("?")) {
                predictedString = predictedString.trim() + "?";
            } else {
                predictedString = predictedString.trim() + ".";
            }
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
            generationActionsMap.put(predictedActionList, di);

            ArrayList<Sequence<IString>> references = new ArrayList<>();
            for (ArrayList<Action> realization : di.getEvalRealizations()) {
                String cleanedWords = "";
                for (Action nlWord : realization) {
                    if (!nlWord.equals(new Action(SFX.TOKEN_START, ""))
                            && !nlWord.equals(new Action(SFX.TOKEN_END, ""))) {
                        if (nlWord.getWord().startsWith(SFX.TOKEN_X)) {
                            cleanedWords += di.getMeaningRepresentation().getDelexMap().get(nlWord.getWord()) + " ";
                        } else {
                            cleanedWords += nlWord.getWord() + " ";
                        }
                    }
                }
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

            int minTotArgDistance = Integer.MAX_VALUE;
            for (ArrayList<String> goldArgs : goldAttributeSequences) {
                int totArgDistance = 0;
                HashSet<Integer> matchedPositions = new HashSet<>();
                for (int i = 0; i < predictedAttrs.size(); i++) {
                    if (!predictedAttrs.get(i).equals(SFX.TOKEN_START)
                            && !predictedAttrs.get(i).equals(SFX.TOKEN_END)) {
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
                ArrayList<String> predictedCopy = (ArrayList<String>) predictedAttrs.clone();
                for (String goldArg : goldArgs) {
                    if (!goldArg.equals(SFX.TOKEN_END)) {
                        boolean contained = predictedCopy.remove(goldArg);
                        if (!contained) {
                            totArgDistance += 1000;
                        }
                    }
                }
                if (totArgDistance < minTotArgDistance) {
                    minTotArgDistance = totArgDistance;
                }
            }
            totalArgDistance += minTotArgDistance;
        }

        previousResults = generationActions;

        crossAvgArgDistances.add(totalArgDistance / (double) testingData.size());

        NISTMetric NIST = new NISTMetric(finalReferences);
        BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
        BLEUMetric BLEUsmooth = new BLEUMetric(finalReferences, 4, true);
        Double nistScore = NIST.score(generations);
        Double bleuScore = BLEU.score(generations);
        Double bleuSmoothScore = BLEUsmooth.score(generations);

        double finalCoverage = 0.0;
        for (double c : attrCoverage) {
            finalCoverage += c;
        }
        finalCoverage /= (double) attrCoverage.size();
        crossNIST.add(nistScore);
        crossBLEU.add(bleuScore);
        crossBLEUSmooth.add(bleuSmoothScore);
        System.out.println("Avg arg distance: \t" + totalArgDistance / (double) testingData.size());
        System.out.println("NIST: \t" + nistScore);
        System.out.println("BLEU: \t" + bleuScore);
        System.out.println("COVERAGE: \t" + finalCoverage);
        System.out.println("g: " + generations);
        System.out.println("attr: " + predictedAttrLists);
        System.out.println("BLEU smooth: \t" + bleuSmoothScore);
        previousBLEU = bleuScore;

        if (printResults) {
            BufferedWriter bw = null;
            File f = null;
            try {
                f = new File("random_SFX" + dataset + "TextsAfter" + (epoch) + "_" + JDAggerForSFX.earlyStopMaxFurtherSteps + "_" + JDAggerForSFX.p + "epochsTESTINGDATA.txt");
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
                    //SFX HOTEL TEXTS WITH LOLS -> 3
                    //SFX RESTAURANT TEXTS WITH LOLS -> 5
                    bw.write("MR;" + predictedStringMRs.get(i).replaceAll(";", ",") + ";");
                    if (dataset.equals("hotel")) {
                        bw.write("LOLS_SFHOT;");
                    } else {
                        bw.write("LOLS_SFRES;");
                    }
                    //bw.write("@@srcdoc@@" + (i + 1));
                    /*String out = predictedStrings.get(i).replaceAll(" i ", " I ").replaceAll(" -ly ", "ly ").replaceAll(" s ", "s ").replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ");
                    out = out.substring(0, 1).toUpperCase() + out.substring(1);
                    bw.write(out + ";");
                    if (dataset.equals("hotel")) {
                        bw.write("WEN_SFHOT;");
                    } else {
                        bw.write("WEN_SFRES;");
                    }
                    if (!wenDaToGen.containsKey(predictedStringMRs.get(i).trim().toLowerCase())) {
                        System.out.println(wenDaToGen.keySet());
                        System.out.println(predictedStringMRs.get(i).trim().toLowerCase());
                        System.exit(0);
                    }
                    out = wenDaToGen.get(predictedStringMRs.get(i).trim().toLowerCase()).replaceAll(" i ", " I ").replaceAll(" -ly ", "ly ").replaceAll(" s ", "s ").replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ");
                    out = out.substring(0, 1).toUpperCase() + out.substring(1);
                    bw.write(out + ";");*/
                    //bw.write("@@judgeFluency@@-1");
                    //bw.write("@@judgeInform@@-1");
                    //bw.write("@@judgeQuality@@-1");

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

    public void printBaselineData(File dataFile, ArrayList<DatasetInstance> testingData, String data) {
        try {
            String str = new String();
            boolean begin = false;
            try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {
                String s;
                while ((s = br.readLine()) != null) {
                    if (s.startsWith("[")) {
                        begin = true;
                    }
                    if (begin) {
                        str += s;
                    }
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            }

            JSONArray overArray = new JSONArray(str);
            ArrayList<String> predictedStrings = new ArrayList<String>();
            ArrayList<String> predictedStringMRs = new ArrayList<String>();
            HashSet<HashMap<String, HashSet<String>>> mentionedAttrs = new HashSet<HashMap<String, HashSet<String>>>();
            for (int o = 0; o < overArray.length(); o++) {
                JSONArray arr = overArray.getJSONObject(o).getJSONArray("dial");
                for (int i = 0; i < arr.length(); i++) {
                    for (DatasetInstance test : testingData) {
                        if (test.getMeaningRepresentation().getMRstr().equals(arr.getJSONObject(i).getJSONObject("S").getString("dact"))
                                && !mentionedAttrs.contains(test.getMeaningRepresentation().getAttributes())) {
                            predictedStrings.add(arr.getJSONObject(i).getJSONObject("S").getString("base").replaceAll(" -s", "s"));
                            predictedStringMRs.add(arr.getJSONObject(i).getJSONObject("S").getString("dact"));
                            mentionedAttrs.add(test.getMeaningRepresentation().getAttributes());
                        }
                    }
                }
            }
            BufferedWriter bw = null;
            File f = null;
            try {
                f = new File("SFX" + data + "TextsAt " + "BASELINE" + "" + ".txt");
            } catch (NullPointerException e) {
                System.err.println("File not found." + e);
            }

            try {
                bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
            } catch (FileNotFoundException e) {
                System.err.println("Error opening file for writing! " + e);
            }

            for (int i = 0; i < predictedStrings.size(); i++) {
                try {
                    //Grafoume to String sto arxeio
                    //SFX BASELINE TEXTS -> 4
                    bw.write("@@srcidx@@4");
                    bw.write("@@srcdoc@@" + (i + 1));
                    bw.write("@@MR@@" + predictedStringMRs.get(i));
                    bw.write("@@src@@" + predictedStrings.get(i));
                    bw.write("@@judgeFluency@@-1");
                    bw.write("@@judgeInform@@-1");
                    bw.write("@@judgeQuality@@-1");

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
        } catch (JSONException ex) {
            ex.printStackTrace();
        }
    }

    public void createLists(File dataFile) {
        try {
            predicates = new ArrayList<>();
            attributes = new HashMap<>();
            attributeValuePairs = new HashMap<>();
            valueAlignments = new HashMap<>();

            String str = new String();
            boolean begin = false;
            try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {
                String s;
                while ((s = br.readLine()) != null) {
                    if (s.startsWith("[")) {
                        begin = true;
                    }
                    if (begin) {
                        str += s;
                    }
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            }

            JSONArray overArray = new JSONArray(str);

            for (int o = 0; o < overArray.length(); o++) {
                JSONArray arr = overArray.getJSONObject(o).getJSONArray("dial");
                for (int i = 0; i < arr.length(); i++) {
                    String MRstr = "";
                    String ref = "";
                    MRstr = arr.getJSONObject(i).getJSONObject("S").getString("dact");
                    ref = arr.getJSONObject(i).getJSONObject("S").getString("ref").replaceAll("-s", "s");

                    if ((MRstr.startsWith("inform(")
                            || MRstr.startsWith("inform_only")
                            || MRstr.startsWith("inform_no_match(")
                            || MRstr.startsWith("?confirm(")
                            || MRstr.startsWith("?select(")
                            || MRstr.startsWith("?request(")
                            || MRstr.startsWith("?reqmore(")
                            || MRstr.startsWith("goodbye("))
                            && !ref.isEmpty()) {
                        String predicate = MRstr.substring(0, MRstr.indexOf("("));
                        if (!predicates.contains(predicate) && predicate != null) {
                            predicates.add(predicate);

                            if (!attributes.containsKey(predicate)) {
                                attributes.put(predicate, new HashSet<String>());
                            }
                            if (!datasetInstances.containsKey(predicate)) {
                                datasetInstances.put(predicate, new ArrayList<DatasetInstance>());
                            }
                        }

                        String attributesStr = MRstr.substring(MRstr.indexOf('(') + 1, MRstr.length() - 1);
                        HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
                        if (!attributesStr.isEmpty()) {
                            HashMap<String, Integer> attrXIndeces = new HashMap<>();

                            String[] args = attributesStr.split(";");
                            if (attributesStr.contains("|")) {
                                System.out.println(attributesStr);
                                System.exit(0);
                            }
                            for (String arg : args) {
                                String attr = "";
                                String value = "";
                                if (arg.contains("=")) {
                                    String[] subAttr = arg.split("=");
                                    value = subAttr[1].toLowerCase();
                                    attr = subAttr[0].toLowerCase().replaceAll("_", "");
                                } else {
                                    attr = arg.replaceAll("_", "");
                                }
                                if (!attributes.get(predicate).contains(attr)) {
                                    attributes.get(predicate).add(attr);
                                }
                                if (!attributeValues.containsKey(attr)) {
                                    attributeValues.put(attr, new HashSet<String>());
                                }
                                if (value.isEmpty()) {
                                    value = attr;
                                }

                                if (value.startsWith("\'")) {
                                    value = value.substring(1, value.length() - 1);
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
                                if (value.isEmpty()) {
                                    System.out.println("EMPTY VALUE");
                                    System.exit(0);
                                }

                                attributeValues.get(attr).add(value.trim().toLowerCase());
                            }
                            for (String attr : attributeValues.keySet()) {
                                if (attributeValues.get(attr).contains("yes")
                                        && attributeValues.get(attr).contains("no")) {
                                    System.out.println(MRstr);
                                    System.out.println(attributeValues);
                                    System.exit(0);
                                }
                            }
                        }

                        //REF
                        //DELEXICALIZATION
                        HashMap<String, HashMap<String, Integer>> attrValuePriorities = new HashMap<>();
                        HashMap<String, HashSet<String>> delexAttributeValues = new HashMap<>();
                        int prio = 0;
                        for (String attr : attributeValues.keySet()) {
                            if (!attr.isEmpty()) {
                                delexAttributeValues.put(attr, new HashSet<String>());
                                if (attr.equals("name")
                                        || attr.equals("type")
                                        || attr.equals("pricerange")
                                        || attr.equals("price")
                                        || attr.equals("phone")
                                        || attr.equals("address")
                                        || attr.equals("postcode")
                                        || attr.equals("area")
                                        || attr.equals("near")
                                        || attr.equals("food")
                                        || attr.equals("count")
                                        || attr.equals("goodformeal")) {
                                    attrValuePriorities.put(attr, new HashMap<String, Integer>());
                                    for (String value : attributeValues.get(attr)) {
                                        if (!value.equals("dont_care")
                                                && !value.equals("none")
                                                && !value.equals("empty")
                                                && !value.equals(attr)) {
                                            attrValuePriorities.get(attr).put(value, prio);
                                            prio++;
                                        } else {
                                            delexAttributeValues.get(attr).add(value);
                                        }
                                    }
                                } else {
                                    for (String value : attributeValues.get(attr)) {
                                        delexAttributeValues.get(attr).add(value);
                                    }
                                }
                            }
                        }
                        boolean change = true;
                        while (change) {
                            change = false;
                            for (String attr1 : attrValuePriorities.keySet()) {
                                for (String value1 : attrValuePriorities.get(attr1).keySet()) {
                                    for (String attr2 : attrValuePriorities.keySet()) {
                                        for (String value2 : attrValuePriorities.get(attr2).keySet()) {
                                            if (!value1.equals(value2)
                                                    && value1.contains(value2)
                                                    && attrValuePriorities.get(attr1).get(value1) > attrValuePriorities.get(attr2).get(value2)) {
                                                int prio1 = attrValuePriorities.get(attr1).get(value1);
                                                int prio2 = attrValuePriorities.get(attr2).get(value2);
                                                attrValuePriorities.get(attr1).put(value1, prio2);
                                                attrValuePriorities.get(attr2).put(value2, prio1);
                                                change = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        HashMap<String, Integer> xCounts = new HashMap<>();
                        HashMap<String, String> delexMap = new HashMap<>();
                        ref = " " + ref + " ";
                        for (int p = 0; p < prio; p++) {
                            for (String attr : attrValuePriorities.keySet()) {
                                if (!xCounts.containsKey(attr)) {
                                    xCounts.put(attr, 0);
                                }
                                for (String value : attrValuePriorities.get(attr).keySet()) {
                                    if (attrValuePriorities.get(attr).get(value) == p) {
                                        if (!ref.contains(" " + value + " ")
                                                && !value.contains(" and ")
                                                && !value.contains(" or ")) {
                                            /*System.out.println(ref);
                                             System.out.println(attr);
                                             System.out.println(value);
                                             System.out.println(attrValuePriorities);*/
                                        } else if (!ref.contains(" " + value + " ")
                                                && (value.contains(" and ")
                                                || value.contains(" or "))) {
                                            String[] values = null;
                                            if (value.contains(" and ")) {
                                                values = value.split(" and ");
                                            } else if (value.contains(" or ")) {
                                                values = value.split(" or ");
                                            }
                                            for (int v = 0; v < values.length; v++) {
                                                if (!ref.contains(" " + values[v] + " ")) {
                                                    /*System.out.println(ref);
                                                     System.out.println(attr);
                                                     System.out.println(value);
                                                     System.out.println(values[v]);
                                                     System.out.println(attrValuePriorities);*/
                                                } else {
                                                    ref = ref.replace(" " + values[v] + " ", " " + SFX.TOKEN_X + attr + "_" + xCounts.get(attr) + " ");
                                                    ref = ref.replaceAll("  ", " ");
                                                    delexAttributeValues.get(attr).add(SFX.TOKEN_X + attr + "_" + xCounts.get(attr));
                                                    delexMap.put(SFX.TOKEN_X + attr + "_" + xCounts.get(attr), values[v]);
                                                    xCounts.put(attr, xCounts.get(attr) + 1);
                                                }
                                            }
                                        } else {
                                            ref = ref.replace(" " + value + " ", " " + SFX.TOKEN_X + attr + "_" + xCounts.get(attr) + " ");
                                            ref = ref.replaceAll("  ", " ");
                                            delexAttributeValues.get(attr).add(SFX.TOKEN_X + attr + "_" + xCounts.get(attr));
                                            delexMap.put(SFX.TOKEN_X + attr + "_" + xCounts.get(attr), value);
                                            xCounts.put(attr, xCounts.get(attr) + 1);
                                        }
                                    }
                                }
                            }
                        }
                        ref = ref.trim();
                        MeaningRepresentation MR = new MeaningRepresentation(predicate, delexAttributeValues, MRstr);
                        MR.setDelexMap(delexMap);

                        ArrayList<String> mentionedValueSequence = new ArrayList<>();
                        ArrayList<String> mentionedAttributeSequence = new ArrayList<>();

                        ArrayList<String> realization = new ArrayList<>();
                        ArrayList<String> alignedRealization = new ArrayList<>();

                        String[] words = ref.replaceAll("([,.?!;:'])", " $1").split(" ");
                        HashMap<String, Integer> attributeXCount = new HashMap<>();
                        for (int w = 0; w < words.length; w++) {
                            String mentionedAttribute = "";
                            if (!words[w].trim().isEmpty()) {
                                int s = words[w].indexOf("[");
                                if (s != -1) {
                                    int e = words[w].indexOf("]", s + 1);

                                    String mentionedValue = words[w].substring(s, e + 1);
                                    words[w] = words[w].replace(mentionedValue, "");
                                    if (mentionedValue.contains("+") && !words[w].trim().isEmpty()) {
                                        mentionedAttribute = mentionedValue.substring(1, mentionedValue.indexOf("+"));

                                        if (MR.getAttributes().containsKey(mentionedAttribute)) {
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
                                    } else if (!words[w].trim().isEmpty()) {
                                        mentionedAttribute = mentionedValue.substring(1, mentionedValue.length() - 1);

                                        if (!MR.getAttributes().containsKey(mentionedAttribute)) {
                                            mentionedAttribute = "";
                                        }
                                    }
                                }
                                if (!words[w].trim().isEmpty()) {
                                    if (words[w].trim().equals("thers")) {
                                        realization.add("there");
                                    } else {
                                        realization.add(words[w].trim().toLowerCase());
                                    }
                                }
                            }
                        }

                        for (String attr : MR.getAttributes().keySet()) {
                            for (String value : MR.getAttributes().get(attr)) {
                                if (attr.equals("name") && value.equals("none")) {
                                    mentionedValueSequence.add(0, attr.toLowerCase() + "=" + value.toLowerCase());
                                    mentionedAttributeSequence.add(0, attr.toLowerCase());
                                }
                            }
                        }

                        mentionedValueSequence.add(SFX.TOKEN_END);
                        mentionedAttributeSequence.add(SFX.TOKEN_END);
                        if (realization.size() > maxWordRealizationSize) {
                            maxWordRealizationSize = realization.size();
                        }

                        for (String word : realization) {
                            if (word.trim().matches("[,.?!;:']")) {
                                alignedRealization.add(SFX.TOKEN_PUNCT);
                            } else {
                                alignedRealization.add("[]");
                            }
                        }

                        //Calculate alignments
                        HashMap<String, HashMap<String, Double>> alignments = new HashMap<>();
                        for (String attr : MR.getAttributes().keySet()) {
                            /*if (attr.equals("name")
                             || attr.equals("type")
                             || attr.equals("pricerange")
                             || attr.equals("price")
                             || attr.equals("phone")
                             || attr.equals("address")
                             || attr.equals("postcode")
                             || attr.equals("area")
                             || attr.equals("near")
                             || attr.equals("food")
                             || attr.equals("count")
                             || attr.equals("goodformeal")) {*/

                            for (String value : MR.getAttributes().get(attr)) {
                                if (!value.equals("name=none")
                                        && !value.startsWith(SFX.TOKEN_X)
                                        && !(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+"))) {
                                    String valueToCheck = value;
                                    if (value.equals("no")
                                            || value.equals("yes")
                                            || value.equals("yes or no")
                                            || value.equals("none")
                                            || value.equals("dont_care")
                                            || value.equals("empty")) {
                                        valueToCheck = attr;
                                        alignments.put(valueToCheck + ":" + value, new HashMap<String, Double>());
                                    } else {
                                        alignments.put(valueToCheck, new HashMap<String, Double>());
                                    }
                                    //For all ngrams
                                    for (int n = 1; n < realization.size(); n++) {
                                        //Calculate all alignment similarities
                                        for (int r = 0; r <= realization.size() - n; r++) {
                                            boolean pass = true;
                                            for (int j = 0; j < n; j++) {
                                                if (realization.get(r + j).startsWith(SFX.TOKEN_X)
                                                        || alignedRealization.get(r + j).equals(SFX.TOKEN_PUNCT)
                                                        || StringNLPUtilities.isArticle(realization.get(r + j))
                                                        || realization.get(r + j).equalsIgnoreCase("and")
                                                        || realization.get(r + j).equalsIgnoreCase("or")) {
                                                    pass = false;
                                                }
                                            }
                                            if (pass) {
                                                String align = "";
                                                String compare = "";
                                                String backwardCompare = "";
                                                for (int j = 0; j < n; j++) {
                                                    align += (r + j) + " ";
                                                    compare += realization.get(r + j);
                                                    backwardCompare = realization.get(r + j) + backwardCompare;
                                                }
                                                align = align.trim();

                                                Double distance = Levenshtein.getSimilarity(valueToCheck.toLowerCase(), compare.toLowerCase(), true, false);
                                                Double backwardDistance = Levenshtein.getSimilarity(valueToCheck.toLowerCase(), backwardCompare.toLowerCase(), true, false);

                                                if (backwardDistance > distance) {
                                                    distance = backwardDistance;
                                                }
                                                if (distance > 0.3) {
                                                    if (value.equals("no")
                                                            || value.equals("yes")
                                                            || value.equals("yes or no")
                                                            || value.equals("none")
                                                            || value.equals("dont_care")
                                                            || value.equals("empty")) {
                                                        alignments.get(valueToCheck + ":" + value).put(align, distance);
                                                    } else {
                                                        alignments.get(valueToCheck).put(align, distance);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            /*} else {
                             for (String value : MR.getAttributes().get(attr)) {
                             if (!value.equals("no")
                             && !value.equals("yes")
                             && !value.equals("none")) {
                             System.out.println(attr + " " + value);
                             }
                             }
                             }*/
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
                                for (int a = Integer.parseInt(coords[0].trim()); a <= Integer.parseInt(coords[coords.length - 1].trim()); a++) {
                                    alignedStr.add(realization.get(a));
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
                        for (int a = alignedRealization.size() - 1; a >= 0; a--) {
                            if (alignedRealization.get(a).isEmpty() || alignedRealization.get(a).equals("[]")) {
                                if (!previousAttr.isEmpty()) {
                                    alignedRealization.set(a, previousAttr);
                                }
                            } else if (!alignedRealization.get(a).equals(SFX.TOKEN_PUNCT)) {
                                previousAttr = alignedRealization.get(a);
                            } else {
                                previousAttr = "";
                            }
                        }
                        previousAttr = "";
                        for (int a = 0; a < alignedRealization.size(); a++) {
                            if (alignedRealization.get(a).isEmpty() || alignedRealization.get(a).equals("[]")) {
                                if (!previousAttr.isEmpty()) {
                                    alignedRealization.set(a, previousAttr);
                                }
                            } else if (!alignedRealization.get(a).equals(SFX.TOKEN_PUNCT)) {
                                previousAttr = alignedRealization.get(a);
                            } else {
                                previousAttr = "";
                            }
                        }

                        //}
                        ArrayList<Action> realizationActions = new ArrayList<>();
                        for (int r = 0; r < realization.size(); r++) {
                            realizationActions.add(new Action(realization.get(r), alignedRealization.get(r)));
                        }

                        //boolean existing = false;
                        DatasetInstance DI = new DatasetInstance(MR, mentionedValueSequence, mentionedAttributeSequence, realizationActions);
                        for (DatasetInstance existingDI : datasetInstances.get(predicate)) {
                            //if (existingDI.getMeaningRepresentation().equals(previousAMR)) {
                            //if (existingDI.getMeaningRepresentation().getAttributes().equals(MR.getAttributes())) {
                            if (existingDI.getMeaningRepresentation().getAttributes().equals(DI.getMeaningRepresentation().getAttributes())) {
                                //existing = true;
                                //existingDI.mergeDatasetInstance(mentionedValueSequence, mentionedAttributeSequence, realizationActions);
                                existingDI.mergeDatasetInstance(DI.getEvalMentionedValueSequences(), DI.getEvalMentionedAttributeSequences(), DI.getEvalRealizations());
                                DI.mergeDatasetInstance(existingDI.getEvalMentionedValueSequences(), existingDI.getEvalMentionedAttributeSequences(), existingDI.getEvalRealizations());
                            }
                        }
                        //if (!existing) {
                        //DatasetInstance DI = new DatasetInstance(MR, mentionedValueSequence, mentionedAttributeSequence, realizationActions);
                        datasetInstances.get(predicate).add(DI);
                        //}
                    }
                    //}
                }
            }
            /*int dis = 0;
             int slots = 0;
             for (String pred : datasetInstances.keySet()) {
             System.out.println("FOR PREDICATE: " + pred);
             System.out.println("=====");
             for (DatasetInstance di : datasetInstances.get(pred)) {
             System.out.println(di.getMeaningRepresentation().getPredicate());
             System.out.println(di.getMeaningRepresentation().getAttributes());

             System.out.println("***********");
             System.out.println("R: " + di.getEvalRealizations());
             System.out.println("=======================");
             dis++;
             slots += di.getMeaningRepresentation().getAttributes().size();
             }
             }
             System.out.println(dis);
             System.out.println((double) slots / (double) dis);*/
        } catch (JSONException ex) {
            ex.printStackTrace();
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

    public Object[] createTrainingDatasets(ArrayList<DatasetInstance> trainingData, HashMap<String, HashSet<String>> availableAttributeActions, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams) {
        HashMap<String, ArrayList<Instance>> predicateAttrTrainingData = new HashMap<>();
        HashMap<String, HashMap<String, ArrayList<Instance>>> predicateWordTrainingData = new HashMap<>();

        if (!availableWordActions.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*/) {
            for (String predicate : predicates) {
                predicateAttrTrainingData.put(predicate, new ArrayList<Instance>());
                predicateWordTrainingData.put(predicate, new HashMap<String, ArrayList<Instance>>());
            }

            for (DatasetInstance di : trainingData) {
                //System.out.println("BEGIN");
                String predicate = di.getMeaningRepresentation().getPredicate();
                //for (ArrayList<Action> realization : di.getEvalRealizations()) {
                ArrayList<Action> realization = di.getTrainRealization();
                //System.out.println(realization);
                HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
                HashSet<String> attrValuesToBeMentioned = new HashSet<>();
                for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                    //int a = 0;
                    for (String value : di.getMeaningRepresentation().getAttributes().get(attribute)) {
                        /*if (value.startsWith("\"x")) {
                         value = "x" + a;
                         a++;
                         } else if (value.startsWith("\"")) {
                         value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                         }*/
                        attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
                    }
                }
                if (attrValuesToBeMentioned.isEmpty()) {
                    attrValuesToBeMentioned.add("empty=empty");
                }

                ArrayList<String> attrs = new ArrayList<>();
                boolean isValueMentioned = false;
                String valueTBM = "";
                String attrValue = "";
                ArrayList<String> subPhrase = new ArrayList<>();
                for (int w = 0; w < realization.size(); w++) {
                    if (!realization.get(w).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                        if (!realization.get(w).getAttribute().equals(attrValue)) {
                            if (!attrValue.isEmpty()) {
                                attrValuesToBeMentioned.remove(attrValue);
                            }
                            Instance attrTrainingVector = SFX.this.createAttrInstance(predicate, realization.get(w).getAttribute(), attrs, new ArrayList<Action>(realization.subList(0, w)), attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableAttributeActions);
                            if (attrTrainingVector != null) {
                                /*System.out.println(realization.get(w).getAttribute() + " >>>> " + attrTrainingVector.getCorrectLabels());
                                 for (String f : attrTrainingVector.getGeneralFeatureVector().keySet()) {
                                 if (f.startsWith("feature_attrValue_5gram_")
                                 || f.startsWith("feature_word_5gram_")) {
                                 System.out.println(">> " + f);
                                 }
                                 }*/
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
                        if (!attrValue.equals(SFX.TOKEN_END)) {
                            ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                            for (int i = 0; i < attrs.size() - 1; i++) {
                                predictedAttributesForInstance.add(attrs.get(i));
                            }
                            if (!attrs.get(attrs.size() - 1).equals(attrValue)) {
                                predictedAttributesForInstance.add(attrs.get(attrs.size() - 1));
                            }
                            Instance wordTrainingVector = createWordInstance(predicate, realization.get(w), predictedAttributesForInstance, new ArrayList<Action>(realization.subList(0, w)), isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions.get(predicate), nGrams, false);

                            if (wordTrainingVector != null) {
                                String attribute = attrValue;
                                if (attribute.contains("=")) {
                                    attribute = attrValue.substring(0, attrValue.indexOf('='));
                                }
                                if (!predicateWordTrainingData.get(predicate).containsKey(attribute)) {
                                    predicateWordTrainingData.get(predicate).put(attribute, new ArrayList<Instance>());
                                }
                                /*System.out.println(realization.get(w) + " >>>> " + wordTrainingVector.getCorrectLabels());
                                 for (String f : wordTrainingVector.getGeneralFeatureVector().keySet()) {
                                 if (f.startsWith("feature_attrValue_5gram_")
                                 || f.startsWith("feature_word_5gram_")) {
                                 System.out.println(">> " + f);
                                 }
                                 }*/
                                predicateWordTrainingData.get(predicate).get(attribute).add(wordTrainingVector);
                                if (!realization.get(w).getWord().equals(SFX.TOKEN_START)
                                        && !realization.get(w).getWord().equals(SFX.TOKEN_END)) {
                                    subPhrase.add(realization.get(w).getWord());
                                }
                            }
                            if (!isValueMentioned) {
                                if (realization.get(w).getWord().startsWith(SFX.TOKEN_X)
                                        && (valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\"")
                                        || valueTBM.startsWith(SFX.TOKEN_X))) {
                                    isValueMentioned = true;
                                } else if (!realization.get(w).getWord().startsWith(SFX.TOKEN_X)
                                        && !(valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\"")
                                        || valueTBM.startsWith(SFX.TOKEN_X))) {
                                    String valueToCheck = valueTBM;
                                    if (valueToCheck.equals("no")
                                            || valueToCheck.equals("yes")
                                            || valueToCheck.equals("yes or no")
                                            || valueToCheck.equals("none")
                                            || valueToCheck.equals("dont_care")
                                            || valueToCheck.equals("empty")) {
                                        String attribute = attrValue;
                                        if (attribute.contains("=")) {
                                            attribute = attrValue.substring(0, attrValue.indexOf('='));
                                        }
                                        valueToCheck = attribute + ":" + valueTBM;
                                    }
                                    if (!valueToCheck.equals("empty:empty")
                                            && valueAlignments.containsKey(valueToCheck)) {
                                        for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
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
                            if (!realization.get(w).getWord().startsWith(SFX.TOKEN_X)) {
                                for (String attrValueTBM : attrValuesToBeMentioned) {
                                    if (attrValueTBM.contains("=")) {
                                        String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                        if (!(value.matches("\"[xX][0-9]+\"")
                                                || value.matches("[xX][0-9]+")
                                                || value.startsWith(SFX.TOKEN_X))) {
                                            String valueToCheck = value;
                                            if (valueToCheck.equals("no")
                                                    || valueToCheck.equals("yes")
                                                    || valueToCheck.equals("yes or no")
                                                    || valueToCheck.equals("none")
                                                    || valueToCheck.equals("dont_care")
                                                    || valueToCheck.equals("empty")) {
                                                valueToCheck = attrValueTBM.replace("=", ":");
                                            }
                                            if (!valueToCheck.equals("empty:empty")
                                                    && valueAlignments.containsKey(valueToCheck)) {
                                                for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                                    if (endsWith(subPhrase, alignedStr)) {
                                                        mentionedAttrValue = attrValueTBM;
                                                        break;
                                                    }
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
                //}
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
     for (DatasetInstance di : datasetInstances.get(predicate)) {
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
     ArrayList<Action> availableActions = new ArrayList(availableWordActions.get(SFX.TOKEN_ATTR));
     for (String at : attributes) {
     if (!at.equals(SFX.TOKEN_PUNCT)) {
     Action act = new Action("", "");
     int c = 0;
     while (!act.getWord().equals(SFX.TOKEN_START)
     && !act.getWord().equals(SFX.TOKEN_END)) {
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
     act = new Action(SFX.TOKEN_END, at);
     realization.add(new Action(SFX.TOKEN_END, at));
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
     if (!realization.get(w).getAttribute().equals(SFX.TOKEN_PUNCT)) {
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
     if (!realization.get(w).getWord().equals(SFX.TOKEN_START)
     && !realization.get(w).getWord().equals(SFX.TOKEN_END)) {
     subPhrase.add(realization.get(w).getWord());
     }
     }
     if (!isValueMentioned) {
     if (realization.get(w).getWord().startsWith(SFX.TOKEN_X)
     && (valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+"))) {
     isValueMentioned = true;
     } else if (!realization.get(w).getWord().startsWith(SFX.TOKEN_X)
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
     for (DatasetInstance di : datasetInstances.get(predicate)) {
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
     if (!at.equals(SFX.TOKEN_PUNCT)
     && !at.equals(SFX.TOKEN_START)
     && !at.equals(SFX.TOKEN_END)
     && !at.equals("[]")) {
     for (Action a : real) {
     if (!a.getWord().equals(SFX.TOKEN_START)
     && !a.getWord().equals(SFX.TOKEN_END)) {
     realization.add(new Action(a.getWord(), at));
     }
     }
     realization.add(new Action(SFX.TOKEN_END, at));
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
     if (!realization.get(w).getAttribute().equals(SFX.TOKEN_PUNCT)) {
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
     if (!realization.get(w).getWord().equals(SFX.TOKEN_START)
     && !realization.get(w).getWord().equals(SFX.TOKEN_END)) {
     subPhrase.add(realization.get(w).getWord());
     }
     }
     if (!isValueMentioned) {
     if (realization.get(w).getWord().startsWith(SFX.TOKEN_X)
     && (valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+"))) {
     isValueMentioned = true;
     } else if (!realization.get(w).getWord().startsWith(SFX.TOKEN_X)
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
        HashMap<ArrayList<Action>, ArrayList<Action>> calculatedRealizationsCache = new HashMap<>();
        HashMap<Integer, HashSet<String>> nGrams = new HashMap<>();
        for (DatasetInstance di : trainingData) {
            HashSet<ArrayList<Action>> initRealizations = new HashSet<>();
            for (ArrayList<Action> real : di.getEvalRealizations()) {
                if (!calculatedRealizationsCache.containsKey(real)) {
                    initRealizations.add(real);
                }
            }
            if (!calculatedRealizationsCache.containsKey(di.getTrainRealization())) {
                initRealizations.add(di.getTrainRealization());
            }
            for (ArrayList<Action> realization : initRealizations) {
                HashMap<String, HashSet<String>> values = new HashMap<>();
                for (String attr : di.getMeaningRepresentation().getAttributes().keySet()) {
                    values.put(attr, new HashSet<>(di.getMeaningRepresentation().getAttributes().get(attr)));
                }

                ArrayList<Action> randomRealization = new ArrayList<Action>();
                for (Action a : realization) {
                    if (a.getAttribute().equals(SFX.TOKEN_PUNCT)) {
                        randomRealization.add(new Action(a.getWord(), a.getAttribute()));
                    } else {
                        randomRealization.add(new Action(a.getWord(), ""));
                    }
                }

                HashSet<String> unalignedAttrs = new HashSet<>();
                if (values.keySet().isEmpty()) {
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!attributes.get(di.getMeaningRepresentation().getPredicate()).contains("empty")) {
                                attributes.get(di.getMeaningRepresentation().getPredicate()).add("empty");
                            }
                            randomRealization.get(i).setAttribute("empty=empty");
                        }
                    }
                } else {
                    for (String attr : values.keySet()) {
                        for (String value : values.get(attr)) {
                            if ((!(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+") || value.startsWith(SFX.TOKEN_X)))
                                    && !value.isEmpty()) {
                                String valueToCheck = value;
                                if (valueToCheck.equals("no")
                                        || valueToCheck.equals("yes")
                                        || valueToCheck.equals("yes or no")
                                        || valueToCheck.equals("none")
                                        || valueToCheck.equals("dont_care")
                                        || valueToCheck.equals("empty")) {
                                    valueToCheck = attr + ":" + value;
                                    unalignedAttrs.add(attr + "=" + value);
                                }
                                if (valueToCheck.equals(attr)) {
                                    unalignedAttrs.add(attr + "=" + value);
                                }
                                if (!valueToCheck.equals("empty:empty")
                                        && valueAlignments.containsKey(valueToCheck)) {
                                    unalignedAttrs.add(attr + "=" + valueToCheck);
                                }
                            } else {
                                unalignedAttrs.add(attr + "=" + value);
                            }
                        }
                    }

                    for (String attrValue : unalignedAttrs) {
                        int index = r.nextInt(randomRealization.size());
                        boolean change = false;
                        while (!change) {
                            if (!randomRealization.get(index).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                                randomRealization.get(index).setAttribute(attrValue.toLowerCase().trim());
                                change = true;
                            } else {
                                index = r.nextInt(randomRealization.size());
                            }
                        }
                    }

                    String previousAttr = "";
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        } else {
                            previousAttr = "";
                        }
                    }
                    //System.out.println("1: " + randomRealization);

                    previousAttr = "";
                    for (int i = randomRealization.size() - 1; i >= 0; i--) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        } else {
                            previousAttr = "";
                        }
                    }
                    //System.out.println("2: " + randomRealization);

                    previousAttr = "";
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        }
                    }
                    //System.out.println("3: " + randomRealization);

                    previousAttr = "";
                    for (int i = randomRealization.size() - 1; i >= 0; i--) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        }
                    }
                    //System.out.println("4: " + randomRealization);
                }

                //FIX WRONG @PUNCT@
                String previousAttr = "";
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
                    if (!a.getAttribute().equals(SFX.TOKEN_PUNCT)) {
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
                            endRandomRealization.add(new Action(SFX.TOKEN_END, previousAttr));
                        }
                    }
                    endRandomRealization.add(a);
                    previousAttr = a.getAttribute();
                }
                endRandomRealization.add(new Action(SFX.TOKEN_END, previousAttr));
                endRandomRealization.add(new Action(SFX.TOKEN_END, SFX.TOKEN_END));
                calculatedRealizationsCache.put(realization, endRandomRealization);
                //System.out.println(di.getMeaningRepresentation().getPredicate() + ": " + endRandomRealization);

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
                    if (a.getAttribute().equals(SFX.TOKEN_PUNCT)
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

            HashSet<ArrayList<Action>> newRealizations = new HashSet<>();
            for (ArrayList<Action> real : di.getEvalRealizations()) {
                newRealizations.add(calculatedRealizationsCache.get(real));
            }
            di.setEvalRealizations(newRealizations);
            di.setTrainRealization(calculatedRealizationsCache.get(di.getTrainRealization()));
            for (ArrayList<Action> rr : di.getEvalRealizations()) {
                for (Action key : rr) {
                    if (key.getWord().trim().isEmpty()) {
                        System.out.println("RR " + di.getMeaningRepresentation().getMRstr());
                        System.out.println("RR " + di.getMeaningRepresentation().getAttributes());
                        System.out.println("RR " + rr);
                        System.out.println("RR " + key);
                        System.exit(0);
                    }
                    if (key.getAttribute().equals("[]")
                            || key.getAttribute().isEmpty()) {
                        System.out.println("RR " + di.getMeaningRepresentation().getMRstr());
                        System.out.println("RR " + di.getMeaningRepresentation().getAttributes());
                        System.out.println("RR " + rr);
                        System.out.println("RR " + key);
                        System.exit(0);
                    }
                }
            }
            for (Action key : di.getTrainRealization()) {
                if (key.getWord().trim().isEmpty()) {
                    System.out.println("RR " + di.getMeaningRepresentation().getMRstr());
                    System.out.println("RR " + di.getMeaningRepresentation().getAttributes());
                    System.out.println("RR " + key);
                    System.exit(0);
                }
                if (key.getAttribute().equals("[]")
                        || key.getAttribute().isEmpty()) {
                    System.out.println("RR " + di.getMeaningRepresentation().getMRstr());
                    System.out.println("RR " + di.getMeaningRepresentation().getAttributes());
                    System.out.println("RR " + di.getTrainRealization());
                    System.out.println("RR " + key);
                    System.exit(0);
                }
            }
            //di.setRealizations(randomRealizations);
        }
        return nGrams;
    }

    public HashMap<Integer, HashSet<String>> createNaiveAlignments(ArrayList<DatasetInstance> trainingData) {
        punctPatterns = new HashMap<>();
        HashMap<ArrayList<Action>, ArrayList<Action>> calculatedRealizationsCache = new HashMap<>();
        HashMap<Integer, HashSet<String>> nGrams = new HashMap<>();
        for (DatasetInstance di : trainingData) {
            HashSet<ArrayList<Action>> initRealizations = new HashSet<>();
            for (ArrayList<Action> real : di.getEvalRealizations()) {
                if (!calculatedRealizationsCache.containsKey(real)) {
                    initRealizations.add(real);
                }
            }
            if (!calculatedRealizationsCache.containsKey(di.getTrainRealization())) {
                initRealizations.add(di.getTrainRealization());
            }
            for (ArrayList<Action> realization : initRealizations) {
                HashMap<String, HashSet<String>> values = new HashMap<>();
                for (String attr : di.getMeaningRepresentation().getAttributes().keySet()) {
                    values.put(attr, new HashSet<>(di.getMeaningRepresentation().getAttributes().get(attr)));
                }

                ArrayList<Action> randomRealization = new ArrayList<Action>();
                for (Action a : realization) {
                    if (a.getAttribute().equals(SFX.TOKEN_PUNCT)) {
                        randomRealization.add(new Action(a.getWord(), a.getAttribute()));
                    } else {
                        randomRealization.add(new Action(a.getWord(), ""));
                    }
                }

                if (values.keySet().isEmpty()) {
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!attributes.get(di.getMeaningRepresentation().getPredicate()).contains("empty")) {
                                attributes.get(di.getMeaningRepresentation().getPredicate()).add("empty");
                            }
                            randomRealization.get(i).setAttribute("empty=empty");
                        }
                    }
                } else {
                    HashMap<Double, HashMap<String, ArrayList<Integer>>> indexAlignments = new HashMap<>();
                    HashSet<String> noValueAttrs = new HashSet<String>();
                    for (String attr : values.keySet()) {
                        for (String value : values.get(attr)) {
                            if ((!(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+") || value.startsWith(SFX.TOKEN_X)))
                                    && !value.isEmpty()) {
                                String valueToCheck = value;
                                if (valueToCheck.equals("no")
                                        || valueToCheck.equals("yes")
                                        || valueToCheck.equals("yes or no")
                                        || valueToCheck.equals("none")
                                        || valueToCheck.equals("dont_care")
                                        || valueToCheck.equals("empty")) {
                                    valueToCheck = attr + ":" + value;
                                    noValueAttrs.add(attr + "=" + value);
                                }
                                if (valueToCheck.equals(attr)) {
                                    noValueAttrs.add(attr + "=" + value);
                                }
                                if (!valueToCheck.equals("empty:empty")
                                        && valueAlignments.containsKey(valueToCheck)) {
                                    for (ArrayList<String> align : valueAlignments.get(valueToCheck).keySet()) {
                                        int n = align.size();
                                        for (int i = 0; i <= randomRealization.size() - n; i++) {
                                            ArrayList<String> compare = new ArrayList<String>();
                                            ArrayList<Integer> indexAlignment = new ArrayList<Integer>();
                                            for (int j = 0; j < n; j++) {
                                                compare.add(randomRealization.get(i + j).getWord());
                                                indexAlignment.add(i + j);
                                            }
                                            if (compare.equals(align)) {
                                                if (!indexAlignments.containsKey(valueAlignments.get(valueToCheck).get(align))) {
                                                    indexAlignments.put(valueAlignments.get(valueToCheck).get(align), new HashMap());
                                                }
                                                indexAlignments.get(valueAlignments.get(valueToCheck).get(align)).put(attr + "=" + valueToCheck, indexAlignment);
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
                        if (a.getWord().startsWith(SFX.TOKEN_X)) {
                            String attr = a.getWord().substring(3, a.getWord().lastIndexOf('_')).toLowerCase().trim();
                            a.setAttribute(attr + "=" + a.getWord());
                        }
                    }
                    //System.out.println("-1: " + randomRealization);

                    HashSet<String> unalignedNoValueAttrs = new HashSet<>();
                    for (String noValueAttr : noValueAttrs) {
                        boolean assigned = false;
                        for (Action a : randomRealization) {
                            if (a.getAttribute().equals(noValueAttr)) {
                                assigned = true;
                            }
                        }
                        if (!assigned) {
                            unalignedNoValueAttrs.add(noValueAttr);
                        }
                    }

                    boolean isAllEmpty = true;
                    boolean hasSpace = false;
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (!randomRealization.get(i).getAttribute().isEmpty()
                                && !randomRealization.get(i).getAttribute().equals("[]")
                                && !randomRealization.get(i).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                            isAllEmpty = false;
                        }
                        if (randomRealization.get(i).getAttribute().isEmpty()
                                || randomRealization.get(i).getAttribute().equals("[]")) {
                            hasSpace = true;
                        }
                    }
                    if (isAllEmpty && hasSpace && !unalignedNoValueAttrs.isEmpty()) {
                        for (String attrValue : unalignedNoValueAttrs) {
                            int index = r.nextInt(randomRealization.size());
                            boolean change = false;
                            while (!change) {
                                if (!randomRealization.get(index).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                                    randomRealization.get(index).setAttribute(attrValue.toLowerCase().trim());
                                    change = true;
                                } else {
                                    index = r.nextInt(randomRealization.size());
                                }
                            }
                        }
                    }
                    //System.out.println(isAllEmpty + " " + hasSpace + " " + unalignedNoValueAttrs);
                    //System.out.println(">> " + noValueAttrs);
                    //System.out.println(">> " + values);
                    //System.out.println("0: " + randomRealization);

                    String previousAttr = "";
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        } else {
                            previousAttr = "";
                        }
                    }
                    //System.out.println("1: " + randomRealization);

                    previousAttr = "";
                    for (int i = randomRealization.size() - 1; i >= 0; i--) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        } else {
                            previousAttr = "";
                        }
                    }
                    //System.out.println("2: " + randomRealization);

                    previousAttr = "";
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        }
                    }
                    //System.out.println("3: " + randomRealization);

                    previousAttr = "";
                    for (int i = randomRealization.size() - 1; i >= 0; i--) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(SFX.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        }
                    }
                    //System.out.println("4: " + randomRealization);
                }

                //FIX WRONG @PUNCT@
                String previousAttr = "";
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
                    if (!a.getAttribute().equals(SFX.TOKEN_PUNCT)) {
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
                            endRandomRealization.add(new Action(SFX.TOKEN_END, previousAttr));
                        }
                    }
                    endRandomRealization.add(a);
                    previousAttr = a.getAttribute();
                }
                endRandomRealization.add(new Action(SFX.TOKEN_END, previousAttr));
                endRandomRealization.add(new Action(SFX.TOKEN_END, SFX.TOKEN_END));
                calculatedRealizationsCache.put(realization, endRandomRealization);
                //System.out.println(di.getMeaningRepresentation().getPredicate() + ": " + endRandomRealization);

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
                    if (a.getAttribute().equals(SFX.TOKEN_PUNCT)
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

            HashSet<ArrayList<Action>> newRealizations = new HashSet<>();
            for (ArrayList<Action> real : di.getEvalRealizations()) {
                newRealizations.add(calculatedRealizationsCache.get(real));
            }
            di.setEvalRealizations(newRealizations);
            di.setTrainRealization(calculatedRealizationsCache.get(di.getTrainRealization()));
            for (ArrayList<Action> rr : di.getEvalRealizations()) {
                for (Action key : rr) {
                    if (key.getWord().trim().isEmpty()) {
                        System.out.println("RR " + di.getMeaningRepresentation().getMRstr());
                        System.out.println("RR " + di.getMeaningRepresentation().getAttributes());
                        System.out.println("RR " + rr);
                        System.out.println("RR " + key);
                        System.exit(0);
                    }
                    if (key.getAttribute().equals("[]")
                            || key.getAttribute().isEmpty()) {
                        System.out.println("RR " + di.getMeaningRepresentation().getMRstr());
                        System.out.println("RR " + di.getMeaningRepresentation().getAttributes());
                        System.out.println("RR " + rr);
                        System.out.println("RR " + key);
                        System.exit(0);
                    }
                }
            }
            for (Action key : di.getTrainRealization()) {
                if (key.getWord().trim().isEmpty()) {
                    System.out.println("RR " + di.getMeaningRepresentation().getMRstr());
                    System.out.println("RR " + di.getMeaningRepresentation().getAttributes());
                    System.out.println("RR " + key);
                    System.exit(0);
                }
                if (key.getAttribute().equals("[]")
                        || key.getAttribute().isEmpty()) {
                    System.out.println("RR " + di.getMeaningRepresentation().getMRstr());
                    System.out.println("RR " + di.getMeaningRepresentation().getAttributes());
                    System.out.println("RR " + di.getTrainRealization());
                    System.out.println("RR " + key);
                    System.exit(0);
                }
            }
            //di.setRealizations(randomRealizations);
        }
        return nGrams;
    }

    public Instance createAttrInstance(String predicate, String bestAction, ArrayList<String> previousGeneratedAttrs, ArrayList<Action> previousGeneratedWords, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, MeaningRepresentation MR, HashMap<String, HashSet<String>> availableAttributeActions) {
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
            if (bestAction.equals(SFX.TOKEN_END)) {
                costs.put(SFX.TOKEN_END, 0.0);
                for (String action : availableAttributeActions.get(predicate)) {
                    costs.put(action, 1.0);
                }
            } else if (!bestAction.equals("@TOK@")) {
                costs.put(SFX.TOKEN_END, 1.0);
                for (String action : availableAttributeActions.get(predicate)) {
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
        return SFX.this.createAttrInstance(predicate, previousGeneratedAttrs, previousGeneratedWords, costs, attrValuesAlreadyMentioned, attrValuesToBeMentioned, MR, availableAttributeActions);
    }

    public Instance createAttrInstance(String predicate, ArrayList<String> previousGeneratedAttrs, ArrayList<Action> previousGeneratedWords, TObjectDoubleHashMap<String> costs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, MeaningRepresentation MR, HashMap<String, HashSet<String>> availableAttributeActions) {
        TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        for (String action : availableAttributeActions.get(predicate)) {
            valueSpecificFeatures.put(action, new TObjectDoubleHashMap<String>());
        }

        ArrayList<String> mentionedAttrValues = new ArrayList<>();
        for (String attrValue : previousGeneratedAttrs) {
            if (!attrValue.equals(SFX.TOKEN_END)
                    && !attrValue.equals(SFX.TOKEN_START)) {
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
        for (String attribute : availableAttributeActions.get(predicate)) {
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
                if (!a.getWord().equals(SFX.TOKEN_START)
                        && !a.getWord().equals(SFX.TOKEN_END)) {
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
        for (String action : availableAttributeActions.get(predicate)) {
            if (action.equals(SFX.TOKEN_END)) {
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
                    //if (!action.equals("empty")
                    //        && !action.equals(SFX.TOKEN_END)) {
                    valueSpecificFeatures.get(action).put("global_feature_specific_isInMR", 1.0);
                    //}
                } else {
                    //valueSpecificFeatures.get(action).put("feature_specific_isNotInMR", 1.0);
                    //if (!action.equals("empty")
                    //        && !action.equals(SFX.TOKEN_END)) {
                    valueSpecificFeatures.get(action).put("global_feature_specific_isNotInMR", 1.0);
                    //}
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
                //Is attr to be mentioned (has value to express)
                boolean toBeMentioned = false;
                for (String attrValue : attrValuesToBeMentioned) {
                    if (attrValue.substring(0, attrValue.indexOf('=')).equals(action)) {
                        toBeMentioned = true;
                        //valueSpecificFeatures.get(action).put("feature_specific_attrToBeMentioned", 1.0);
                        //if (!action.equals("empty")) {
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrToBeMentioned", 1.0);
                        //}
                    }
                }
                if (!toBeMentioned) {
                    //valueSpecificFeatures.get(action).put("feature_specific_attrNotToBeMentioned", 1.0);
                    //if (!action.equals("empty")) {
                    valueSpecificFeatures.get(action).put("global_feature_specific_attrNotToBeMentioned", 1.0);
                    //}
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
            String attr = bestAction.getAttribute();
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

            if (bestAction.getWord().trim().equalsIgnoreCase(SFX.TOKEN_END)) {
                costs.put(SFX.TOKEN_END, 0.0);
            } else {
                costs.put(SFX.TOKEN_END, 1.0);
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
        if (currentValue.contains(":")) {
            currentValue = currentAttrValue.substring(currentAttrValue.indexOf(':') + 1);
        }
        if (currentValue.isEmpty()) {
            //System.exit(0);
        }

        TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        for (Action action : availableWordActions.get(currentAttr)) {
            valueSpecificFeatures.put(action.getWord(), new TObjectDoubleHashMap<String>());
        }

        /*if (gWords.get(wIndex).getWord().equals(SFX.TOKEN_END)) {
         System.out.println("!!! "+ gWords.subList(0, wIndex + 1));
         }*/
        ArrayList<Action> generatedWords = new ArrayList<>();
        ArrayList<Action> generatedWordsInSameAttrValue = new ArrayList<>();
        ArrayList<String> generatedPhrase = new ArrayList<>();
        for (int i = 0; i < previousGeneratedWords.size(); i++) {
            Action a = previousGeneratedWords.get(i);
            if (!a.getWord().equals(SFX.TOKEN_START)
                    && !a.getWord().equals(SFX.TOKEN_END)) {
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
                    valuesThatFollow.add(SFX.TOKEN_X + attr + "_" + v.substring(1));
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
                    mentionedValues.add(SFX.TOKEN_X + attr + "_" + v.substring(1));
                } else {
                    mentionedValues.add(v);
                }
            } else {
                generalFeatures.put("feature_attrsAlreadyMentioned_" + attrValue.toLowerCase(), 1.0);
            }
        }

        /*System.out.println("currentAttrValue: " + currentAttrValue);
         System.out.println("5W: " + prev5gram);
         System.out.println("5AW: " + prevAttrWord5gram);
         System.out.println("5A: " + prevAttr5gram);
         System.out.println("VM: " + wasValueMentioned);
         System.out.println("A_TF: " + attrValuesThatFollow);
         System.out.println("==============================");*/
        if (currentValue.equals("no")
                || currentValue.equals("yes")
                || currentValue.equals("yes or no")
                || currentValue.equals("none")
                || currentValue.equals("empty")
                || currentValue.equals("dont_care")) {
            generalFeatures.put("feature_emptyValue", 1.0);
        }

        //Word specific features (and also global features)
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
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_notAppearedInSameAttrValue", 1.0);
                    //valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_notAppearedInSameAttrValue", 1.0);
                }
            }
            //Has word appeared before
            for (Action previousAction : generatedWords) {
                if (previousAction.getWord().equals(action.getWord())) {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_appeared", 1.0);
                    valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_appeared", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_notAppeared", 1.0);
                    //valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_notAppeared", 1.0);
                }
            }
            if (currentValue.equals("no")
                    || currentValue.equals("yes")
                    || currentValue.equals("yes or no")
                    || currentValue.equals("none")
                    || currentValue.equals("empty")
                    || currentValue.equals("dont_care")) {
                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_emptyValue", 1.0);
                valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_emptyValue", 1.0);
            } else {
                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_notEmptyValue", 1.0);
                //valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_notEmptyValue", 1.0);
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

        /*HashSet<String> keys = new HashSet<>(generalFeatures.keySet());
         for (String feature1 : keys) {
         for (String feature2 : keys) {
         if (generalFeatures.get(feature1) == 1.0
         && generalFeatures.get(feature2) == 1.0
         && feature1.compareTo(feature2) < 0) {
         generalFeatures.put(feature1 + "&&" + feature2, 1.0);
         }
         }
         }*/
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
