/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import similarity_measures.Levenshtein;
import uk.ac.ucl.jdagger.JDAggerForBagel;

/**
 *
 * @author localadmin
 */
public class Bagel {

    static HashMap<String, HashMap<String, HashSet<Action>>> dictionary = new HashMap<>();
    static HashMap<String, HashSet<String>> attributes = new HashMap<>();
    static HashMap<String, HashSet<String>> attributeValuePairs = new HashMap<>();
    static HashMap<String, HashMap<MeaningRepresentation, HashSet<String>>> meaningReprs = new HashMap<>();
    static HashMap<String, ArrayList<DatasetInstance>> abstractDatasetInstances = new HashMap<>();
    static HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments = new HashMap<>();
    static ArrayList<String> predicates = new ArrayList<>();
    HashMap<String, ArrayList<Instance>> predicateArgTrainingData;
    HashMap<String, HashMap<String, ArrayList<Instance>>> predicateWordTrainingData;
    final public static String TOKEN_END = "@end@";
    final public static String TOKEN_PUNCT = "@punct@";
    final public static String TOKEN_X = "@x@";
    final public static String TOKEN_ATTR = "@attr@";
    public static int rounds = 10;
    public static int maxRealizationSize = 100;
    public static ArrayList<Double> crossAvgArgDistances = new ArrayList<>();
    public static ArrayList<Double> crossNIST = new ArrayList<>();
    public static ArrayList<Double> crossBLEU = new ArrayList<>();
    public static ArrayList<Double> crossBLEUSmooth = new ArrayList<>();

    public static boolean useAlignments = false;

    public static void main(String[] args) {
        boolean useDAggerArg = false;
        boolean useLolsWord = true;

        runTestWithJAROW(useDAggerArg, useLolsWord, useAlignments);
    }

    public static void runTestWithJAROW(boolean useDAggerArg, boolean useDAggerWord, boolean useAlignments) {
        File dataFile = new File("bagel_data\\ACL10-inform-training.txt");

        createLists(dataFile);
        //initializeEvaluation();

        HashMap<String, JAROW> classifiersArgs = new HashMap<>();
        HashMap<String, HashMap<String, JAROW>> classifiersWords = new HashMap<>();

        HashMap<String, HashMap<DatasetInstance, ArrayList<Instance>>> predicateArgTrainingData = createArgTrainingDatasets();
        HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData;
        if (useAlignments) {
            predicateWordTrainingData = createWordTrainingDatasets();
        } else {
            predicateWordTrainingData = createRandomAlignmentWordTrainingDatasets(); //createAllWithAllWordTrainingDatasets(); //createRandomWordTrainingDatasets();
        }

        for (String predicate : predicates) {
            ArrayList<DatasetInstance> datasetInstances = new ArrayList(predicateArgTrainingData.get(predicate).keySet());
            //for (double f = 0.0; f < 1.0; f += 0.1) {
            for (double f = 0.0; f < 0.1; f += 0.1) {
                int from = ((int) Math.round(datasetInstances.size() * f)); //+ 1;

                if (from < datasetInstances.size()) {
                    int to = (int) Math.round(datasetInstances.size() * (f + 0.1));
                    if (to > datasetInstances.size()) {
                        to = datasetInstances.size();
                    }
                    ArrayList<DatasetInstance> testingData = new ArrayList<>(datasetInstances.subList(from, to));
                    ArrayList<DatasetInstance> trainingData = new ArrayList<>(datasetInstances);
                    trainingData.removeAll(testingData);

                    ArrayList<Instance> testingArgInstances = new ArrayList();
                    ArrayList<Instance> trainingArgInstances = new ArrayList();
                    HashMap<String, ArrayList<Instance>> testingWordInstances = new HashMap();
                    HashMap<String, ArrayList<Instance>> trainingWordInstances = new HashMap();
                    for (String attribute : attributes.get(predicate)) {
                        testingWordInstances.put(attribute, new ArrayList<Instance>());
                        trainingWordInstances.put(attribute, new ArrayList<Instance>());
                    }
                    for (DatasetInstance di : testingData) {
                        testingArgInstances.addAll(predicateArgTrainingData.get(predicate).get(di));
                        for (String attribute : attributes.get(predicate)) {
                            if (predicateWordTrainingData.get(predicate).get(di).containsKey(attribute)) {
                                testingWordInstances.get(attribute).addAll(predicateWordTrainingData.get(predicate).get(di).get(attribute));
                            }
                        }
                    }
                    for (DatasetInstance di : predicateArgTrainingData.get(predicate).keySet()) {
                        if (!testingData.contains(di)) {
                            trainingArgInstances.addAll(predicateArgTrainingData.get(predicate).get(di));
                        }
                    }

                    for (DatasetInstance di : predicateWordTrainingData.get(predicate).keySet()) {
                        if (!testingData.contains(di)) {
                            for (String attribute : attributes.get(predicate)) {
                                if (predicateWordTrainingData.get(predicate).get(di).containsKey(attribute)) {
                                    trainingWordInstances.get(attribute).addAll(predicateWordTrainingData.get(predicate).get(di).get(attribute));
                                }
                            }
                        }
                    }

                    boolean setToGo = true;
                    if (trainingWordInstances.isEmpty() || trainingArgInstances.isEmpty() || testingWordInstances.isEmpty() || testingArgInstances.isEmpty()) {
                        setToGo = false;
                    }

                    if (setToGo) {
                        if (useDAggerWord) {
                            classifiersArgs.put(predicate, train(predicate, trainingArgInstances));
                            classifiersWords.put(predicate, JDAggerForBagel.runLOLS(predicate, attributes.get(predicate), abstractDatasetInstances.get(predicate), trainingWordInstances, dictionary, valueAlignments, 0.8, classifiersArgs.get(predicate), testingData));
                        } else {
                            classifiersArgs.put(predicate, train(predicate, trainingArgInstances));
                            for (String attribute : attributes.get(predicate)) {
                                if (!classifiersWords.containsKey(predicate)) {
                                    classifiersWords.put(predicate, new HashMap<String, JAROW>());
                                }
                                if (!trainingWordInstances.get(attribute).isEmpty()) {
                                    classifiersWords.get(predicate).put(attribute, train(predicate, trainingWordInstances.get(attribute)));
                                }
                            }
                        }

                        evaluateGeneration(classifiersArgs.get(predicate), classifiersWords.get(predicate), testingData, predicate);
                    }
                }
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

    public static JAROW trainWords(String predicate, String attribute, ArrayList<DatasetInstance> trainingData) {
        JDAggerForBagel dagger = new JDAggerForBagel();
        System.out.println("Run dagger for property " + predicate);
        //JAROW classifier = dagger.runVDAggerForWords(predicate, attribute, trainingData, availableWordActions, 5, 0.7);

        //return classifier;
        return null;
    }

    public static JAROW train(String predicate, ArrayList<Instance> trainingData) {
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

    public static Double evaluateGeneration(JAROW classifierArgs, HashMap<String, JAROW> classifierWords, ArrayList<DatasetInstance> testingData, String predicate) {
        System.out.println("Evaluate argument generation " + predicate);

        int totalArgDistance = 0;
        ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
        ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
        for (DatasetInstance di : testingData) {
            //PHRASE GENERATION EVALUATION
            String predictedArg = "";
            int aW = 0;
            ArrayList<String> predictedAttrValuePairs = new ArrayList<>();
            ArrayList<String> argsToBeMentioned = new ArrayList<>();
            for (String argument : di.getMeaningRepresentation().getAttributes().keySet()) {
                //for (String value : di.getMeaningRepresentation().getAttributes().get(argument)) {
                argsToBeMentioned.add(argument);
                //}
            }
            while (!predictedArg.equals(Bagel.TOKEN_END) && predictedAttrValuePairs.size() < 10000) {
                ArrayList<String> tempList = new ArrayList(predictedAttrValuePairs);
                tempList.add("@TOK@");
                Instance argTrainingVector = createArgInstance(predicate, tempList, aW, new HashSet(argsToBeMentioned), di.getMeaningRepresentation());

                if (argTrainingVector != null) {
                    Prediction predict = classifierArgs.predict(argTrainingVector);
                    predictedArg = predict.getLabel().trim();

                    predictedAttrValuePairs.add(predictedArg);
                    argsToBeMentioned.remove(predictedArg);
                }
                aW++;
            }

            //FILTER OUT ALL ATTRIBUTES THAT SHOULD NOT HAVE BEEN GENERATED
            HashSet<String> toBeRemoved = new HashSet<>();
            for (String attributeValuePair : predictedAttrValuePairs) {
                if (!attributeValuePair.equals(TOKEN_END)) {
                    String[] args = attributeValuePair.split("=");
                    if (!di.getMeaningRepresentation().getAttributes().keySet().contains(args[0])) {
                        toBeRemoved.add(attributeValuePair);
                    }/* else if (!di.getMeaningRepresentation().getAttributes().get(args[0]).contains(args[1])) {
                     toBeRemoved.add(attributeValuePair);
                     }*/

                }
            }
            predictedAttrValuePairs.removeAll(toBeRemoved);

            ArrayList<String> predictedAttrs = new ArrayList<>();
            for (String attributeValuePair : predictedAttrValuePairs) {
                predictedAttrs.add(attributeValuePair.split("=")[0]);
            }

            //GENERATE PHRASES
            ArrayList<Action> predictedWordsList = new ArrayList<>();
            String predictedString = "";
            ArrayList<String> mentionedValueSequence = null;
            for (ArrayList<String> m : di.getMentionedValueSequences().values()) {
                mentionedValueSequence = m;
                break;
            }
            ArrayList<String> predictedAttributes = new ArrayList<>();
            HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
            for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                valuesToBeMentioned.put(attribute, new ArrayList(di.getMeaningRepresentation().getAttributes().get(attribute)));
            }
            //for (int a = 0; a < predictedAttrValuePairs.size(); a++) {
            //String[] args = predictedAttrValuePairs.get(a).split("=");
            //String attribute = args[0];
            for (int a = 0; a < mentionedValueSequence.size(); a++) {
                String attribute = mentionedValueSequence.get(a).split("=")[0];
                predictedAttributes.add(attribute);

                if (!predictedAttributes.get(a).equals(Bagel.TOKEN_END)) {
                    String predictedWord = "";
                    int wW = predictedWordsList.size();

                    boolean valueWasMentioned = false;
                    ArrayList<String> subPhrase = new ArrayList<>();
                    while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < maxRealizationSize) {
                        ArrayList<Action> tempList = new ArrayList(predictedWordsList);
                        tempList.add(new Action("@TOK@", predictedAttributes.get(a)));
                        Instance wordTrainingVector = createWordInstance(predicate, predictedAttributes, a, tempList, wW, valuesToBeMentioned.get(attribute), di.getMeaningRepresentation(), false, false);

                        if (wordTrainingVector != null) {
                            if (classifierWords.get(attribute) != null) {
                                Prediction predict = classifierWords.get(attribute).predict(wordTrainingVector);
                                if (predict.getLabel() != null) {
                                    predictedWord = predict.getLabel().trim();
                                    predictedWordsList.add(new Action(predictedWord, predictedAttributes.get(a)));
                                    if (!predictedWord.equals(Bagel.TOKEN_END)) {
                                        subPhrase.add(predictedWord);
                                    }
                                } else {
                                    predictedWord = Bagel.TOKEN_END;
                                    predictedWordsList.add(new Action(predictedWord, predictedAttributes.get(a)));
                                }
                            } else {
                                predictedWord = Bagel.TOKEN_END;
                                predictedWordsList.add(new Action(predictedWord, predictedAttributes.get(a)));
                            }
                        }
                        /*System.out.println("==============");
                         System.out.println(valuesToBeMentioned);
                         System.out.println(attribute);
                         System.out.println(predictedWord);*/
                        if (!valuesToBeMentioned.get(attribute).isEmpty()) {
                            if (predictedWord.startsWith(Bagel.TOKEN_X)) {
                                int x = 1;

                                boolean containsVariables = false;
                                for (String value : valuesToBeMentioned.get(attribute)) {
                                    if (value.startsWith("\"X")) {
                                        containsVariables = true;
                                    }
                                }
                                if (containsVariables) {
                                    while (!valuesToBeMentioned.get(attribute).remove("\"X" + x + "\"")) {
                                        x++;
                                    }
                                }
                                valueWasMentioned = true;
                            } else {
                                HashSet<String> conveyedValues = new HashSet<>();
                                for (String value : valuesToBeMentioned.get(attribute)) {
                                    if (!value.matches("\"X[0-9]+\"")) {
                                        for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                            if (endsWith(subPhrase, alignedStr)) {
                                                conveyedValues.add(value);
                                            }
                                        }
                                    }
                                }

                                valuesToBeMentioned.get(attribute).removeAll(conveyedValues);
                                if (!conveyedValues.isEmpty()) {
                                    valueWasMentioned = true;
                                }
                            }
                        }
                        // System.out.println(valuesToBeMentioned);
                        //System.out.println("==============");
                        wW++;
                    }
                    if (!valueWasMentioned && !valuesToBeMentioned.get(attribute).isEmpty()) {
                        HashMap<String[], Double> alignments = new HashMap<>();
                        //Calculate all alignment similarities
                        for (String value : valuesToBeMentioned.get(attribute)) {
                            for (String subWord : subPhrase) {
                                if (!value.matches("\"X[0-9]+\"")) {
                                    String[] alignment = new String[2];
                                    alignment[0] = value;
                                    alignment[1] = subWord;

                                    Double distance = Levenshtein.getSimilarity(value, subWord, true, false);
                                    alignments.put(alignment, distance);
                                }
                            }
                        }
                        Double max = Double.MIN_VALUE;
                        String[] bestAlignment = new String[2];
                        for (String[] alignment : alignments.keySet()) {
                            if (alignments.get(alignment) > max) {
                                max = alignments.get(alignment);
                                bestAlignment = alignment;
                            }
                        }
                        valuesToBeMentioned.get(attribute).remove(bestAlignment[0]);
                    }
                    predictedWordsList.remove(predictedWordsList.size() - 1);

                    /*predictedWordsList.addAll(predictedWordsAttrList);
                     for (String word : predictedWordsAttrList) {
                     predictedString += word + " ";
                     }*/
                }
            }
            for (Action action : predictedWordsList) {
                if (action.getWord().startsWith(Bagel.TOKEN_X)) {
                    predictedString += "X ";
                } else {
                    predictedString += action.getWord() + " ";
                }
            }
            predictedString = predictedString.replaceAll("\\p{Punct}|\\d", "");
            predictedString = predictedString.trim() + ".";

            Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedString.toLowerCase()));
            ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
            generations.add(tran);

            ArrayList<Sequence<IString>> references = new ArrayList<>();
            for (ArrayList<Action> realization : di.getRealizations()) {
                String cleanedWords = "";
                for (Action nlWord : realization) {
                    if (!nlWord.equals(new Action(Bagel.TOKEN_END, ""))) {
                        if (nlWord.getWord().startsWith(Bagel.TOKEN_X)) {
                            cleanedWords += "X ";
                        } else {
                            cleanedWords += nlWord.getWord() + " ";
                        }
                    }
                }
                //UNCOMMENTED THE REPLACES HERE
                cleanedWords = cleanedWords.replaceAll("\\p{Punct}|\\d", "");
                cleanedWords = cleanedWords.trim();
                if (!cleanedWords.endsWith(".")) {
                    cleanedWords += ".";
                }
                references.add(IStrings.tokenize(NISTTokenizer.tokenize(cleanedWords)));
            }
            finalReferences.add(references);

            //EVALUATE ATTRIBUTE SEQUENCE
            HashSet<ArrayList<String>> goldAttributeSequences = new HashSet<>();
            for (DatasetInstance di2 : testingData) {
                if (di2.getMeaningRepresentation().getAttributes().equals(di.getMeaningRepresentation().getAttributes())) {
                    goldAttributeSequences.addAll(di2.getMentionedAttributeSequences().values());
                }
            }

            //for (ArrayList<String> goldArgs : abstractMeaningReprs.get(predicate).get(mr).values()) {
            int minTotArgDistance = Integer.MAX_VALUE;
            ArrayList<String> minGoldArgs = null;
            for (ArrayList<String> goldArgs : goldAttributeSequences) {
                int totArgDistance = 0;
                HashSet<Integer> matchedPositions = new HashSet<>();
                for (int i = 0; i < predictedAttrs.size(); i++) {
                    if (!predictedAttrs.get(i).equals(Bagel.TOKEN_END)) {
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
                    if (!goldArg.equals(Bagel.TOKEN_END)) {
                        boolean contained = predictedCopy.remove(goldArg);
                        if (!contained) {
                            totArgDistance += 100;
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
        System.out.println("Avg arg distance: " + totalArgDistance / (double) testingData.size());
        System.out.println("NIST: " + nistScore);
        System.out.println("BLEU: " + bleuScore);
        System.out.println("BLEU smooth: " + bleuSmoothScore);

        return bleuScore;
    }

    public static void createLists(File dataFile) {
        dictionary = new HashMap<>();
        attributes = new HashMap<>();
        attributeValuePairs = new HashMap<>();
        valueAlignments = new HashMap<>();
        predicates = new ArrayList<>();

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
                        if (!dictionary.containsKey(previousPredicate)) {
                            dictionary.put(previousPredicate, new HashMap<String, HashSet<Action>>());
                        }
                    }

                    line = line.substring(line.indexOf("(") + 1, line.lastIndexOf(")"));

                    HashMap<String, String> names = new HashMap<>();
                    int s = line.indexOf("\"");
                    int a = 1;
                    while (s != -1) {
                        int e = line.indexOf("\"", s + 1);

                        String name = line.substring(s, e + 1);
                        //line = line.replace(name, "@@$$" + a + "$$@@");
                        //names.put("@@$$" + a + "$$@@", name);
                        line = line.replace(name, "X" + a);
                        names.put("X" + a, name);
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
                            argumentValues.put(subArg[0], new HashSet<String>());
                        }
                        argumentValues.get(subArg[0]).add(value.toLowerCase());
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
                    previousMR = new MeaningRepresentation(previousPredicate, argumentValues);
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
                        if (!dictionary.containsKey(previousPredicate)) {
                            dictionary.put(previousPredicate, new HashMap<String, HashSet<Action>>());
                        }
                    }

                    line = line.substring(line.indexOf("(") + 1, line.lastIndexOf(")"));

                    HashMap<String, String> names = new HashMap<>();
                    int s = line.indexOf("\"");
                    int a = 0;
                    while (s != -1) {
                        int e = line.indexOf("\"", s + 1);

                        String name = line.substring(s, e + 1);
                        //line = line.replace(name, "@@$$" + a + "$$@@");
                        //names.put("@@$$" + a + "$$@@", name);
                        line = line.replace(name, "X" + a);
                        names.put("X" + a, name);
                        a++;

                        s = line.indexOf("\"");
                    }

                    HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
                    String[] args = line.split(",");

                    for (String arg : args) {
                        String[] subAttr = arg.split("=");
                        String value = subAttr[1];
                        if (names.containsKey(value)) {
                            value = names.get(value);
                        }
                        if (!attributes.get(previousPredicate).contains(subAttr[0])) {
                            attributes.get(previousPredicate).add(subAttr[0]);
                        }
                        if (!attributeValues.containsKey(subAttr[0])) {
                            attributeValues.put(subAttr[0], new HashSet<String>());
                        }
                        attributeValues.get(subAttr[0]).add(value);
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

                    previousAMR = new MeaningRepresentation(previousPredicate, attributeValues);
                } else if (line.startsWith("->")) {
                    line = line.substring(line.indexOf("\"") + 1, line.lastIndexOf("\""));

                    ArrayList<String> mentionedValueSequence = new ArrayList<>();
                    ArrayList<String> mentionedAttributeSequence = new ArrayList<>();

                    ArrayList<String> realization = new ArrayList<>();
                    ArrayList<String> alignedRealization = new ArrayList<>();

                    String[] words = line.replaceAll("([,.?!;:'])", " $1").split(" ");
                    int a = 0;
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
                                if (mentionedValue.contains("+")) {
                                    mentionedAttribute = mentionedValue.substring(1, mentionedValue.indexOf("+"));

                                    if (previousAMR.getAttributes().containsKey(mentionedAttribute)) {
                                        if (mentionedValueSequence.isEmpty()) {
                                            String v = mentionedValue.substring(1, mentionedValue.length() - 1).replaceAll("\\+", "=");
                                            if (v.endsWith("=X")) {
                                                //v = v.replace("=X", "=@@$$" + a + "$$@@");
                                                v = v.replace("=X", "=X" + a);
                                            }
                                            mentionedValueSequence.add(v);
                                        } else if (!mentionedValueSequence.get(mentionedValueSequence.size() - 1).equals(mentionedValue)) {
                                            String v = mentionedValue.substring(1, mentionedValue.length() - 1).replaceAll("\\+", "=");
                                            if (v.endsWith("=X")) {
                                                //v = v.replace("=X", "=@@$$" + +a + "$$@@");
                                                v = v.replace("=X", "=X" + a);
                                            }
                                            mentionedValueSequence.add(v);
                                        }

                                        if (mentionedAttributeSequence.isEmpty()) {
                                            mentionedAttributeSequence.add(mentionedAttribute);
                                        } else if (!mentionedAttributeSequence.get(mentionedAttributeSequence.size() - 1).equals(mentionedAttribute)) {
                                            mentionedAttributeSequence.add(mentionedAttribute);
                                        }
                                    }
                                } else {
                                    mentionedAttribute = mentionedValue.substring(1, mentionedValue.length() - 1);

                                    if (!previousAMR.getAttributes().containsKey(mentionedAttribute)) {
                                        mentionedAttribute = "";
                                    }
                                }

                                //s = line.indexOf("[");
                            }
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
                                realization.add(Bagel.TOKEN_X + mentionedAttribute);
                            } else {
                                realization.add(words[i].trim().toLowerCase());
                            }
                            //}
                        }
                    }

                    if (realization.size() > maxRealizationSize) {
                        maxRealizationSize = realization.size();
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
                                mentionedValueSequence.add(0, attr + "=" + value);
                                mentionedAttributeSequence.add(0, attr);

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
                            if (!value.equals("name=none") && !value.matches("\"X[0-9]+\"")) {
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

                                            Double distance = Levenshtein.getSimilarity(value, compare, true, false);
                                            alignments.get(value).put(align, distance);
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
                        if (existingDI.getMeaningRepresentation().equals(previousAMR)) {
                            existing = true;
                            existingDI.mergeDatasetInstance(mentionedValueSequence, mentionedAttributeSequence, realizationActions);
                        }
                    }
                    if (!existing) {
                        DatasetInstance DI = new DatasetInstance(previousAMR, mentionedValueSequence, mentionedAttributeSequence, realizationActions);
                        abstractDatasetInstances.get(previousPredicate).add(DI);
                    }

                    if (!useAlignments) {
                        for (int i = 0; i < realization.size(); i++) {
                            String word = realization.get(i);
                            String attribute = alignedRealization.get(i);

                            if (!attribute.equals(Bagel.TOKEN_PUNCT)) {
                                if (!dictionary.get(previousPredicate).containsKey(Bagel.TOKEN_ATTR)) {
                                    dictionary.get(previousPredicate).put(Bagel.TOKEN_ATTR, new HashSet());
                                    dictionary.get(previousPredicate).get(Bagel.TOKEN_ATTR).add(new Action(Bagel.TOKEN_END, Bagel.TOKEN_ATTR));
                                    dictionary.get(previousPredicate).get(Bagel.TOKEN_ATTR).add(new Action(Bagel.TOKEN_X + previousPredicate, Bagel.TOKEN_ATTR));
                                }
                                if (!word.equalsIgnoreCase("X") && !word.trim().isEmpty()) {
                                    dictionary.get(previousPredicate).get(Bagel.TOKEN_ATTR).add(new Action(word.toLowerCase(), Bagel.TOKEN_ATTR));
                                }
                            }
                        }
                    } else {
                        for (int i = 0; i < realization.size(); i++) {
                            String word = realization.get(i);
                            String attribute = alignedRealization.get(i);

                            if (!attribute.equals(Bagel.TOKEN_PUNCT)) {
                                if (!dictionary.get(previousPredicate).containsKey("")) {
                                    dictionary.get(previousPredicate).put(attribute, new HashSet());
                                    dictionary.get(previousPredicate).get(attribute).add(new Action(Bagel.TOKEN_END, attribute));
                                    dictionary.get(previousPredicate).get(attribute).add(new Action(Bagel.TOKEN_X + previousPredicate, attribute));
                                }
                                if (!word.equalsIgnoreCase("X")) {
                                    dictionary.get(previousPredicate).get(attribute).add(new Action(word.toLowerCase(), attribute));
                                }
                            }
                        }
                    }
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static HashMap<String, HashMap<DatasetInstance, ArrayList<Instance>>> createArgTrainingDatasets() {
        HashMap<String, HashMap<DatasetInstance, ArrayList<Instance>>> predicateArgTrainingData = new HashMap<>();

        if (!dictionary.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*/) {
            for (String predicate : predicates) {
                predicateArgTrainingData.put(predicate, new HashMap<DatasetInstance, ArrayList<Instance>>());
                for (DatasetInstance di : abstractDatasetInstances.get(predicate)) {
                    ArrayList<Instance> instances = new ArrayList<>();

                    for (ArrayList<String> mentionedValueSequence : di.getMentionedValueSequences().values()) {
                        //For every mentioned argument in realization
                        ArrayList<String> argsToBeMentioned = new ArrayList<>();
                        for (String arg : di.getMeaningRepresentation().getAttributes().keySet()) {
                            // for (String value : di.getMeaningRepresentation().getAttributes().get(arg)) {
                            argsToBeMentioned.add(arg);
                            //}
                        }

                        for (int w = 0; w < mentionedValueSequence.size(); w++) {
                            Instance argTrainingVector = createArgInstance(predicate, mentionedValueSequence, w, new HashSet(argsToBeMentioned), di.getMeaningRepresentation());

                            if (argTrainingVector != null) {
                                instances.add(argTrainingVector);
                            }
                            argsToBeMentioned.remove(mentionedValueSequence.get(w));
                        }
                    }

                    predicateArgTrainingData.get(predicate).put(di, instances);
                }
            }
        }
        return predicateArgTrainingData;
    }

    public static HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> createWordTrainingDatasets() {
        HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData = new HashMap<>();

        if (!dictionary.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*/) {
            for (String predicate : predicates) {
                predicateWordTrainingData.put(predicate, new HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>());

                ArrayList<String> attrs = new ArrayList<>();
                for (DatasetInstance di : abstractDatasetInstances.get(predicate)) {
                    HashMap<String, ArrayList<Instance>> instances = new HashMap<>();
                    for (ArrayList<Action> realization : di.getRealizations()) {
                        HashMap<String, HashSet<String>> values = new HashMap();
                        for (String attr : di.getMeaningRepresentation().getAttributes().keySet()) {
                            values.put(attr, new HashSet(di.getMeaningRepresentation().getAttributes().get(attr)));
                        }
                        HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
                        for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                            valuesToBeMentioned.put(attribute, new ArrayList(di.getMeaningRepresentation().getAttributes().get(attribute)));
                        }
                        String previousAlignment = "";
                        ArrayList<String> subPhrase = new ArrayList<>();
                        for (int w = 0; w < realization.size(); w++) {
                            if (!realization.get(w).getAttribute().equals(previousAlignment) && !previousAlignment.isEmpty()) {
                                Instance wordTrainingVector = createWordInstance(predicate, attrs, attrs.size() - 1, realization, w, valuesToBeMentioned.get(previousAlignment), di.getMeaningRepresentation(), true, false);

                                if (wordTrainingVector != null) {
                                    if (!instances.containsKey(previousAlignment)) {
                                        instances.put(previousAlignment, new ArrayList<Instance>());
                                    }
                                    instances.get(previousAlignment).add(wordTrainingVector);
                                }
                                subPhrase = new ArrayList<>();
                            }

                            if (!realization.get(w).getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                                if (!previousAlignment.equals(realization.get(w).getAttribute())) {
                                    attrs.add(realization.get(w).getAttribute());
                                }
                                previousAlignment = realization.get(w).getAttribute();
                                Instance wordTrainingVector = createWordInstance(predicate, attrs, attrs.size() - 1, realization, w, valuesToBeMentioned.get(previousAlignment), di.getMeaningRepresentation(), false, false);

                                if (wordTrainingVector != null) {
                                    if (!instances.containsKey(previousAlignment)) {
                                        instances.put(previousAlignment, new ArrayList<Instance>());
                                    }
                                    instances.get(previousAlignment).add(wordTrainingVector);
                                    subPhrase.add(realization.get(w).getWord());
                                }
                                if (!valuesToBeMentioned.get(previousAlignment).isEmpty()) {
                                    if (realization.get(w).getWord().startsWith(Bagel.TOKEN_X)) {
                                        int x = 1;
                                        while (!valuesToBeMentioned.get(previousAlignment).remove("\"X" + x + "\"")) {
                                            x++;
                                        }
                                    } else {
                                        HashSet<String> conveyedValues = new HashSet<>();
                                        for (String value : valuesToBeMentioned.get(previousAlignment)) {
                                            if (!value.matches("\"X[0-9]+\"")) {
                                                for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                                    if (endsWith(subPhrase, alignedStr)) {
                                                        conveyedValues.add(value);
                                                    }
                                                }
                                            }
                                        }
                                        valuesToBeMentioned.get(previousAlignment).removeAll(conveyedValues);
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

    public static HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> createRandomWordTrainingDatasets() {
        HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData = new HashMap<>();

        if (!dictionary.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*/) {
            for (String predicate : predicates) {
                predicateWordTrainingData.put(predicate, new HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>());

                ArrayList<String> attrs = new ArrayList<>();
                for (DatasetInstance di : abstractDatasetInstances.get(predicate)) {
                    HashMap<String, ArrayList<Instance>> instances = new HashMap<>();
                    for (ArrayList<Action> real : di.getRealizations()) {
                        ArrayList<String> attributes = new ArrayList<String>();
                        for (Action act : real) {
                            if (attributes.isEmpty()) {
                                attributes.add(act.getAttribute());
                            } else if (!attributes.get(attributes.size() - 1).equals(act.getAttribute())) {
                                attributes.add(act.getAttribute());
                            }
                        }
                        ArrayList<Action> realization = new ArrayList<Action>();
                        Random r = new Random();
                        ArrayList<Action> availableActions = new ArrayList(dictionary.get(predicate).get(Bagel.TOKEN_ATTR));
                        for (String at : attributes) {
                            if (!at.equals(Bagel.TOKEN_PUNCT)) {
                                Action act = new Action("", "");
                                int c = 0;
                                while (!act.getWord().equals(Bagel.TOKEN_END)) {
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
                        HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
                        for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                            valuesToBeMentioned.put(attribute, new ArrayList(di.getMeaningRepresentation().getAttributes().get(attribute)));
                        }
                        String previousAlignment = "";
                        ArrayList<String> subPhrase = new ArrayList<>();
                        for (int w = 0; w < realization.size(); w++) {
                            if (!realization.get(w).getAttribute().equals(previousAlignment) && !previousAlignment.isEmpty()) {
                                Instance wordTrainingVector = createWordInstance(predicate, attrs, attrs.size() - 1, realization, w, valuesToBeMentioned.get(previousAlignment), di.getMeaningRepresentation(), true, true);

                                if (wordTrainingVector != null) {
                                    if (!instances.containsKey(previousAlignment)) {
                                        instances.put(previousAlignment, new ArrayList<Instance>());
                                    }
                                    instances.get(previousAlignment).add(wordTrainingVector);
                                }
                                subPhrase = new ArrayList<>();
                            }

                            if (!realization.get(w).getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                                if (!previousAlignment.equals(realization.get(w).getAttribute())) {
                                    attrs.add(realization.get(w).getAttribute());
                                }
                                previousAlignment = realization.get(w).getAttribute();
                                Instance wordTrainingVector = createWordInstance(predicate, attrs, attrs.size() - 1, realization, w, valuesToBeMentioned.get(previousAlignment), di.getMeaningRepresentation(), false, true);

                                if (wordTrainingVector != null) {
                                    if (!instances.containsKey(previousAlignment)) {
                                        instances.put(previousAlignment, new ArrayList<Instance>());
                                    }
                                    instances.get(previousAlignment).add(wordTrainingVector);
                                    subPhrase.add(realization.get(w).getWord());
                                }
                                if (!valuesToBeMentioned.get(previousAlignment).isEmpty()) {
                                    if (realization.get(w).getWord().startsWith(Bagel.TOKEN_X)) {
                                        String xValue = "";
                                        for (String v : valuesToBeMentioned.get(previousAlignment)) {
                                            if (v.startsWith("\"X")) {
                                                xValue = v;
                                                break;
                                            }
                                        }
                                        if (!xValue.isEmpty()) {
                                            valuesToBeMentioned.get(previousAlignment).remove(xValue);
                                        }
                                    } else {
                                        HashSet<String> conveyedValues = new HashSet<>();
                                        for (String value : valuesToBeMentioned.get(previousAlignment)) {
                                            if (!value.matches("\"X[0-9]+\"")) {
                                                for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                                    if (endsWith(subPhrase, alignedStr)) {
                                                        conveyedValues.add(value);
                                                    }
                                                }
                                            }
                                        }
                                        valuesToBeMentioned.get(previousAlignment).removeAll(conveyedValues);
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

    public static HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> createAllWithAllWordTrainingDatasets() {
        HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData = new HashMap<>();

        if (!dictionary.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*/) {
            for (String predicate : predicates) {
                predicateWordTrainingData.put(predicate, new HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>());

                ArrayList<String> attrs = new ArrayList<>();
                for (DatasetInstance di : abstractDatasetInstances.get(predicate)) {
                    HashMap<String, ArrayList<Instance>> instances = new HashMap<>();
                    for (ArrayList<Action> real : di.getRealizations()) {
                        ArrayList<String> mentionedAttributes = di.getMentionedAttributeSequences().get(real);
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
                                    && !at.equals(Bagel.TOKEN_END)
                                    && !at.equals("[]")) {
                                for (Action a : real) {
                                    if (!a.getWord().equals(Bagel.TOKEN_END)) {
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
                        HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
                        for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                            valuesToBeMentioned.put(attribute, new ArrayList(di.getMeaningRepresentation().getAttributes().get(attribute)));
                        }
                        String previousAlignment = "";
                        ArrayList<String> subPhrase = new ArrayList<>();
                        for (int w = 0; w < realization.size(); w++) {
                            if (!realization.get(w).getAttribute().equals(previousAlignment) && !previousAlignment.isEmpty()) {
                                Instance wordTrainingVector = createWordInstance(predicate, attrs, attrs.size() - 1, realization, w, valuesToBeMentioned.get(previousAlignment), di.getMeaningRepresentation(), true, false);

                                if (wordTrainingVector != null) {
                                    if (!instances.containsKey(previousAlignment)) {
                                        instances.put(previousAlignment, new ArrayList<Instance>());
                                    }
                                    instances.get(previousAlignment).add(wordTrainingVector);
                                }
                                subPhrase = new ArrayList<>();
                            }

                            if (!realization.get(w).getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                                if (!previousAlignment.equals(realization.get(w).getAttribute())) {
                                    attrs.add(realization.get(w).getAttribute());
                                }
                                previousAlignment = realization.get(w).getAttribute();
                                Instance wordTrainingVector = createWordInstance(predicate, attrs, attrs.size() - 1, realization, w, valuesToBeMentioned.get(previousAlignment), di.getMeaningRepresentation(), false, false);

                                if (wordTrainingVector != null) {
                                    if (!instances.containsKey(previousAlignment)) {
                                        instances.put(previousAlignment, new ArrayList<Instance>());
                                    }
                                    instances.get(previousAlignment).add(wordTrainingVector);
                                    subPhrase.add(realization.get(w).getWord());
                                }
                                if (!valuesToBeMentioned.get(previousAlignment).isEmpty()) {
                                    if (realization.get(w).getWord().startsWith(Bagel.TOKEN_X)) {
                                        String xValue = "";
                                        for (String v : valuesToBeMentioned.get(previousAlignment)) {
                                            if (v.startsWith("\"X")) {
                                                xValue = v;
                                                break;
                                            }
                                        }
                                        if (!xValue.isEmpty()) {
                                            valuesToBeMentioned.get(previousAlignment).remove(xValue);
                                        }
                                    } else {
                                        HashSet<String> conveyedValues = new HashSet<>();
                                        for (String value : valuesToBeMentioned.get(previousAlignment)) {
                                            if (!value.matches("\"X[0-9]+\"")) {
                                                for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                                    if (endsWith(subPhrase, alignedStr)) {
                                                        conveyedValues.add(value);
                                                    }
                                                }
                                            }
                                        }
                                        valuesToBeMentioned.get(previousAlignment).removeAll(conveyedValues);
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

    public static HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> createRandomAlignmentWordTrainingDatasets() {
        HashMap<String, HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData = new HashMap<>();

        if (!dictionary.isEmpty() && !predicates.isEmpty()/* && !arguments.isEmpty()*/) {
            for (String predicate : predicates) {
                predicateWordTrainingData.put(predicate, new HashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>());

                ArrayList<String> attrs = new ArrayList<>();
                for (DatasetInstance di : abstractDatasetInstances.get(predicate)) {
                    HashMap<String, ArrayList<Instance>> instances = new HashMap<>();
                    for (ArrayList<Action> realization : di.getRealizations()) {
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

                        //x assignment
                        /*HashMap<Integer, HashSet<ArrayList<Integer>>> xClusters = new HashMap<>();
                         for (int i = 0; i < randomRealization.size(); i++) {
                         if (randomRealization.get(i).getWord().startsWith(Bagel.TOKEN_X)) {
                         ArrayList<Integer> index = new ArrayList<>();
                         index.add(i);
                         if (!xClusters.containsKey(index.size())) {
                         xClusters.put(index.size(), new HashSet<>());
                         }
                         xClusters.get(index.size()).add(index);

                         int j = i + 1;
                         while (j < randomRealization.size()) {
                         if (randomRealization.get(j).getWord().startsWith(Bagel.TOKEN_X)) {
                         index = new ArrayList<>(index);
                         index.add(j);

                         if (!xClusters.containsKey(index.size())) {
                         xClusters.put(index.size(), new HashSet<>());
                         }
                         xClusters.get(index.size()).add(index);
                         j++;
                         } else if (randomRealization.get(j).getWord().equals("and")
                         || randomRealization.get(j).getWord().equals(",")
                         || randomRealization.get(j).getWord().equals("or")) {
                         j++;
                         } else {
                         j = randomRealization.size();
                         }
                         }
                         }
                         }*/
                        HashMap<Double, HashMap<String, ArrayList<Integer>>> indexAlignments = new HashMap<>();
                        //HashMap<Integer, HashSet<ArrayList<String>>> attrXValues = new HashMap<>();
                        for (String attr : values.keySet()) {
                            if (!attr.equals("type")) {
                                //ArrayList<String> xValues = new ArrayList<>();
                                for (String value : values.get(attr)) {
                                    if (!value.matches("\"X[0-9]+\"")) {
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
                                                    indexAlignments.get(valueAlignments.get(value).get(align)).put(attr + "|" + value, indexAlignment);
                                                }
                                            }
                                        }
                                    } //else {
                                    //    xValues.add(attr + "|" + value);
                                    //}
                                }
                                /*if (!xValues.isEmpty()) {
                                 if (!attrXValues.containsKey(xValues.size())) {
                                 attrXValues.put(xValues.size(), new HashSet<>());
                                 }
                                 attrXValues.get(xValues.size()).add(xValues);
                                 }*/
                            }
                        }
                        /*ArrayList<Integer> attrXValuesLengths = new ArrayList<>(attrXValues.keySet());
                         Collections.sort(attrXValuesLengths);
                         HashSet<Integer> assignedXIntegers = new HashSet<Integer>();
                         HashSet<String> assignedXAttrValues = new HashSet<String>();
                         for (int a = attrXValuesLengths.size() - 1; a >= 0; a--) {
                         for (ArrayList<String> attrValues : attrXValues.get(attrXValuesLengths.get(a))) {
                         if (xClusters.get(attrValues.size()) == null) {
                         ArrayList<String> splitAttrValuesLeft = new ArrayList<String>();
                         ArrayList<String> splitAttrValuesRight = new ArrayList<String>();
                         for (int i = 0; i < attrValues.size() - 1; i++) {
                         splitAttrValuesLeft.add(attrValues.get(i));
                         }
                         splitAttrValuesRight.add(attrValues.get(attrValues.size() - 1));

                         attrXValues.get(splitAttrValuesLeft.size()).add(splitAttrValuesLeft);
                         attrXValues.get(splitAttrValuesRight.size()).add(splitAttrValuesRight);
                         } else {
                         ArrayList<ArrayList<Integer>> xCls = new ArrayList<>(xClusters.get(attrValues.size()));
                         //Collections.shuffle(xCls);

                         for (int i = 0; i < xCls.size(); i++) {
                         boolean isUnassigned = true;

                         for (Integer index : xCls.get(i)) {
                         if (assignedXIntegers.contains(index)) {
                         isUnassigned = false;
                         }
                         }
                         if (isUnassigned) {
                         assignedXAttrValues.add(attrValues.get(0));
                         String attr = attrValues.get(0).substring(0, attrValues.get(0).indexOf("|"));
                         for (int j = xCls.get(i).get(0); j <= xCls.get(i).get(xCls.get(i).size() - 1); j++) {
                         assignedXIntegers.add(j);
                         randomRealization.get(j).setAttribute(attr);
                         }
                         i = xCls.size();
                         }
                         }
                         }
                         }
                         }*/
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
                                        String attr = attrValue.substring(0, attrValue.indexOf("|"));
                                        for (Integer index : indexAlignments.get(similarities.get(i)).get(attrValue)) {
                                            assignedIntegers.add(index);
                                            randomRealization.get(index).setAttribute(attr);
                                        }
                                    }
                                }
                            }
                        }

                        for (Action a : randomRealization) {
                            if (a.getWord().startsWith(Bagel.TOKEN_X)) {
                                a.setAttribute(a.getWord().substring(3));
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

                        HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
                        for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                            valuesToBeMentioned.put(attribute, new ArrayList(di.getMeaningRepresentation().getAttributes().get(attribute)));
                        }
                        String previousAlignment = "";
                        ArrayList<String> subPhrase = new ArrayList<>();
                        for (int w = 0; w < randomRealization.size(); w++) {
                            if (!randomRealization.get(w).getAttribute().equals(previousAlignment) && !previousAlignment.isEmpty()) {
                                Instance wordTrainingVector = createWordInstance(predicate, attrs, attrs.size() - 1, randomRealization, w, valuesToBeMentioned.get(previousAlignment), di.getMeaningRepresentation(), true, false);

                                if (wordTrainingVector != null) {
                                    if (!instances.containsKey(previousAlignment)) {
                                        instances.put(previousAlignment, new ArrayList<Instance>());
                                    }
                                    instances.get(previousAlignment).add(wordTrainingVector);
                                }
                                subPhrase = new ArrayList<>();
                            }

                            if (!randomRealization.get(w).getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                                if (!previousAlignment.equals(randomRealization.get(w).getAttribute())) {
                                    attrs.add(randomRealization.get(w).getAttribute());
                                }
                                previousAlignment = randomRealization.get(w).getAttribute();
                                Instance wordTrainingVector = createWordInstance(predicate, attrs, attrs.size() - 1, randomRealization, w, valuesToBeMentioned.get(previousAlignment), di.getMeaningRepresentation(), false, false);

                                if (wordTrainingVector != null) {
                                    if (!instances.containsKey(previousAlignment)) {
                                        instances.put(previousAlignment, new ArrayList<Instance>());
                                    }
                                    instances.get(previousAlignment).add(wordTrainingVector);
                                    subPhrase.add(randomRealization.get(w).getWord());
                                }
                                if (!valuesToBeMentioned.get(previousAlignment).isEmpty()) {
                                    if (randomRealization.get(w).getWord().startsWith(Bagel.TOKEN_X)) {
                                        int x = 1;
                                        while (!valuesToBeMentioned.get(previousAlignment).remove("\"X" + x + "\"")) {
                                            x++;
                                        }
                                    } else {
                                        HashSet<String> conveyedValues = new HashSet<>();
                                        for (String value : valuesToBeMentioned.get(previousAlignment)) {
                                            if (!value.matches("\"X[0-9]+\"")) {
                                                for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                                    if (endsWith(subPhrase, alignedStr)) {
                                                        conveyedValues.add(value);
                                                    }
                                                }
                                            }
                                        }
                                        valuesToBeMentioned.get(previousAlignment).removeAll(conveyedValues);
                                    }
                                }
                            }
                        }
                    }
                    predicateWordTrainingData.get(predicate).put(di, instances);
                    for (String attr : attributes.get(predicate)) {
                        if (instances.containsKey(attr)) {
                            for (Instance in : instances.get(attr)) {
                                for (String s : in.getFeatureVector().keySet()) {
                                    if (s.startsWith("feature_5gram")) {
                                        System.out.println(s + " " + in.getCorrectLabels());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return predicateWordTrainingData;
    }

    public static Instance createArgInstance(String predicate, ArrayList<String> mentionedArgs, int w, HashSet<String> argsToBeMentioned, MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        String bestAction = mentionedArgs.get(w).trim();
        if (!bestAction.isEmpty()) {
            //COSTS
            for (String action : attributeValuePairs.get(predicate)) {
                if (action.equals(bestAction)) {
                    costs.put(action, 0.0);
                } else {
                    costs.put(action, 1.0);
                }
            }
            if (bestAction.equals(Bagel.TOKEN_END)) {
                costs.put(Bagel.TOKEN_END, 0.0);
            } else {
                costs.put(Bagel.TOKEN_END, 1.0);
            }
        }
        return createArgInstance(predicate, mentionedArgs, w, costs, argsToBeMentioned, MR);
    }

    public static Instance createArgInstance(String predicate, ArrayList<String> mentionedArgs, int w, double cost, HashSet<String> argsToBeMentioned, MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        String bestAction = mentionedArgs.get(w).trim();
        if (!bestAction.isEmpty()) {
            //COSTS
            for (String action : attributeValuePairs.get(predicate)) {
                if (action.equals(bestAction)) {
                    costs.put(action, 1.0 - cost);
                } else {
                    costs.put(action, 1.0);
                }
            }
            if (bestAction.equals(Bagel.TOKEN_END)) {
                costs.put(Bagel.TOKEN_END, 0.0);
            } else {
                costs.put(Bagel.TOKEN_END, 1.0);
            }
        }
        return createArgInstance(predicate, mentionedArgs, w, costs, argsToBeMentioned, MR);
    }

    public static Instance createArgInstance(String predicate, ArrayList<String> mentionedArgs, int w, TObjectDoubleHashMap<String> costs, HashSet<String> argsToBeMentioned, MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> features = new TObjectDoubleHashMap<>();

        //Previous word features
        for (int j = 1; j <= 5; j++) {
            String previousAttr = "";
            if (w - j >= 0) {
                previousAttr = mentionedArgs.get(w - j).trim();
            }
            if (!previousAttr.isEmpty()) {
                features.put("feature_" + j + "_" + previousAttr, 1.0);
            } else {
                features.put("feature_" + j + "_@@", 1.0);
            }
        }
        //Word N-Grams            
        String prevWord = "@@";
        if (w - 1 >= 0) {
            prevWord = mentionedArgs.get(w - 1).trim();
        }
        String prevPrevWord = "@@";
        if (w - 2 >= 0) {
            prevPrevWord = mentionedArgs.get(w - 2).trim();
        }
        String prevPrevPrevWord = "@@";
        if (w - 3 >= 0) {
            prevPrevPrevWord = mentionedArgs.get(w - 3).trim();
        }

        String prevBigram = prevPrevWord + "|" + prevWord;
        String prevTrigram = prevPrevPrevWord + "|" + prevPrevWord + "|" + prevWord;

        features.put("feature_bigram_" + prevBigram, 1.0);
        features.put("feature_trigram_" + prevTrigram, 1.0);

        //Word Positions
        //features.put("feature_pos:", w);
        //If arguments have already been generated or not
        for (String attr : attributes.get(predicate)) {
            if (argsToBeMentioned.contains(attr)) {
                features.put("feature_" + attr, 1.0);
            }
        }

        for (String attr : MR.getAttributes().keySet()) {
            for (String value : MR.getAttributes().get(attr)) {
                features.put("feature_v_" + value, 1.0);
            }
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
        return new Instance(features, costs);
    }

    public static Instance createWordInstance(String predicate, ArrayList<String> generatedAttributes, int a, ArrayList<Action> generatedWords, int w, ArrayList<String> valuesToBeMentioned, MeaningRepresentation MR, boolean isEnd, boolean isRandom) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        Action bestAction = generatedWords.get(w);
        if (isEnd) {
            bestAction = new Action(Bagel.TOKEN_END, generatedAttributes.get(a));
        }
        if (!bestAction.getWord().trim().isEmpty()) {
            //COSTS
            String attr = bestAction.getAttribute().trim();
            if (!useAlignments) {
                attr = Bagel.TOKEN_ATTR;
            }
            for (Action action : dictionary.get(predicate).get(attr)) {
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
        return createWordInstance(predicate, generatedAttributes, a, generatedWords, w, costs, valuesToBeMentioned, MR);
    }

    public static Instance createWordInstance(String predicate, ArrayList<String> generatedAttributes, int a, ArrayList<Action> gWords, int wIndex, TObjectDoubleHashMap<String> costs, ArrayList<String> valuesToBeMentioned, MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> features = new TObjectDoubleHashMap<>();

        int r = 0;
        ArrayList<Action> generatedWords = new ArrayList<>(gWords);
        while (generatedWords.contains(new Action(Bagel.TOKEN_END, ""))) {
            generatedWords.remove(new Action(Bagel.TOKEN_END, ""));
            r++;
        }
        int w = wIndex - r;

        //Previous word features
        //System.out.println("Â£Â£Â£ "  + nlWords);
        for (int j = 1; j <= 5; j++) {
            String previousWord = "";
            if (w - j >= 0) {
                previousWord = generatedWords.get(w - j).getWord().trim();
            }
            if (!previousWord.isEmpty()) {
                features.put("feature_" + j + "_" + previousWord.toLowerCase(), 1.0);
            } else {
                features.put("feature_" + j + "_@@", 1.0);
            }
        }
        //Word N-Grams            
        String prevWord = "@@";
        if (w - 1 >= 0) {
            prevWord = generatedWords.get(w - 1).getWord().trim();
        }
        String prevPrevWord = "@@";
        if (w - 2 >= 0) {
            prevPrevWord = generatedWords.get(w - 2).getWord().trim();
        }
        String prevPrevPrevWord = "@@";
        if (w - 3 >= 0) {
            prevPrevPrevWord = generatedWords.get(w - 3).getWord().trim();
        }
        String prevPrevPrevPrevWord = "@@";
        if (w - 4 >= 0) {
            prevPrevPrevPrevWord = generatedWords.get(w - 4).getWord().trim();
        }
        String prevPrevPrevPrevPrevWord = "@@";
        if (w - 5 >= 0) {
            prevPrevPrevPrevPrevWord = generatedWords.get(w - 5).getWord().trim();
        }
        String prevPrevPrevPrevPrevPrevWord = "@@";
        if (w - 6 >= 0) {
            prevPrevPrevPrevPrevPrevWord = generatedWords.get(w - 6).getWord().trim();
        }
        String prevPrevPrevPrevPrevPrevPrevWord = "@@";
        if (w - 7 >= 0) {
            prevPrevPrevPrevPrevPrevPrevWord = generatedWords.get(w - 7).getWord().trim();
        }

        String prevBigram = prevPrevWord + "|" + prevWord;
        String prevTrigram = prevPrevPrevWord + "|" + prevPrevWord + "|" + prevWord;
        String prev4gram = prevPrevPrevPrevWord + "|" + prevPrevPrevWord + "|" + prevPrevWord + "|" + prevWord;
        String prev5gram = prevPrevPrevPrevPrevWord + "|" + prevPrevPrevPrevWord + "|" + prevPrevPrevWord + "|" + prevPrevWord + "|" + prevWord;
        String prev6gram = prevPrevPrevPrevPrevPrevWord + "|" + prevPrevPrevPrevPrevWord + "|" + prevPrevPrevPrevWord + "|" + prevPrevPrevWord + "|" + prevPrevWord + "|" + prevWord;
        String prev7gram = prevPrevPrevPrevPrevPrevPrevWord + "|" + prevPrevPrevPrevPrevPrevWord + "|" + prevPrevPrevPrevPrevWord + "|" + prevPrevPrevPrevWord + "|" + prevPrevPrevWord + "|" + prevPrevWord + "|" + prevWord;

        //System.out.println(prev5gram.toLowerCase());
        features.put("feature_bigram_" + prevBigram.toLowerCase(), 1.0);
        features.put("feature_trigram_" + prevTrigram.toLowerCase(), 1.0);
        features.put("feature_4gram_" + prev4gram.toLowerCase(), 1.0);
        features.put("feature_5gram_" + prev5gram.toLowerCase(), 1.0);
        //features.put("feature_6gram_" + prev6gram.toLowerCase(), 1.0);
        //features.put("feature_7gram_" + prev7gram.toLowerCase(), 1.0);

        //Word N-Grams (after previous end)
        /*boolean metPreviousEnd = false;
         String prevAEWord = "@@";
         if (w - 1 >= 0) {
         if (!generatedWords.get(w - 1).trim().equals(Bagel.TOKEN_END) && !metPreviousEnd) {
         prevAEWord = generatedWords.get(w - 1).trim();
         } else {
         metPreviousEnd = true;
         }
         }
         String prevPrevAEWord = "@@";
         if (w - 2 >= 0) {
         if (!generatedWords.get(w - 2).trim().equals(Bagel.TOKEN_END) && !metPreviousEnd) {
         prevPrevAEWord = generatedWords.get(w - 2).trim();
         } else {
         metPreviousEnd = true;
         }
         }
         String prevPrevPrevAEWord = "@@";
         if (w - 3 >= 0) {
         if (!generatedWords.get(w - 3).trim().equals(Bagel.TOKEN_END) && !metPreviousEnd) {
         prevPrevPrevAEWord = generatedWords.get(w - 3).trim();
         } else {
         metPreviousEnd = true;
         }
         }
         String prevPrevPrevPrevAEWord = "@@";
         if (w - 4 >= 0) {
         if (!generatedWords.get(w - 4).trim().equals(Bagel.TOKEN_END) && !metPreviousEnd) {
         prevPrevPrevPrevAEWord = generatedWords.get(w - 4).trim();
         } else {
         metPreviousEnd = true;
         }
         }
         String prevPrevPrevPrevPrevAEWord = "@@";
         if (w - 5 >= 0) {
         if (!generatedWords.get(w - 5).trim().equals(Bagel.TOKEN_END) && !metPreviousEnd) {
         prevPrevPrevPrevPrevAEWord = generatedWords.get(w - 5).trim();
         } else {
         metPreviousEnd = true;
         }
         }
        
         String prevAEBigram = prevPrevAEWord + "|" + prevAEWord;
         String prevAETrigram = prevPrevPrevAEWord + "|" + prevPrevAEWord + "|" + prevAEWord;
         String prevAE4gram = prevPrevPrevPrevAEWord + "|" + prevPrevPrevAEWord + "|" + prevPrevAEWord + "|" + prevAEWord;
         String prevAE5gram = prevPrevPrevPrevPrevAEWord + "|" + prevPrevPrevPrevAEWord + "|" + prevPrevPrevAEWord + "|" + prevPrevAEWord + "|" + prevAEWord;
        
         features.put("feature_AE_2gram_" + prevAEBigram.toLowerCase(), 1.0);
         features.put("feature_AE_3gram_" + prevAETrigram.toLowerCase(), 1.0);
         features.put("feature_AE_4gram_" + prevAE4gram.toLowerCase(), 1.0);
         features.put("feature_AE_5gram_" + prevAE5gram.toLowerCase(), 1.0);*/
        //Previous attr features
        //System.out.println("Â£Â£Â£ "  + nlWords);
        
        //ENABLE ATTR FEATURES FOR A "weird" BOOST IN SCORES
        //Remember to check bug regarding the initialization of attrs collection in createRandomAlignment function
        /*for (int j = 1; j <= 5; j++) {
            String previousAttr = "";
            if (a - j >= 0) {
                previousAttr = generatedAttributes.get(a - j).trim();
            }
            if (!previousAttr.isEmpty()) {
                features.put("feature_attr_" + j + "_" + previousAttr, 1.0);
            } else {
                features.put("feature_attr_" + j + "_@@", 1.0);
            }
        }*/
        //Attribute N-Grams            
        String prevAttr = "@@";
        if (a - 1 >= 0) {
            prevAttr = generatedAttributes.get(a - 1).trim();
        }
        String prevPrevAttr = "@@";
        if (a - 2 >= 0) {
            prevPrevAttr = generatedAttributes.get(a - 2).trim();
        }
        String prevPrevPrevAttr = "@@";
        if (a - 3 >= 0) {
            prevPrevPrevAttr = generatedAttributes.get(a - 3).trim();
        }
        String prevPrevPrevPrevAttr = "@@";
        if (a - 4 >= 0) {
            prevPrevPrevPrevAttr = generatedAttributes.get(a - 4).trim();
        }
        String prevPrevPrevPrevPrevAttr = "@@";
        if (a - 5 >= 0) {
            prevPrevPrevPrevPrevAttr = generatedAttributes.get(a - 5).trim();
        }

        String prevAttrBigram = prevPrevAttr + "|" + prevAttr;
        String prevAttrTrigram = prevPrevPrevAttr + "|" + prevPrevAttr + "|" + prevAttr;
        String prevAttr4gram = prevPrevPrevPrevAttr + "|" + prevPrevPrevAttr + "|" + prevPrevAttr + "|" + prevAttr;
        String prevAttr5gram = prevPrevPrevPrevPrevAttr + "|" + prevPrevPrevPrevAttr + "|" + prevPrevPrevAttr + "|" + prevPrevAttr + "|" + prevAttr;

        /*features.put("feature_Attr_bigram_" + prevAttrBigram.toLowerCase(), 1.0);
        features.put("feature_Attr_trigram_" + prevAttrTrigram.toLowerCase(), 1.0);
        features.put("feature_Attr_4gram_" + prevAttr4gram.toLowerCase(), 1.0);
        features.put("feature_Attr_5gram_" + prevAttr5gram.toLowerCase(), 1.0);*/

        //If values have already been generated or not
        int x = 1;
        for (String value : valuesToBeMentioned) {
            if (value.matches("\"X[0-9]+\"")) {
                features.put("feature_valueToBeMentioned_X" + x, 1.0);
                x++;
            } else {
                features.put("feature_valueToBeMentioned_" + value.toLowerCase(), 1.0);
            }
        }
        //Value ngrams
        //THIS NEEDS WORK
        for (String value1 : valuesToBeMentioned) {
            for (String value2 : valuesToBeMentioned) {
                if (!value1.equalsIgnoreCase(value2)) {
                    String v1 = value1;
                    String v2 = value2;
                    x = 1;
                    /*if (v1.matches("\"X[0-9]+\"")) {
                     v1 = "X" + x;
                     x++;
                     } 
                     if (v2.matches("\"X[0-9]+\"")) {
                     v2 = "X" + x;
                     x++;
                     }*/
                    features.put("feature_2valueToBeMentioned_" + v1.toLowerCase() + "_" + v2.toLowerCase(), 1.0);
                }
            }
        }
        /*for (String value1 : valuesToBeMentioned) {
         for (String value2 : valuesToBeMentioned) {
         for (String value3 : valuesToBeMentioned) {
         if (!value1.equals(value2) && !value1.equals(value3) && !value2.equals(value3)) {                        
         String v1 = value1;
         String v2 = value2;
         String v3 = value3;
         x = 1;
         if (v1.matches("\"X[0-9]+\"")) {
         v1 = "X" + x;
         x++;
         } 
         if (v2.matches("\"X[0-9]+\"")) {
         v2 = "X" + x;
         x++;
         } 
         if (v3.matches("\"X[0-9]+\"")) {
         v3 = "X" + x;
         x++;
         } 
         features.put("feature_3valueToBeMentioned_" + v1 + "_" + v2 + "_" + v3, 1.0);
         }
         }
         }
         }*/
        /*
        
         for (String value1 : valuesToBeMentioned) {
         for (String value2 : valuesToBeMentioned) {
         if (!value1.equals(value2)) {
         ArrayList<String> arr = new ArrayList<>();
         arr.add(value1);
         arr.add(value2);
        
         x = 1;
         for (int i = 0; i < arr.size(); i++) {                        
         if (arr.get(i).matches("\"X[0-9]+\"")) {
         arr.set(i, "X" + x);
         x++;
         } 
         }
         Collections.sort(arr);
         String featureName = "feature_2valueToBeMentioned_";                    
         for (int i = 0; i < arr.size(); i++) {  
         featureName += arr.get(i) + "_";
         }
         features.put(featureName, 1.0);
         }
         }
         }
         for (String value1 : valuesToBeMentioned) {
         for (String value2 : valuesToBeMentioned) {
         for (String value3 : valuesToBeMentioned) {
         if (!value1.equals(value2) && !value1.equals(value3) && !value2.equals(value3)) {                        
         ArrayList<String> arr = new ArrayList<>();
         arr.add(value1);
         arr.add(value2);
         arr.add(value3);
        
         x = 1;
         for (int i = 0; i < arr.size(); i++) {                        
         if (arr.get(i).matches("\"X[0-9]+\"")) {
         arr.set(i, "X" + x);
         x++;
         } 
         }
         Collections.sort(arr);
         String featureName = "feature_3valueToBeMentioned_";                    
         for (int i = 0; i < arr.size(); i++) {  
         featureName += arr.get(i) + "_";
         }
         features.put(featureName, 1.0);
         }
         }
         }
         }
         */

        //Which values should be generated
        /*x = 1;
         for (String value : MR.getAttributes().get(generatedAttributes.get(a))) {
         if (value.matches("\"X[0-9]+\"")) {
         features.put("feature_v_" + x, 1.0);
         x++;
         } else {
         features.put("feature_v_" + value, 1.0);
         }
         }*/
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
         }
         //System.exit(0);
         return new Instance(features, costs);
         }
         }        //if (w == 12) {
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
         }
         //System.exit(0);
         return new Instance(features, costs);
         }
         }        //if (w == 12) {
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
         }
         //System.exit(0);
         return new Instance(features, costs);
         }
         }        //if (w == 12) {
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
         }
         //System.exit(0);*/
        //System.out.println(costs);
        return new Instance(features, costs);
    }

    public static boolean endsWith(ArrayList<String> phrase, ArrayList<String> subPhrase) {
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
}
