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
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import jdagger.JLOLS;
import similarity_measures.Levenshtein;
import similarity_measures.Rouge;
import simpleLM.SimpleLM;

/**
 *
 * @author Gerasimos Lampouras
 */
public class Bagel extends DatasetParser {

    HashMap<String, HashSet<String>> availableAttributeActions = new HashMap<>();
    HashMap<String, HashSet<String>> attributeValuePairs = new HashMap<>();
    HashMap<String, HashMap<MeaningRepresentation, HashSet<String>>> meaningReprs = new HashMap<>();
    HashMap<String, ArrayList<DatasetInstance>> datasetInstances = new HashMap<>();
    HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments = new HashMap<>();
    HashMap<ArrayList<Action>, Action> punctuationPatterns = new HashMap<>();
    ArrayList<String> predicates = new ArrayList<>();

    /**
     *
     */
    public ArrayList<Double> crossAvgArgDistances = new ArrayList<>();

    /**
     *
     */
    public ArrayList<Double> crossNIST = new ArrayList<>();

    /**
     *
     */
    public ArrayList<Double> crossBLEU = new ArrayList<>();

    /**
     *
     */
    public ArrayList<Double> crossBLEUSmooth = new ArrayList<>();
    static final int SEED = 13;
    boolean useAlignments = false;
    private HashMap<String, ArrayList<Instance>> predicateAttrTrainingData;
    private HashMap<String, HashMap<String, ArrayList<Instance>>> predicateWordTrainingData;

    SimpleLM wordLM;
    HashMap<String, SimpleLM> wordLMsPerPredicate = new HashMap<>();
    HashMap<String, SimpleLM> attrLMsPerPredicate = new HashMap<>();
    final static int THREADS_COUNT = Runtime.getRuntime().availableProcessors();
    static int fold = 0;

    /**
     *
     * @param args
     */
    public static void main(String[] args) {
        boolean useDAggerArg = false;
        boolean useLolsWord = true;

        fold = Integer.parseInt(args[0]);
        JLOLS.earlyStopMaxFurtherSteps = Integer.parseInt(args[1]);
        JLOLS.p = Double.parseDouble(args[2]);

        if (!args[3].isEmpty()
                && (args[3].equals("B")
                || args[3].equals("R")
                || args[3].equals("BC")
                || args[3].equals("RC")
                || args[3].equals("BRC")
                || args[3].equals("BR"))) {
            System.out.println("Using " + args[3] + " metric!");
            ActionSequence.metric = args[3];
        }

        Bagel bagel = new Bagel();
        bagel.runImitationLearning(useDAggerArg, useLolsWord);
    }

    /**
     *
     * @param useDAggerArg
     * @param useDAggerWord
     */
    @Override
    public void runImitationLearning(boolean useDAggerArg, boolean useDAggerWord) {
        averaging = true;
        shuffling = false;
        rounds = 10;
        initialTrainingParam = 1000.0;
        additionalTrainingParam = 1000.0;
        adapt = true;

        useLMs = true;
        useSubsetData = false;
        dataset = "Bagel";

        detailedResults = false;
        resetLists = true;

        File dataFile = new File("bagel_data/ACL10-inform-training.txt");

        if (resetLists || !loadLists()) {
            createLists(dataFile);
        }
        /*int counter = 0;
        for (String predicate : predicates) {
            ArrayList<DatasetInstance> datasetInstances = new ArrayList(this.datasetInstances.get(predicate));
            for (DatasetInstance in : datasetInstances) {
                System.out.println("BAGEL-" + counter + " \tinform(" + in.getMeaningRepresentation().getMRstr().replaceAll("X1", "X").replaceAll("X2", "X").replaceAll("X3", "X").replaceAll("X4", "X") +")");
                counter++;
            }
        }
        System.exit(0);*/
        for (String predicate : predicates) {
            evaluateDusek(predicate);
            System.exit(0);
        }
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
        //for (int f = 0; f < 10; f += 1) {
        //fold = f;
        for (int f = fold; f < fold + 1; f += 1) {
            System.out.println("======================================================");
            System.out.println("======================================================");
            System.out.println("                      ===F=" + f + "===                      ");
            System.out.println("======================================================");
            System.out.println("======================================================");

            for (String predicate : predicates) {
                randomGen = new Random(SEED);

                ArrayList<DatasetInstance> datasetInstances = new ArrayList<DatasetInstance>(this.datasetInstances.get(predicate));

                ArrayList<DatasetInstance> testingData = new ArrayList<>();
                trainingData = new ArrayList<>();
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
                createNaiveAlignments(trainingData);

                if (resetLists || !loadLMs()) {
                    ArrayList<ArrayList<String>> LMWordTraining = new ArrayList<>();

                    HashMap<String, ArrayList<ArrayList<String>>> LMWordTrainingPerPredicate = new HashMap<>();
                    HashMap<String, ArrayList<ArrayList<String>>> LMAttrTrainingPerPredicate = new HashMap<>();
                    for (DatasetInstance di : trainingData) {
                        if (!LMWordTrainingPerPredicate.containsKey(di.getMeaningRepresentation().getPredicate())) {
                            LMWordTrainingPerPredicate.put(di.getMeaningRepresentation().getPredicate(), new ArrayList<ArrayList<String>>());
                            LMAttrTrainingPerPredicate.put(di.getMeaningRepresentation().getPredicate(), new ArrayList<ArrayList<String>>());
                        }
                        HashSet<ArrayList<Action>> seqs = new HashSet<>();
                        seqs.add(di.getTrainRealization());
                        seqs.addAll(di.getEvalRealizations());
                        for (ArrayList<Action> seq : seqs) {
                            ArrayList<String> wordSeq = new ArrayList<>();
                            ArrayList<String> attrSeq = new ArrayList<>();

                            wordSeq.add("@@");
                            wordSeq.add("@@");
                            attrSeq.add("@@");
                            attrSeq.add("@@");
                            for (int i = 0; i < seq.size(); i++) {
                                if (!seq.get(i).getAttribute().equals(Action.TOKEN_END)
                                        && !seq.get(i).getWord().equals(Action.TOKEN_END)) {
                                    wordSeq.add(seq.get(i).getWord());
                                }
                                if (attrSeq.isEmpty()) {
                                    //if (!seq.get(i).getAttribute().equals(Action.TOKEN_END)) {
                                    attrSeq.add(seq.get(i).getAttribute());
                                    //}
                                } else if (!attrSeq.get(attrSeq.size() - 1).equals(seq.get(i).getAttribute())) {
                                    //if (!seq.get(i).getAttribute().equals(Action.TOKEN_END)) {
                                    attrSeq.add(seq.get(i).getAttribute());
                                    //}
                                }
                            }
                            wordSeq.add(Action.TOKEN_END);
                            LMWordTraining.add(wordSeq);
                            LMWordTrainingPerPredicate.get(di.getMeaningRepresentation().getPredicate()).add(wordSeq);
                            LMAttrTrainingPerPredicate.get(di.getMeaningRepresentation().getPredicate()).add(attrSeq);
                        }
                    }
                    wordLM = new SimpleLM(3);
                    wordLM.trainOnStrings(LMWordTraining);

                    wordLMsPerPredicate = new HashMap<>();
                    attrLMsPerPredicate = new HashMap<>();
                    for (String pred : LMWordTrainingPerPredicate.keySet()) {
                        SimpleLM simpleWordLM = new SimpleLM(3);
                        simpleWordLM.trainOnStrings(LMWordTrainingPerPredicate.get(pred));
                        wordLMsPerPredicate.put(pred, simpleWordLM);

                        SimpleLM simpleAttrLM = new SimpleLM(3);
                        simpleAttrLM.trainOnStrings(LMAttrTrainingPerPredicate.get(pred));
                        attrLMsPerPredicate.put(pred, simpleAttrLM);
                    }
                    writeLMs();
                }

                availableAttributeActions.get(predicate).add(Action.TOKEN_END);
                HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions = new HashMap<>();
                for (DatasetInstance DI : trainingData) {
                    HashSet<String> mentionedAttributes = new HashSet<>();
                    HashSet<Action> mentionedWords = new HashSet<>();
                    for (ArrayList<Action> realization : DI.getEvalRealizations()) {
                        for (Action a : realization) {
                            if (!a.getAttribute().equals(Action.TOKEN_END)) {
                                String attr = a.getAttribute().substring(0, a.getAttribute().indexOf('='));
                                mentionedAttributes.add(attr);
                                if (!a.getWord().equals(Action.TOKEN_START)
                                        && !a.getWord().equals(Action.TOKEN_END)
                                        && !a.getWord().matches("([,.?!;:'])")) {
                                    mentionedWords.add(a);
                                }
                                if (attr.equals("[]")) {
                                    System.out.println("RR " + realization);
                                    System.out.println("RR " + a);
                                    System.exit(0);
                                }
                            }
                        }
                        if (!availableWordActions.containsKey(predicate)) {
                            availableWordActions.put(predicate, new HashMap<String, HashSet<Action>>());
                        }
                        for (String attribute : mentionedAttributes) {
                            if (!availableWordActions.get(predicate).containsKey(attribute)) {
                                availableWordActions.get(predicate).put(attribute, new HashSet<Action>());
                                availableWordActions.get(predicate).get(attribute).add(new Action(Action.TOKEN_END, attribute));
                            }
                            for (Action a : mentionedWords) {
                                if (a.getWord().startsWith(Action.TOKEN_X)) {
                                    if (a.getWord().substring(3, a.getWord().lastIndexOf('_')).toLowerCase().trim().equals(attribute)) {
                                        availableWordActions.get(predicate).get(attribute).add(new Action(a.getWord(), attribute));
                                    }
                                } else {
                                    availableWordActions.get(predicate).get(attribute).add(new Action(a.getWord(), attribute));
                                }
                            }
                        }
                    }
                }

                if (resetLists || !loadTrainingData()) {
                    Object[] results = createTrainingDatasets(trainingData, availableWordActions);
                    if (results[0] instanceof HashMap) {
                        predicateAttrTrainingData = (HashMap<String, ArrayList<Instance>>) results[0];
                    }
                    if (results[1] instanceof HashMap) {
                        predicateWordTrainingData = (HashMap<String, HashMap<String, ArrayList<Instance>>>) results[1];
                    }
                    writeTrainingData();
                }

                boolean setToGo = true;
                if (predicateWordTrainingData.isEmpty() || predicateAttrTrainingData.isEmpty()) {
                    setToGo = false;
                }

                if (setToGo) {
                    JLOLS JDWords = new JLOLS(this);
                    /*Object[] LOLSResults = */
                    JDWords.runLOLS(availableAttributeActions, trainingData, predicateAttrTrainingData, predicateWordTrainingData, availableWordActions, valueAlignments, wordRefRolloutChance, testingData, detailedResults);
                    //JDWords.runLOLS(fold, availableAttributeActions.get(predicate), trainingData, predicateAttrTrainingData.get(predicate), predicateWordTrainingData.get(predicate), availableWordActions.get(predicate), valueAlignments, wordRefRolloutChance, testingData);

                    //classifiersAttrs.put(predicate, (JAROW) LOLSResults[0]);
                    //classifiersWords.put(predicate, (HashMap<String, JAROW>) LOLSResults[1]);

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
                    //evaluateGeneration(classifiersAttrs.get(predicate), classifiersWords.get(predicate), trainingData, testingData, availableWordActions, nGrams, predicate, true, 10000);
                }
                //RANDOM DATA SPLIT
                //}
            }
            //System.exit(0);
        }/*
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
        System.out.println("==========================");*/
        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);        //evaluateGenerationDoc(classifiersWords, file);
        //printEvaluation("robocup_data\\results", -1);
    }

    /**
     *
     * @param classifierAttrs
     * @param classifierWords
     * @param testingData
     * @param availableAttributeActions
     * @param availableWordActions
     * @param printResults
     * @param epoch
     * @param detailedResults
     * @return
     */
    public Double evaluateGeneration(HashMap<String, JAROW> classifierAttrs, HashMap<String, HashMap<String, JAROW>> classifierWords, ArrayList<DatasetInstance> testingData, HashMap<String, HashSet<String>> availableAttributeActions, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions, boolean printResults, int epoch, boolean detailedResults) {
        System.out.println("Evaluate argument generation");

        int totalArgDistance = 0;
        ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
        HashMap<String, ArrayList<Action>> generationActions = new HashMap<>();
        HashMap<ArrayList<Action>, DatasetInstance> generationActionsMap = new HashMap<>();
        ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
        HashMap<String, ArrayList<String>> finalReferencesWordSequences = new HashMap<>();
        ArrayList<String> allPredictedWordSequences = new ArrayList<>();
        ArrayList<String> predictedWordSequencesMRs = new ArrayList<>();
        HashMap<String, Double> attrCoverage = new HashMap<>();
        HashSet<HashMap<String, HashSet<String>>> mentionedAttrs = new HashSet<>();

        for (DatasetInstance di : testingData) {
            String predicate = di.getMeaningRepresentation().getPredicate();
            ArrayList<Action> predictedActionList = new ArrayList<>();
            ArrayList<Action> predictedWordList = new ArrayList<>();

            //PHRASE GENERATION EVALUATION
            String predictedAttr = "";
            ArrayList<String> predictedAttrValues = new ArrayList<>();

            HashSet<String> attrValuesToBeMentioned = new HashSet<>();
            HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
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
            while (!predictedAttr.equals(Action.TOKEN_END) && predictedAttrValues.size() < maxAttrRealizationSize) {
                if (!predictedAttr.isEmpty()) {
                    attrValuesToBeMentioned.remove(predictedAttr);
                }
                if (!attrValuesToBeMentioned.isEmpty()) {
                    Instance attrTrainingVector = createAttrInstance(predicate, "@TOK@", predictedAttrValues, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableAttributeActions);

                    if (attrTrainingVector != null) {
                        Prediction predictAttr = classifierAttrs.get(predicate).predict(attrTrainingVector);
                        predictedAttr = predictAttr.getLabel().trim();
                        if (!classifierAttrs.get(predicate).getCurrentWeightVectors().keySet().containsAll(di.getMeaningRepresentation().getAttributes().keySet())) {
                            System.out.println("MR ATTR NOT IN CLASSIFIERS");
                            System.out.println(classifierAttrs.get(predicate).getCurrentWeightVectors().keySet());
                            System.out.println(di.getMeaningRepresentation().getAbstractMR());
                        }
                        String predictedValue = "";
                        if (!predictedAttr.equals(Action.TOKEN_END)) {
                            predictedValue = chooseNextValue(predictedAttr, attrValuesToBeMentioned);

                            HashSet<String> rejectedAttrs = new HashSet<String>();
                            while ((predictedValue.isEmpty() || predictedValue.equals("placetoeat")) && (!predictedAttr.equals(Action.TOKEN_END) || predictedAttrValues.isEmpty())) {
                                rejectedAttrs.add(predictedAttr);

                                predictedAttr = Action.TOKEN_END;
                                double maxScore = -Double.MAX_VALUE;
                                for (String attr : predictAttr.getLabel2Score().keySet()) {
                                    if (!rejectedAttrs.contains(attr)
                                            && (Double.compare(predictAttr.getLabel2Score().get(attr), maxScore) > 0)) {
                                        maxScore = predictAttr.getLabel2Score().get(attr);
                                        predictedAttr = attr;
                                    }
                                }
                                if (!predictedAttr.equals(Action.TOKEN_END)) {
                                    predictedValue = chooseNextValue(predictedAttr, attrValuesToBeMentioned);
                                }
                            }
                        }

                        if (!predictedAttr.equals(Action.TOKEN_END)) {
                            predictedAttr += "=" + predictedValue;
                        }
                        predictedAttrValues.add(predictedAttr);
                        if (!predictedAttr.isEmpty()) {
                            attrValuesAlreadyMentioned.add(predictedAttr);
                            attrValuesToBeMentioned.remove(predictedAttr);
                        }
                    } else {
                        predictedAttr = Action.TOKEN_END;
                        predictedAttrValues.add(predictedAttr);
                    }
                } else {
                    predictedAttr = Action.TOKEN_END;
                    predictedAttrValues.add(predictedAttr);
                }
            }

            //WORD SEQUENCE EVALUATION
            predictedAttr = "";
            ArrayList<String> predictedAttributes = new ArrayList<>();

            attrValuesToBeMentioned = new HashSet<>();
            attrValuesAlreadyMentioned = new HashSet<>();
            HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
            for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                for (String value : di.getMeaningRepresentation().getAttributes().get(attribute)) {
                    attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
                }
                valuesToBeMentioned.put(attribute, new ArrayList<>(di.getMeaningRepresentation().getAttributes().get(attribute)));
            }
            HashSet<String> attrValuesToBeMentionedCopy = new HashSet<>(attrValuesToBeMentioned);

            int a = -1;
            for (String attrValue : predictedAttrValues) {
                a++;
                if (!attrValue.equals(Action.TOKEN_END)) {
                    String attribute = attrValue.split("=")[0];
                    predictedAttributes.add(attrValue);

                    //GENERATE PHRASES
                    if (!attribute.equals(Action.TOKEN_END)) {
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
                            while (!predictedWord.equals(Action.TOKEN_END) && predictedWordList.size() < maxWordRealizationSize) {
                                ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                                for (int i = 0; i < predictedAttributes.size() - 1; i++) {
                                    predictedAttributesForInstance.add(predictedAttributes.get(i));
                                }
                                if (!predictedAttributes.get(predictedAttributes.size() - 1).equals(attrValue)) {
                                    predictedAttributesForInstance.add(predictedAttributes.get(predictedAttributes.size() - 1));
                                }
                                ArrayList<String> nextAttributesForInstance = new ArrayList<>(predictedAttrValues.subList(a + 1, predictedAttrValues.size()));
                                Instance wordTrainingVector = createWordInstance(predicate, new Action("@TOK@", attrValue), predictedAttributesForInstance, predictedActionList, nextAttributesForInstance, attrValuesAlreadyMentioned, attrValuesToBeMentioned, isValueMentioned, availableWordActions.get(predicate));

                                if (wordTrainingVector != null) {
                                    if (classifierWords.get(predicate).get(attribute) != null) {
                                        Prediction predictWord = classifierWords.get(predicate).get(attribute).predict(wordTrainingVector);
                                        if (predictWord.getLabel() != null) {
                                            predictedWord = predictWord.getLabel().trim();
                                            while (predictedWord.equals(Action.TOKEN_END) && predictedActionList.get(predictedActionList.size() - 1).getWord().equals(Action.TOKEN_END)) {
                                                double maxScore = -Double.MAX_VALUE;
                                                for (String word : predictWord.getLabel2Score().keySet()) {
                                                    if (!word.equals(Action.TOKEN_END)
                                                            && (Double.compare(predictWord.getLabel2Score().get(word), maxScore) > 0)) {
                                                        maxScore = predictWord.getLabel2Score().get(word);
                                                        predictedWord = word;
                                                    }
                                                }
                                            }
                                            predictedActionList.add(new Action(predictedWord, attrValue));
                                            if (!predictedWord.equals(Action.TOKEN_END)) {
                                                subPhrase.add(predictedWord);
                                                predictedWordList.add(new Action(predictedWord, attrValue));
                                            }
                                        } else {
                                            predictedWord = Action.TOKEN_END;
                                            predictedActionList.add(new Action(predictedWord, attrValue));
                                        }
                                    } else {
                                        predictedWord = Action.TOKEN_END;
                                        predictedActionList.add(new Action(predictedWord, attrValue));
                                    }
                                }
                                if (!isValueMentioned) {
                                    if (!predictedWord.equals(Action.TOKEN_END)) {
                                        if (predictedWord.startsWith(Action.TOKEN_X)
                                                && (valueTBM.matches("\"[xX][0-9]+\"")
                                                || valueTBM.matches("[xX][0-9]+")
                                                || valueTBM.startsWith(Action.TOKEN_X))) {
                                            isValueMentioned = true;
                                        } else if (!predictedWord.startsWith(Action.TOKEN_X)
                                                && !(valueTBM.matches("\"[xX][0-9]+\"")
                                                || valueTBM.matches("[xX][0-9]+")
                                                || valueTBM.startsWith(Action.TOKEN_X))) {
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
                                if (!predictedWord.startsWith(Action.TOKEN_X)) {
                                    for (String attrValueTBM : attrValuesToBeMentioned) {
                                        if (attrValueTBM.contains("=")) {
                                            String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                            if (!(value.matches("\"[xX][0-9]+\"")
                                                    || value.matches("[xX][0-9]+")
                                                    || value.startsWith(Action.TOKEN_X))) {
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
                                    && !predictedActionList.get(predictedActionList.size() - 1).getWord().equals(Action.TOKEN_END)) {
                                predictedWord = Action.TOKEN_END;
                                predictedActionList.add(new Action(predictedWord, attrValue));
                            }
                        } else {
                            String predictedWord = Action.TOKEN_END;
                            predictedActionList.add(new Action(predictedWord, attrValue));
                        }
                    }
                }
            }
            ArrayList<String> predictedAttrs = new ArrayList<>();
            for (String attributeValuePair : predictedAttrValues) {
                predictedAttrs.add(attributeValuePair.split("=")[0]);
            }

            String predictedWordSequence = postProcessWordSequence(di, predictedActionList);

            ArrayList<String> predictedAttrList = getPredictedAttrList(predictedActionList);
            if (attrValuesToBeMentionedCopy.size() != 0.0) {
                double missingAttrs = 0.0;
                for (String attr : attrValuesToBeMentionedCopy) {
                    if (!predictedAttrList.contains(attr)) {
                        missingAttrs += 1.0;
                    }
                }
                double attrSize = (double) attrValuesToBeMentionedCopy.size();
                attrCoverage.put(predictedWordSequence, missingAttrs / attrSize);
            }

            if (!mentionedAttrs.contains(di.getMeaningRepresentation().getAttributes())) {
                allPredictedWordSequences.add(predictedWordSequence);
                predictedWordSequencesMRs.add(di.getMeaningRepresentation().getMRstr());
                mentionedAttrs.add(di.getMeaningRepresentation().getAttributes());
            }

            Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedWordSequence.toLowerCase()));
            ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
            generations.add(tran);
            generationActions.put(predictedWordSequence, predictedActionList);
            generationActionsMap.put(predictedActionList, di);

            ArrayList<Sequence<IString>> references = new ArrayList<>();
            ArrayList<String> referencesStrings = new ArrayList<>();
            for (String ref : di.getEvalReferences()) {
                referencesStrings.add(ref);
                references.add(IStrings.tokenize(NISTTokenizer.tokenize(ref)));
            }

            //System.out.println(predictedWordSequence + ">>>"+ referencesStrings);
            finalReferencesWordSequences.put(predictedWordSequence, referencesStrings);
            finalReferences.add(references);

            //EVALUATE ATTRIBUTE SEQUENCE
            HashSet<ArrayList<String>> goldAttributeSequences = new HashSet<>();
            for (DatasetInstance di2 : testingData) {
                if (di2.getMeaningRepresentation().getMRstr().equals(di.getMeaningRepresentation().getMRstr())) {
                    goldAttributeSequences.addAll(di2.getEvalMentionedAttributeSequences().values());
                }
            }

            //for (ArrayList<String> goldArgs : abstractMeaningReprs.get(predicate).get(mr).values()) {
            int minTotArgDistance = Integer.MAX_VALUE;
            for (ArrayList<String> goldArgs : goldAttributeSequences) {
                int totArgDistance = 0;
                HashSet<Integer> matchedPositions = new HashSet<>();
                for (int i = 0; i < predictedAttrs.size(); i++) {
                    if (!predictedAttrs.get(i).equals(Action.TOKEN_START)
                            && !predictedAttrs.get(i).equals(Action.TOKEN_END)) {
                        int minArgDistance = Integer.MAX_VALUE;
                        int minArgPos = -1;
                        for (int j = 0; j < goldArgs.size(); j++) {
                            if (!matchedPositions.contains(j) && goldArgs.get(j).equals(predictedAttrs.get(i))) {
                                int argDistance = Math.abs(j - i);

                                if (argDistance < minArgDistance) {
                                    minArgDistance = argDistance;
                                    minArgPos = j;
                                }
                            }
                        }

                        if (minArgPos == -1) {
                            totArgDistance += 100;
                            /*System.out.println("ADDITIONAL ARG: " + predictedAttrs.get(i));
                            System.out.println(goldArgs);
                            System.out.println(predictedAttrs);
                            System.out.println(predictedAttrValues);*/
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
                    if (!goldArg.equals(Action.TOKEN_END)) {
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
        /*for (DatasetInstance di : trainingData) {
            String trainRef = (new ActionSequence(di.getTrainRealization())).getWordSequenceToNoPunctString().replaceAll("@x@.+?[\\s\\d]", "x ").replaceAll("  ", " ").trim();
            for (String gen : allPredictedWordSequences) {
                /*System.out.println(gen.replaceAll("@x@.+?[\\s\\d]", "x ").replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim());
                System.out.println(trainRef);
                System.out.println("==========================");*/
 /*if (gen.replaceAll("@x@.+?[\\s\\d]", "x ").replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim().equals(trainRef)) {

                    System.out.println("++++==========================");
                    System.out.println(gen);
                    System.out.println(di.getTrainRealization());
                    System.out.println(generationActions.get(gen));
                    System.out.println("++++==========================");
                }
            }
        }*/
        crossAvgArgDistances.add(totalArgDistance / (double) testingData.size());

        NISTMetric NIST = new NISTMetric(finalReferences);
        BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
        BLEUMetric BLEUsmooth = new BLEUMetric(finalReferences, 4, true);
        Double nistScore = NIST.score(generations);
        Double bleuScore = BLEU.score(generations);
        Double bleuSmoothScore = BLEUsmooth.score(generations);

        double finalCoverage = 0.0;
        for (double c : attrCoverage.values()) {
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
        //System.out.println("g: " + generations);
        //System.out.println("attr: " + predictedAttrLists);
        System.out.println("BLEU smooth: \t" + bleuSmoothScore);

        double avgRougeScore = 0.0;
        String detailedRes = "";
        int a = 0;
        for (String predictedString : allPredictedWordSequences) {
            double maxRouge = 0.0;
            for (String ref : finalReferencesWordSequences.get(predictedString)) {
                double rouge = Rouge.ROUGE_N(predictedString, ref, 4);
                if (rouge > maxRouge) {
                    maxRouge = rouge;
                }
            }
            avgRougeScore += maxRouge;

            double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedString, finalReferencesWordSequences.get(predictedString), 4);
            double cover = 1.0 - attrCoverage.get(predictedString);
            double avg = (BLEUSmooth + maxRouge + cover) / 3.0;
            double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

            /*System.out.println("===============");
            System.out.println(predictedString);
            ArrayList<String> sort = new ArrayList<String>(finalReferencesWordSequences.get(predictedString));
            Collections.sort(sort);
            System.out.println(sort);
            System.out.println("BLEUSmooth:" + BLEUSmooth);*/
            detailedRes += predictedString + "\t" + finalReferencesWordSequences.get(predictedString) + "\t" + predictedWordSequencesMRs.get(a) + "\t" + BLEUSmooth + "\t" + maxRouge + "\t" + cover + "\t" + avg + "\t" + harmonicMean + "|";
            a++;
        }
        System.out.println("ROUGE: \t" + (avgRougeScore / (double) allPredictedWordSequences.size()));
        System.out.println();
        System.out.println(detailedRes);

        if (printResults) {
            BufferedWriter bw = null;
            File f = null;
            try {
                f = new File("BAGELTextsAt " + fold + "FoldAfter" + (epoch + 1) + "epochs" + "_" + JLOLS.earlyStopMaxFurtherSteps + "_" + JLOLS.p + ".txt");
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
            for (int i = 0; i < allPredictedWordSequences.size(); i++) {
                try {
                    //Grafoume to String sto arxeio
                    bw.write("MR," + predictedWordSequencesMRs.get(i) + ",");
                    bw.write("BAGEL,");
                    bw.write(allPredictedWordSequences.get(i) + ",");

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

    /**
     *
     * @param predicate
     * @return
     */
    public Double evaluateDusek(String predicate) {
        System.out.println("Evaluate Dusek");

        ArrayList<Double> BLEUs = new ArrayList<>();
        ArrayList<Double> BLEUSmooths = new ArrayList<>();
        ArrayList<Double> NISTs = new ArrayList<>();
        ArrayList<Double> ROUGEs = new ArrayList<>();
        for (int f = 0; f < 10; f++) {
            HashMap<String, DatasetInstance> dusekGenerations = new HashMap<>();
            HashMap<Integer, HashSet<String>> testMap = new HashMap<>();
            String outPath = "bagel_data/Ondrej data/basic_perceptron/cv0" + f + "/out-text.sgm";
            ArrayList<String> outTexts = new ArrayList<String>();
            try (BufferedReader br = new BufferedReader(new FileReader(outPath))) {
                String line;
                String text;
                while ((line = br.readLine()) != null) {
                    if (line.startsWith("<seg")) {
                        text = line.substring(line.indexOf(">") + 1, line.lastIndexOf("<")).replaceAll("\\.", " \\.").replaceAll("  ", " ");
                        outTexts.add(text.toLowerCase());
                    }
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            }
            int i = 0;
            String mrPath = "bagel_data/Ondrej data/basic_perceptron/cv0" + f + "/test-das.sgm";
            try (BufferedReader br = new BufferedReader(new FileReader(mrPath))) {
                String line;
                String MRstr;
                while ((line = br.readLine()) != null) {
                    if (line.startsWith("<seg")) {
                        MRstr = line.substring(line.indexOf(">") + 1, line.lastIndexOf("<"));
                        int xInd = MRstr.indexOf("X-");
                        int x = 1;
                        while (xInd != -1) {
                            String xStr = MRstr.substring(xInd, MRstr.indexOf(")", xInd));
                            MRstr = MRstr.replaceFirst(xStr, "\"X" + x + "\"");
                            x++;

                            xInd = MRstr.indexOf("X-");
                        }
                        MRstr = MRstr.replaceAll("inform\\(", "").replaceAll("&", ",").replaceAll("\\)", "");
                        DatasetInstance corrDi = null;

                        for (DatasetInstance di : datasetInstances.get(predicate)) {
                            if (di.getMeaningRepresentation().getMRstr().equals(MRstr)) {
                                corrDi = di;
                            }
                        }
                        dusekGenerations.put(outTexts.get(i), corrDi);
                        i++;
                    }
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            }

            i = 0;
            String testPath = "bagel_data/Ondrej data/basic_perceptron/cv0" + f + "/test-conc.sgm";
            try (BufferedReader br = new BufferedReader(new FileReader(testPath))) {
                String line;
                String test;
                while ((line = br.readLine()) != null) {
                    if (line.startsWith("<seg")) {
                        test = line.substring(line.indexOf(">") + 1, line.lastIndexOf("<"));

                        if (!testMap.containsKey(outTexts.get(i))) {
                            testMap.put(i, new HashSet<String>());
                        }
                        testMap.get(i).add(test);
                        i++;
                        if (i >= outTexts.size()) {
                            i = 0;
                        }
                    }
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            }

            ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
            ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
            HashMap<String, ArrayList<String>> finalReferencesStrings = new HashMap<>();
            for (int j = 0; j < outTexts.size(); j++) {
                String predictedString = outTexts.get(j);
                Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedString.toLowerCase()));
                ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
                generations.add(tran);

                ArrayList<Sequence<IString>> references = new ArrayList<>();
                ArrayList<String> referencesStrings = new ArrayList<>();
                /*for (ArrayList<Action> realization : dusekGenerations.get(predictedString).getEvalRealizations()) {
                String cleanedWords = "";
                for (Action nlWord : realization) {
                if (!nlWord.equals(new Action(Action.TOKEN_START, "", ""))
                && !nlWord.equals(new Action(Action.TOKEN_END, "", ""))) {
                if (nlWord.getWord().startsWith(Action.TOKEN_X)) {
                cleanedWords += "x ";
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
                referencesStrings.add(cleanedWords);
                references.add(IStrings.tokenize(NISTTokenizer.tokenize(cleanedWords)));
                }*/

                //for (String realization : testMap.get(j)) {
                for (String realization : dusekGenerations.get(predictedString).getEvalReferences()) {
                    referencesStrings.add(realization);
                    references.add(IStrings.tokenize(NISTTokenizer.tokenize(realization)));
                }
                finalReferencesStrings.put(predictedString, referencesStrings);
                finalReferences.add(references);
                System.out.println(predictedString + "\t" + referencesStrings);
            }
            NISTMetric NIST = new NISTMetric(finalReferences);
            BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
            BLEUMetric BLEUsmooth = new BLEUMetric(finalReferences, 4, true);
            Double nistScore = NIST.score(generations);
            Double bleuScore = BLEU.score(generations);
            Double bleuSmoothScore = BLEUsmooth.score(generations);

            NISTs.add(nistScore);
            BLEUs.add(bleuScore);
            BLEUSmooths.add(bleuSmoothScore);

            double avgRougeScore = 0.0;
            for (String predictedString : dusekGenerations.keySet()) {
                double maxRouge = 0.0;
                for (String ref : finalReferencesStrings.get(predictedString)) {
                    double rouge = Rouge.ROUGE_N(predictedString.toLowerCase(), ref.toLowerCase(), 4);
                    if (rouge > maxRouge) {
                        maxRouge = rouge;
                    }
                }
                avgRougeScore += maxRouge;
            }
            ROUGEs.add((avgRougeScore / (double) dusekGenerations.keySet().size()));
        }
        /*HashSet<String> shown = new HashSet<String>();
        for (DatasetInstance di : datasetInstances.get(predicate)) {
            if (!shown.contains(di.getMeaningRepresentation().getMRstr())) {
                shown.add(di.getMeaningRepresentation().getMRstr());
                String refs = "";
                for (String ref : di.getEvalReferences()) {
                    refs += ref + ";";
                }
                System.out.println("inform(" + di.getMeaningRepresentation().getMRstr() +")\t" + refs);
            }
        }*/
        Double avgBLUEs = 0.0;
        Double avgBLUESmooths = 0.0;
        Double avgNISTs = 0.0;
        Double avgROUGEs = 0.0;
        for (Double d : BLEUs) {
            avgBLUEs += d;
        }
        avgBLUEs /= BLEUs.size();
        for (Double d : BLEUSmooths) {
            avgBLUESmooths += d;
        }
        avgBLUESmooths /= BLEUSmooths.size();
        for (Double d : NISTs) {
            avgNISTs += d;
        }
        avgNISTs /= NISTs.size();
        for (Double d : ROUGEs) {
            avgROUGEs += d;
        }
        avgROUGEs /= ROUGEs.size();

        System.out.println("NIST: \t" + avgNISTs);
        System.out.println("BLEU: \t" + avgBLUEs);
        //System.out.println("g: " + generations);
        //System.out.println("attr: " + predictedAttrLists);
        System.out.println("BLEU smooth: \t" + avgBLUESmooths);

        System.out.println("ROUGE: \t" + avgROUGEs);
        System.out.println();

        return 0.0;
    }

    /**
     *
     * @param dataFile
     */
    public void createLists(File dataFile) {
        predicates = new ArrayList<>();
        availableAttributeActions = new HashMap<>();
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
                        if (!availableAttributeActions.containsKey(previousPredicate)) {
                            availableAttributeActions.put(previousPredicate, new HashSet<String>());
                        }
                        if (!attributeValuePairs.containsKey(previousPredicate)) {
                            attributeValuePairs.put(previousPredicate, new HashSet<String>());
                        }
                        if (!meaningReprs.containsKey(previousPredicate)) {
                            meaningReprs.put(previousPredicate, new HashMap<MeaningRepresentation, HashSet<String>>());
                        }
                        if (!datasetInstances.containsKey(previousPredicate)) {
                            datasetInstances.put(previousPredicate, new ArrayList<DatasetInstance>());
                        }
                    }

                    line = line.substring(line.indexOf("(") + 1, line.lastIndexOf(")"));

                    String MRstr = line;
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
                        availableAttributeActions.get(previousPredicate).add(subArg[0].toLowerCase());
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
                        if (!availableAttributeActions.containsKey(previousPredicate)) {
                            availableAttributeActions.put(previousPredicate, new HashSet<String>());
                        }
                        if (!meaningReprs.containsKey(previousPredicate)) {
                            meaningReprs.put(previousPredicate, new HashMap<MeaningRepresentation, HashSet<String>>());
                        }
                        if (!datasetInstances.containsKey(previousPredicate)) {
                            datasetInstances.put(previousPredicate, new ArrayList<DatasetInstance>());
                        }
                    }

                    line = line.substring(line.indexOf("(") + 1, line.lastIndexOf(")"));
                    String MRstr = line;

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
                        if (!availableAttributeActions.get(previousPredicate).contains(attr)) {
                            availableAttributeActions.get(previousPredicate).add(attr);
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
                            value = Action.TOKEN_X + attr + "_" + index;
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
                                        alignedRealization.add(Action.TOKEN_PUNCT);
                                    } else if (isEmptyAttr) {
                                        alignedRealization.add("[]");
                                    } else {
                                        alignedRealization.add(mentionedAttribute);
                                    }
                                }
                                //if (!words[i].trim().matches("[,.?!;:']")) {
                                if (words[i].trim().equalsIgnoreCase("x")) {
                                    //realization.add(Action.TOKEN_X + mentionedAttribute + "_" + (attributeXCount.get(mentionedAttribute) - 1));
                                    realization.add(Action.TOKEN_X + mentionedAttribute + "_" + (attributeXCount.get(mentionedAttribute) - 1));
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

                    mentionedValueSequence.add(Action.TOKEN_END);
                    mentionedAttributeSequence.add(Action.TOKEN_END);

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
                            } else if (!alignedRealization.get(i).equals(Action.TOKEN_PUNCT)) {
                                previousAttr = alignedRealization.get(i);
                            } else {
                                previousAttr = "";
                            }
                        }
                    } else {
                        for (String word : realization) {
                            if (word.trim().matches("[,.?!;:']")) {
                                alignedRealization.add(Action.TOKEN_PUNCT);
                            } else {
                                alignedRealization.add("[]");
                            }
                        }
                    }

                    //Calculate alignments
                    HashMap<String, HashMap<String, Double>> alignments = new HashMap<>();
                    for (String attr : previousAMR.getAttributes().keySet()) {
                        for (String value : previousAMR.getAttributes().get(attr)) {
                            if (!value.equals("name=none") && !(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+") || value.startsWith(Action.TOKEN_X))) {
                                alignments.put(value, new HashMap<String, Double>());
                                //For all ngrams
                                for (int n = 1; n < realization.size(); n++) {
                                    //Calculate all alignment similarities
                                    for (int i = 0; i <= realization.size() - n; i++) {
                                        boolean pass = true;
                                        for (int j = 0; j < n; j++) {
                                            if (realization.get(i + j).startsWith(Action.TOKEN_X)
                                                    || alignedRealization.get(i + j).equals(Action.TOKEN_PUNCT)
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

                                            Double distance = Levenshtein.getSimilarity(value.toLowerCase(), compare.toLowerCase(), true);
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
                        } else if (!alignedRealization.get(i).equals(Action.TOKEN_PUNCT)) {
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
                        } else if (!alignedRealization.get(i).equals(Action.TOKEN_PUNCT)) {
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
                    if (!previousAlignment.isEmpty() && !currentAlignment.equals(previousAlignment) && !previousAlignment.equals(Action.TOKEN_PUNCT)) {
                    realization.add(i, Action.TOKEN_END);
                    alignedRealization.add(i, previousAlignment);                            
                    i++;                            
                    }
                    previousAlignment = currentAlignment;
                    }
                    if (!previousAlignment.equals(Action.TOKEN_PUNCT)) {
                    realization.add(Action.TOKEN_END);
                    alignedRealization.add(previousAlignment);
                    }
                    System.out.println("====");
                    System.out.println(realization);
                    System.out.println(alignedRealization);
                    System.out.println("===============");*/

 /*for (int i = 0; i < alignedRealization.size(); i++) {
                    if (alignedRealization.get(i).equals(Action.TOKEN_PUNCT)) {
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

                    /*String w = "";
                    for (Action act : realizationActions) {
                        w += act.getWord()+ " ";
                    }
                    if (w.trim().equals("@x@name_0 is a restaurant in @x@area_0 near to @x@near_0 and @x@near_1 in the expensive pricerange .")) {
                        System.out.println(w.trim());
                        System.out.println(previousAMR.getMRstr());
                    }*/
                    //boolean existing = false;
                    ArrayList<DatasetInstance> existingDIs = new ArrayList<>();
                    for (DatasetInstance existingDI : datasetInstances.get(previousPredicate)) {
                        /*if (w.trim().equals("@x@name_0 is a restaurant in @x@area_0 near to @x@near_0 and @x@near_1 in the expensive pricerange .")) {                   
                            for (ArrayList<Action> r : existingDI.getEvalRealizations()) {
                                String l = "";
                                for (Action act : r) {
                                    l += act.getWord()+ " ";
                                }
                                if (l.trim().equals("@x@name_0 is a restaurant in the expensive price range in the @x@area_0 area near @x@near_0 and @x@near_1 .")) {
                                    System.out.println("BEING MERGED???");
                                    System.out.println(existingDI.getMeaningRepresentation().getMRstr());
                                    System.out.println(previousAMR.getMRstr());
                                }
                            }
                        }*/
                        if (existingDI.getMeaningRepresentation().getMRstr().equals(previousAMR.getMRstr())) {
                            //existing = true;
                            existingDI.mergeDatasetInstance(mentionedValueSequence, mentionedAttributeSequence, realizationActions);
                            existingDIs.add(existingDI);
                        }
                    }
                    //if (!existing) {
                    DatasetInstance DI = new DatasetInstance(previousAMR, mentionedValueSequence, mentionedAttributeSequence, realizationActions);
                    for (DatasetInstance existingDI : existingDIs) {
                        DI.mergeDatasetInstance(existingDI.getEvalMentionedValueSequences(), existingDI.getEvalMentionedAttributeSequences(), existingDI.getEvalRealizations());
                    }
                    datasetInstances.get(previousPredicate).add(DI);
                    //}
                    /*if (w.trim().equals("@x@name_0 is a restaurant in @x@area_0 near to @x@near_0 and @x@near_1 in the expensive pricerange .")) {
                        for (ArrayList<Action> r : DI.getEvalRealizations()) {
                            String l = "";
                            for (Action act : r) {
                                l += act.getWord()+ " ";
                            }
                            System.out.println(l.trim());
                        }
                    }*/
                }
            }
            /*for (String pred : datasetInstances.keySet()) {
                for (DatasetInstance di : datasetInstances.get(pred)) {                    
                    for (ArrayList<Action> r : di.getEvalRealizations()) {
                        String l = "";
                        for (Action act : r) {
                            l += act.getWord()+ " ";
                        }                        
                        //if (l.trim().equals("@x@name_0 is a restaurant in @x@area_0 near to @x@near_0 and @x@near_1 in the expensive pricerange .")) {
                        if (l.trim().equals("@x@name_0 is a restaurant in the expensive price range in the @x@area_0 area near @x@near_0 and @x@near_1 .")) {
                            System.out.println(">>> " + l.trim());                                     
                                for (ArrayList<Action> g : di.getEvalRealizations()) {
                                String f = "";
                                for (Action act : g) {
                                    f += act.getWord()+ " ";
                                }                        
                                System.out.println(f.trim());
                            }
                        }
                    }
                }
            }*/
            for (String pred : datasetInstances.keySet()) {
                for (DatasetInstance di : datasetInstances.get(pred)) {
                    HashSet<String> refs = new HashSet<>();
                    for (ArrayList<Action> refSeq : di.getEvalRealizations()) {
                        refs.add(postProcessRef(di, refSeq));
                    }
                    di.setEvalReferences(refs);
                    di.setTrainReference(postProcessRef(di, di.getTrainRealization()));
                    /*System.out.println("CREATE LISTS");
                    System.out.println(di);
                    System.out.println(di.getMeaningRepresentation().getPredicate());
                    System.out.println(di.getMeaningRepresentation().getMRstr());
                    System.out.println(di.getEvalReferences());
                    System.out.println("=============");*/
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        writeLists();
    }

    /**
     *
     * @param trainingData
     * @param availableWordActions
     * @return
     */
    public Object[] createTrainingDatasets(ArrayList<DatasetInstance> trainingData, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions) {
        predicateAttrTrainingData = new HashMap<>();
        predicateWordTrainingData = new HashMap<>();

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

                        ArrayList<String> attributeSequence = new ArrayList<>();
                        String attrValue = "";
                        for (int w = 0; w < realization.size(); w++) {
                            if (!realization.get(w).getAttribute().equals(Action.TOKEN_PUNCT)
                                    && !realization.get(w).getAttribute().equals(attrValue)) {
                                if (!attrValue.isEmpty()) {
                                    attrValuesToBeMentioned.remove(attrValue);
                                }
                                Instance attrTrainingVector = createAttrInstance(predicate, realization.get(w).getAttribute(), attributeSequence, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableAttributeActions);
                                if (attrTrainingVector != null) {
                                    predicateAttrTrainingData.get(predicate).add(attrTrainingVector);
                                }
                                attributeSequence.add(realization.get(w).getAttribute());

                                attrValue = realization.get(w).getAttribute();
                                if (!attrValue.isEmpty()) {
                                    attrValuesAlreadyMentioned.add(attrValue);
                                    attrValuesToBeMentioned.remove(attrValue);
                                }
                            }
                        }
                        attrValuesAlreadyMentioned = new HashSet<>();
                        attrValuesToBeMentioned = new HashSet<>();
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
                        attrValue = "";
                        int a = -1;
                        ArrayList<String> subPhrase = new ArrayList<>();
                        for (int w = 0; w < realization.size(); w++) {
                            if (!realization.get(w).getAttribute().equals(Action.TOKEN_PUNCT)) {
                                if (!realization.get(w).getAttribute().equals(attrValue)) {
                                    a++;
                                    if (!attrValue.isEmpty()) {
                                        attrValuesToBeMentioned.remove(attrValue);
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

                                if (!attrValue.equals(Action.TOKEN_END)) {
                                    ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                                    for (int i = 0; i < attrs.size() - 1; i++) {
                                        predictedAttributesForInstance.add(attrs.get(i));
                                    }
                                    if (!attrs.get(attrs.size() - 1).equals(attrValue)) {
                                        predictedAttributesForInstance.add(attrs.get(attrs.size() - 1));
                                    }
                                    ArrayList<String> nextAttributesForInstance = new ArrayList<>(attrs.subList(a + 1, attrs.size()));
                                    Instance wordTrainingVector = createWordInstance(predicate, realization.get(w), predictedAttributesForInstance, new ArrayList<Action>(realization.subList(0, w)), nextAttributesForInstance, attrValuesAlreadyMentioned, attrValuesToBeMentioned, isValueMentioned, availableWordActions.get(predicate));

                                    if (wordTrainingVector != null) {
                                        String attribute = attrValue.substring(0, attrValue.indexOf('='));
                                        if (!predicateWordTrainingData.get(predicate).containsKey(attribute)) {
                                            predicateWordTrainingData.get(predicate).put(attribute, new ArrayList<Instance>());
                                        }
                                        predicateWordTrainingData.get(predicate).get(attribute).add(wordTrainingVector);
                                        if (!realization.get(w).getWord().equals(Action.TOKEN_START)
                                                && !realization.get(w).getWord().equals(Action.TOKEN_END)) {
                                            subPhrase.add(realization.get(w).getWord());
                                        }
                                    }
                                    if (!isValueMentioned) {
                                        if (realization.get(w).getWord().startsWith(Action.TOKEN_X)
                                                && (valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.startsWith(Action.TOKEN_X))) {
                                            isValueMentioned = true;
                                        } else if (!realization.get(w).getWord().startsWith(Action.TOKEN_X)
                                                && !(valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.startsWith(Action.TOKEN_X))) {
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
                                    if (!realization.get(w).getWord().startsWith(Action.TOKEN_X)) {
                                        for (String attrValueTBM : attrValuesToBeMentioned) {
                                            if (attrValueTBM.contains("=")) {
                                                String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                                if (!(value.matches("\"[xX][0-9]+\"")
                                                        || value.matches("[xX][0-9]+")
                                                        || value.startsWith(Action.TOKEN_X))) {
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

    /**
     *
     * @param trainingData
     */
    public void createNaiveAlignments(ArrayList<DatasetInstance> trainingData) {
        HashMap<ArrayList<Action>, HashMap<Action, Integer>> punctPatterns = new HashMap<>();
        HashMap<DatasetInstance, ArrayList<Action>> punctRealizations = new HashMap<DatasetInstance, ArrayList<Action>>();

        for (DatasetInstance di : trainingData) {
            HashMap<ArrayList<Action>, ArrayList<Action>> calculatedRealizationsCache = new HashMap<>();
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
                HashMap<String, HashSet<String>> values = new HashMap();
                for (String attr : di.getMeaningRepresentation().getAttributes().keySet()) {
                    values.put(attr, new HashSet(di.getMeaningRepresentation().getAttributes().get(attr)));
                }

                ArrayList<Action> randomRealization = new ArrayList<>();
                for (int i = 0; i < realization.size(); i++) {
                    Action a = realization.get(i);
                    if (a.getAttribute().equals(Action.TOKEN_PUNCT)) {
                        randomRealization.add(new Action(a.getWord(), a.getAttribute()));
                    } else {
                        randomRealization.add(new Action(a.getWord(), ""));
                    }
                }

                HashMap<Double, HashMap<String, ArrayList<Integer>>> indexAlignments = new HashMap<>();
                values.keySet().stream().forEach((String attr) -> {
                    values.get(attr).stream().filter((value) -> (!value.equals("placetoeat")
                            && !(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+") || value.startsWith(Action.TOKEN_X)))).forEach((value) -> {
                        valueAlignments.get(value).keySet().stream().forEach((align) -> {
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
                                        indexAlignments.put(valueAlignments.get(value).get(align), new HashMap<>());
                                    }
                                    indexAlignments.get(valueAlignments.get(value).get(align)).put(attr + "=" + value, indexAlignment);
                                }
                            }
                        });
                    });
                });

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

                for (Action a : randomRealization) {
                    if (a.getWord().startsWith(Action.TOKEN_X)) {
                        String attr = a.getWord().substring(3, a.getWord().lastIndexOf('_')).toLowerCase().trim();
                        /*int index = 0;
                        if (!attrXIndeces.containsKey(attr)) {
                            attrXIndeces.put(attr, 1);
                        } else {
                            index = attrXIndeces.get(attr);
                            attrXIndeces.put(attr, index + 1);
                        }
                        a.setAttribute(attr + "=x" + index);*/
                        a.setAttribute(attr + "=" + a.getWord());
                    }
                }

                String previousAttr = "";
                /*int start = -1;
                for (int i = 0; i < randomRealization.size(); i++) {
                    if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)
                            && !randomRealization.get(i).getAttribute().isEmpty()
                            && !randomRealization.get(i).getAttribute().equals("[]")) {
                        if (start != -1) {
                            int middle = (start + i - 1) / 2 + 1;
                            for (int j = start; j < middle; j++) {
                                if (randomRealization.get(j).getAttribute().isEmpty()
                                        || randomRealization.get(j).getAttribute().equals("[]")) {
                                    randomRealization.get(j).setAttribute(previousAttr);
                                }
                            }
                            for (int j = middle; j < i; j++) {
                                if (randomRealization.get(j).getAttribute().isEmpty()
                                        || randomRealization.get(j).getAttribute().equals("[]")) {
                                    randomRealization.get(j).setAttribute(randomRealization.get(i).getAttribute());
                                }
                            }
                        }
                        start = i;
                        previousAttr = randomRealization.get(i).getAttribute();
                    } else {
                        previousAttr = "";
                    }
                }*/

                previousAttr = "";
                for (int i = 0; i < randomRealization.size(); i++) {
                    if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                        }
                    } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                    }
                }

                previousAttr = "";
                for (int i = randomRealization.size() - 1; i >= 0; i--) {
                    if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                        }
                    } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                    }
                }

                //FIX WRONG @PUNCT@                
                previousAttr = "";
                for (int i = randomRealization.size() - 1; i >= 0; i--) {
                    if (randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT) && !randomRealization.get(i).getWord().matches("[,.?!;:']")) {
                        if (!previousAttr.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                        }
                    } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                    }
                }
                ArrayList<Action> cleanRandomRealization = new ArrayList<>();
                for (Action a : randomRealization) {
                    if (!a.getAttribute().equals(Action.TOKEN_PUNCT)) {
                        cleanRandomRealization.add(a);
                    }
                }
                //ADD END TOKENS
                ArrayList<Action> endRandomRealization = new ArrayList<>();
                previousAttr = "";
                for (int i = 0; i < cleanRandomRealization.size(); i++) {
                    Action a = cleanRandomRealization.get(i);
                    if (!previousAttr.isEmpty()
                            && !a.getAttribute().equals(previousAttr)) {
                        endRandomRealization.add(new Action(Action.TOKEN_END, previousAttr));
                    }
                    endRandomRealization.add(a);
                    previousAttr = a.getAttribute();
                }
                endRandomRealization.add(new Action(Action.TOKEN_END, previousAttr));
                endRandomRealization.add(new Action(Action.TOKEN_END, Action.TOKEN_END));
                calculatedRealizationsCache.put(realization, endRandomRealization);

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

                ArrayList<Action> punctRealization = new ArrayList<>();
                punctRealization.addAll(randomRealization);
                previousAttr = "";
                for (int i = 0; i < punctRealization.size(); i++) {
                    if (!punctRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                        if (!punctRealization.get(i).getAttribute().equals(previousAttr)
                                && !previousAttr.isEmpty()) {
                            punctRealization.add(i, new Action(Action.TOKEN_END, previousAttr));
                            i++;
                        }
                        previousAttr = punctRealization.get(i).getAttribute();
                    }
                }
                if (!punctRealization.get(punctRealization.size() - 1).getWord().equals(Action.TOKEN_END)) {
                    punctRealization.add(new Action(Action.TOKEN_END, previousAttr));
                }
                punctRealizations.put(di, punctRealization);
                for (int i = 0; i < punctRealization.size(); i++) {
                    Action a = punctRealization.get(i);
                    if (a.getAttribute().equals(Action.TOKEN_PUNCT)) {
                        boolean legal = true;
                        ArrayList<Action> surroundingActions = new ArrayList<>();
                        /*if (i - 3 >= 0) {
                            surroundingActions.add(punctRealization.get(i - 3));
                        } else {
                            surroundingActions.add(null);
                        }*/
                        if (i - 2 >= 0) {
                            surroundingActions.add(punctRealization.get(i - 2));
                        } else {
                            surroundingActions.add(null);
                        }
                        if (i - 1 >= 0) {
                            surroundingActions.add(punctRealization.get(i - 1));
                        } else {
                            legal = false;
                        }
                        boolean oneMore = false;
                        if (i + 1 < punctRealization.size()) {
                            surroundingActions.add(punctRealization.get(i + 1));
                            if (!punctRealization.get(i + 1).getAttribute().equals(Action.TOKEN_END)) {
                                oneMore = true;
                            }
                        } else {
                            legal = false;
                        }
                        if (oneMore && i + 2 < punctRealization.size()) {
                            surroundingActions.add(punctRealization.get(i + 2));
                        } else {
                            surroundingActions.add(null);
                        }
                        if (legal) {
                            if (!punctPatterns.get(surroundingActions).containsKey(a)) {
                                punctPatterns.get(surroundingActions).put(a, 1);
                            } else {
                                punctPatterns.get(surroundingActions).put(a, punctPatterns.get(surroundingActions).get(a) + 1);
                            }
                        }
                    }
                }
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

            /*HashSet<String> attrValuesToBeMentioned = new HashSet<>();
            for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
                int a = 0;
                for (String value : di.getMeaningRepresentation().getAttributes().get(attribute)) {
                    if (value.startsWith("\"x")) {
                        value = "x" + a;
                        a++;
                    } else if (value.startsWith("\"")) {
                        value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                    }
                    attrValuesToBeMentioned.add(attribute + "=" + value);
                }
            }
            for (Action key : di.getTrainRealization()) {
                attrValuesToBeMentioned.remove(key.getAttribute());
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
            if (!attrValuesToBeMentioned.isEmpty()) {
                System.out.println("EE " + di.getMeaningRepresentation().getMRstr());
                System.out.println("EE " + di.getMeaningRepresentation().getAttributes());
                System.out.println("EE " + di.getTrainRealization());
                System.out.println(attrValuesToBeMentioned);
            }*/
        }
        for (DatasetInstance di : punctRealizations.keySet()) {
            ArrayList<Action> punctRealization = punctRealizations.get(di);
            for (ArrayList<Action> surrounds : punctPatterns.keySet()) {
                int beforeNulls = 0;
                if (surrounds.get(0) == null) {
                    beforeNulls++;
                }
                if (surrounds.get(1) == null) {
                    beforeNulls++;
                }
                for (int i = 0 - beforeNulls; i < punctRealization.size(); i++) {
                    boolean matches = true;
                    int m = 0;
                    for (int s = 0; s < surrounds.size(); s++) {
                        if (surrounds.get(s) != null) {
                            if (i + s < punctRealization.size()) {
                                if (!punctRealization.get(i + s).getWord().equals(surrounds.get(s).getWord()) /*|| !cleanActionList.get(i).getAttribute().equals(surrounds.get(s).getAttribute())*/) {
                                    matches = false;
                                    s = surrounds.size();
                                } else {
                                    m++;
                                }
                            } else {
                                matches = false;
                                s = surrounds.size();
                            }
                        } else if (s < 2 && i + s >= 0) {
                            matches = false;
                            s = surrounds.size();
                        } else if (s >= 2 && i + s < punctRealization.size()) {
                            matches = false;
                            s = surrounds.size();
                        }
                    }
                    if (matches && m > 0) {
                        Action a = new Action("", "");
                        if (!punctPatterns.get(surrounds).containsKey(a)) {
                            punctPatterns.get(surrounds).put(a, 1);
                        } else {
                            punctPatterns.get(surrounds).put(a, punctPatterns.get(surrounds).get(a) + 1);
                        }
                    }
                }
            }
        }
        for (ArrayList<Action> punct : punctPatterns.keySet()) {
            Action bestAction = null;
            int bestCount = 0;
            for (Action a : punctPatterns.get(punct).keySet()) {
                if (punctPatterns.get(punct).get(a) > bestCount) {
                    bestAction = a;
                    bestCount = punctPatterns.get(punct).get(a);
                } else if (punctPatterns.get(punct).get(a) == bestCount
                        && bestAction.getWord().isEmpty()) {
                    bestAction = a;
                }
            }
            if (!bestAction.getWord().isEmpty()) {
                punctuationPatterns.put(punct, bestAction);
            }
        }
    }

    /**
     *
     * @param predicate
     * @param bestAction
     * @param previousGeneratedAttrs
     * @param attrValuesAlreadyMentioned
     * @param attrValuesToBeMentioned
     * @param MR
     * @param availableAttributeActions
     * @return
     */
    public Instance createAttrInstance(String predicate, String bestAction, ArrayList<String> previousGeneratedAttrs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, MeaningRepresentation MR, HashMap<String, HashSet<String>> availableAttributeActions) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

        if (!bestAction.isEmpty()) {
            //COSTS
            if (bestAction.equals(Action.TOKEN_END)) {
                costs.put(Action.TOKEN_END, 0.0);
                for (String action : availableAttributeActions.get(predicate)) {
                    costs.put(action, 1.0);
                }
            } else if (!bestAction.equals("@TOK@")) {
                costs.put(Action.TOKEN_END, 1.0);
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
        return createAttrInstanceWithCosts(predicate, costs, previousGeneratedAttrs, attrValuesAlreadyMentioned, attrValuesToBeMentioned, availableAttributeActions, MR);
    }

    /**
     *
     * @param predicate
     * @param costs
     * @param previousGeneratedAttrs
     * @param attrValuesAlreadyMentioned
     * @param attrValuesToBeMentioned
     * @param availableAttributeActions
     * @param MR
     * @return
     */
    public Instance createAttrInstanceWithCosts(String predicate, TObjectDoubleHashMap<String> costs, ArrayList<String> previousGeneratedAttrs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, HashMap<String, HashSet<String>> availableAttributeActions, MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        for (String action : availableAttributeActions.get(predicate)) {
            valueSpecificFeatures.put(action, new TObjectDoubleHashMap<String>());
        }

        ArrayList<String> mentionedAttrValues = new ArrayList<>();
        for (String attrValue : previousGeneratedAttrs) {
            if (!attrValue.equals(Action.TOKEN_END)
                    && !attrValue.equals(Action.TOKEN_START)) {
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
        //Attr N-Grams            
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
            if (!attrValue.endsWith("placetoeat")) {
                generalFeatures.put("feature_attrValue_toBeMentioned_" + attrValue, 1.0);
            }
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
            if (!attrValue.endsWith("placetoeat")) {
                String attr = attrValue;
                if (attr.contains("=")
                        && !attr.equals("type=placetoeat")) {
                    attr = attrValue.substring(0, attrValue.indexOf('='));
                }
                attrsToBeMentioned.add(attr);
            }
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
 /*ArrayList<Action> generatedWords = new ArrayList<>();
        ArrayList<Action> generatedWordsInPreviousAttrValue = new ArrayList<>();
        if (!mentionedAttrValues.isEmpty()) {
            String previousAttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 1);
            for (int i = 0; i < previousGeneratedWords.size(); i++) {
                Action a = previousGeneratedWords.get(i);
                if (!a.getWord().equals(Action.TOKEN_START)
                        && !a.getWord().equals(Action.TOKEN_END)) {
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
        generalFeatures.put("feature_word_5gram_" + prev5gram.toLowerCase(), 1.0);*/

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
        /*for (int j = 1; j <= 1; j++) {
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
        generalFeatures.put("feature_previousAttrValueWord_5gram_" + prevCurrentAttrValue5gram.toLowerCase(), 1.0);*/

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
        /*for (int j = 1; j <= 1; j++) {
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
        generalFeatures.put("feature_attrWord_5gram_" + prevAttrWord5gram.toLowerCase(), 1.0);*/

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
        //Previous POS features
        /*for (int j = 1; j <= 1; j++) {
        String previousPOS = "@@";
        if (generatedWords.size() - j >= 0) {
        previousPOS = generatedWords.get(generatedWords.size() - j).getPOS().trim();
        }
        generalFeatures.put("feature_POS_" + j + "_" + previousPOS.toLowerCase(), 1.0);
        }
        String prevPOS = "@@";
        if (generatedWords.size() - 1 >= 0) {
        prevPOS = generatedWords.get(generatedWords.size() - 1).getPOS().trim();
        }
        String prev2POS = "@@";
        if (generatedWords.size() - 2 >= 0) {
        prev2POS = generatedWords.get(generatedWords.size() - 2).getPOS().trim();
        }
        String prev3POS = "@@";
        if (generatedWords.size() - 3 >= 0) {
        prev3POS = generatedWords.get(generatedWords.size() - 3).getPOS().trim();
        }
        String prev4POS = "@@";
        if (generatedWords.size() - 4 >= 0) {
        prev4POS = generatedWords.get(generatedWords.size() - 4).getPOS().trim();
        }
        String prev5POS = "@@";
        if (generatedWords.size() - 5 >= 0) {
        prev5POS = generatedWords.get(generatedWords.size() - 5).getPOS().trim();
        }
        
        String prevPOSBigram = prev2POS + "|" + prevPOS;
        String prevPOSTrigram = prev3POS + "|" + prev2POS + "|" + prevPOS;
        String prevPOS4gram = prev4POS + "|" + prev3POS + "|" + prev2POS + "|" + prevPOS;
        String prevPOS5gram = prev5POS + "|" + prev4POS + "|" + prev3POS + "|" + prev2POS + "|" + prevPOS;
        
        generalFeatures.put("feature_POS_bigram_" + prevPOSBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_POS_trigram_" + prevPOSTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_POS_4gram_" + prevPOS4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_POS_5gram_" + prevPOS5gram.toLowerCase(), 1.0);*/
        //Previous AttrValue|Word features
        /*for (int j = 1; j <= 1; j++) {
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
        generalFeatures.put("feature_attrValueWord_5gram_" + prevAttrValueWord5gram.toLowerCase(), 1.0);*/

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
            if (action.equals(Action.TOKEN_END)) {
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
                    if (!attrValue.endsWith("placetoeat")
                            && attrValue.substring(0, attrValue.indexOf('=')).equals(action)) {
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
            if (useLMs) {
                String nextValue = chooseNextValue(action, attrValuesToBeMentioned);
                if (nextValue.isEmpty() && !action.equals(Action.TOKEN_END)) {
                    valueSpecificFeatures.get(action).put("global_feature_LMAttr_score", 0.0);
                } else {
                    ArrayList<String> fullGramLM = new ArrayList<>();
                    for (int i = 0; i < mentionedAttrValues.size(); i++) {
                        fullGramLM.add(mentionedAttrValues.get(i));
                    }
                    ArrayList<String> prev5attrValueGramLM = new ArrayList<String>();
                    int j = 0;
                    for (int i = mentionedAttrValues.size() - 1; (i >= 0 && j < 5); i--) {
                        prev5attrValueGramLM.add(0, mentionedAttrValues.get(i));
                        j++;
                    }
                    if (!action.equals(Action.TOKEN_END)) {
                        prev5attrValueGramLM.add(action + "=" + chooseNextValue(action, attrValuesToBeMentioned));
                    } else {
                        prev5attrValueGramLM.add(action);
                    }
                    if (prev5attrValueGramLM.size() < 3) {
                        prev5attrValueGramLM.add(0, "@@");
                    }
                    if (prev5attrValueGramLM.size() < 4) {
                        prev5attrValueGramLM.add(0, "@@");
                    }

                    double afterLMScore = attrLMsPerPredicate.get(predicate).getProbability(prev5attrValueGramLM);
                    valueSpecificFeatures.get(action).put("global_feature_LMAttr_score", afterLMScore);

                    afterLMScore = attrLMsPerPredicate.get(predicate).getProbability(fullGramLM);
                    valueSpecificFeatures.get(action).put("global_feature_LMAttrFull_score", afterLMScore);
                }
            }
        }
        return new Instance(generalFeatures, valueSpecificFeatures, costs);
    }

    /**
     *
     * @param predicate
     * @param bestAction
     * @param previousGeneratedAttributes
     * @param previousGeneratedWords
     * @param nextGeneratedAttributes
     * @param attrValuesAlreadyMentioned
     * @param attrValuesThatFollow
     * @param wasValueMentioned
     * @param availableWordActions
     * @return
     */
    public Instance createWordInstance(String predicate, Action bestAction, ArrayList<String> previousGeneratedAttributes, ArrayList<Action> previousGeneratedWords, ArrayList<String> nextGeneratedAttributes, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesThatFollow, boolean wasValueMentioned, HashMap<String, HashSet<Action>> availableWordActions) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        if (!bestAction.getWord().trim().isEmpty()) {
            //COSTS
            String attr = bestAction.getAttribute().substring(0, bestAction.getAttribute().indexOf('='));
            if (bestAction.getAttribute().contains("=")) {
                attr = bestAction.getAttribute().substring(0, bestAction.getAttribute().indexOf('='));
            }
            for (Action action : availableWordActions.get(attr)) {
                if (action.getWord().equalsIgnoreCase(bestAction.getWord().trim())) {
                    costs.put(action.getAction(), 0.0);
                } else {
                    costs.put(action.getAction(), 1.0);
                }
            }

            if (bestAction.getWord().trim().equalsIgnoreCase(Action.TOKEN_END)) {
                costs.put(Action.TOKEN_END, 0.0);
            } else {
                costs.put(Action.TOKEN_END, 1.0);
            }
        }
        return createWordInstanceWithCosts(predicate, bestAction.getAttribute(), costs, previousGeneratedAttributes, previousGeneratedWords, nextGeneratedAttributes, attrValuesAlreadyMentioned, attrValuesThatFollow, wasValueMentioned, availableWordActions);
    }

    /**
     *
     * @param predicate
     * @param currentAttrValue
     * @param costs
     * @param generatedAttributes
     * @param previousGeneratedWords
     * @param nextGeneratedAttributes
     * @param attrValuesAlreadyMentioned
     * @param attrValuesThatFollow
     * @param wasValueMentioned
     * @param availableWordActions
     * @return
     */
    public Instance createWordInstanceWithCosts(String predicate, String currentAttrValue, TObjectDoubleHashMap<String> costs, ArrayList<String> generatedAttributes, ArrayList<Action> previousGeneratedWords, ArrayList<String> nextGeneratedAttributes, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesThatFollow, boolean wasValueMentioned, HashMap<String, HashSet<Action>> availableWordActions) {
        String currentAttr = currentAttrValue;
        String currentValue = "";
        if (currentAttr.contains("=")) {
            currentAttr = currentAttrValue.substring(0, currentAttrValue.indexOf('='));
            currentValue = currentAttrValue.substring(currentAttrValue.indexOf('=') + 1);
        }

        TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        for (Action action : availableWordActions.get(currentAttr)) {
            valueSpecificFeatures.put(action.getAction(), new TObjectDoubleHashMap<String>());
        }

        /*if (gWords.get(wIndex).getWord().equals(Action.TOKEN_END)) {
        System.out.println("!!! "+ gWords.subList(0, wIndex + 1));
        }*/
        ArrayList<Action> generatedWords = new ArrayList<>();
        ArrayList<Action> generatedWordsInSameAttrValue = new ArrayList<>();
        ArrayList<String> generatedPhrase = new ArrayList<>();
        for (int i = 0; i < previousGeneratedWords.size(); i++) {
            Action a = previousGeneratedWords.get(i);
            if (!a.getWord().equals(Action.TOKEN_START)
                    && !a.getWord().equals(Action.TOKEN_END)) {
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
        /*if (generatedWordsInSameAttrValue.isEmpty()) {
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
        generalFeatures.put("feature_currentAttrValueWord_5gram_" + prevCurrentAttrValue5gram.toLowerCase(), 1.0);*/

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
        //Previous POS features
        /*for (int j = 1; j <= 1; j++) {
        String previousPOS = "@@";
        if (generatedWords.size() - j >= 0) {
        previousPOS = generatedWords.get(generatedWords.size() - j).getPOS().trim();
        }
        generalFeatures.put("feature_POS_" + j + "_" + previousPOS.toLowerCase(), 1.0);
        }
        String prevPOS = "@@";
        if (generatedWords.size() - 1 >= 0) {
        prevPOS = generatedWords.get(generatedWords.size() - 1).getPOS().trim();
        }
        String prev2POS = "@@";
        if (generatedWords.size() - 2 >= 0) {
        prev2POS = generatedWords.get(generatedWords.size() - 2).getPOS().trim();
        }
        String prev3POS = "@@";
        if (generatedWords.size() - 3 >= 0) {
        prev3POS = generatedWords.get(generatedWords.size() - 3).getPOS().trim();
        }
        String prev4POS = "@@";
        if (generatedWords.size() - 4 >= 0) {
        prev4POS = generatedWords.get(generatedWords.size() - 4).getPOS().trim();
        }
        String prev5POS = "@@";
        if (generatedWords.size() - 5 >= 0) {
        prev5POS = generatedWords.get(generatedWords.size() - 5).getPOS().trim();
        }
        
        String prevPOSBigram = prev2POS + "|" + prevPOS;
        String prevPOSTrigram = prev3POS + "|" + prev2POS + "|" + prevPOS;
        String prevPOS4gram = prev4POS + "|" + prev3POS + "|" + prev2POS + "|" + prevPOS;
        String prevPOS5gram = prev5POS + "|" + prev4POS + "|" + prev3POS + "|" + prev2POS + "|" + prevPOS;
        
        generalFeatures.put("feature_POS_bigram_" + prevPOSBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_POS_trigram_" + prevPOSTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_POS_4gram_" + prevPOS4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_POS_5gram_" + prevPOS5gram.toLowerCase(), 1.0);*/
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
        //Next attr features
        for (int j = 0; j < 1; j++) {
            String nextAttr = "@@";
            if (j < nextGeneratedAttributes.size()) {
                if (nextGeneratedAttributes.get(j).contains("=")) {
                    nextAttr = nextGeneratedAttributes.get(j).trim().substring(0, nextGeneratedAttributes.get(j).indexOf('='));
                } else {
                    nextAttr = nextGeneratedAttributes.get(j).trim();
                }
            }
            generalFeatures.put("feature_nextAttr_" + j + "_" + nextAttr, 1.0);
        }
        String nextAttr = "@@";
        if (0 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(0).contains("=")) {
                nextAttr = nextGeneratedAttributes.get(0).trim().substring(0, nextGeneratedAttributes.get(0).indexOf('='));
            } else {
                nextAttr = nextGeneratedAttributes.get(0).trim();
            }
        }
        String next2Attr = "@@";
        if (1 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(1).contains("=")) {
                next2Attr = nextGeneratedAttributes.get(1).trim().substring(0, nextGeneratedAttributes.get(1).indexOf('='));
            } else {
                next2Attr = nextGeneratedAttributes.get(1).trim();
            }
        }
        String next3Attr = "@@";
        if (2 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(2).contains("=")) {
                next3Attr = nextGeneratedAttributes.get(2).trim().substring(0, nextGeneratedAttributes.get(2).indexOf('='));
            } else {
                next3Attr = nextGeneratedAttributes.get(2).trim();
            }
        }
        String next4Attr = "@@";
        if (3 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(3).contains("=")) {
                next4Attr = nextGeneratedAttributes.get(3).trim().substring(0, nextGeneratedAttributes.get(3).indexOf('='));
            } else {
                next4Attr = nextGeneratedAttributes.get(3).trim();
            }
        }
        String next5Attr = "@@";
        if (4 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(4).contains("=")) {
                next5Attr = nextGeneratedAttributes.get(4).trim().substring(0, nextGeneratedAttributes.get(4).indexOf('='));
            } else {
                next5Attr = nextGeneratedAttributes.get(4).trim();
            }
        }

        String nextAttrBigram = nextAttr + "|" + next2Attr;
        String nextAttrTrigram = nextAttr + "|" + next2Attr + "|" + next3Attr;
        String nextAttr4gram = nextAttr + "|" + next2Attr + "|" + next3Attr + "|" + next4Attr;
        String nextAttr5gram = nextAttr + "|" + next2Attr + "|" + next3Attr + "|" + next4Attr + "|" + next5Attr;

        generalFeatures.put("feature_nextAttr_bigram_" + nextAttrBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_nextAttr_trigram_" + nextAttrTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_nextAttr_4gram_" + nextAttr4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_nextAttr_5gram_" + nextAttr5gram.toLowerCase(), 1.0);

        //Next attrValue features
        for (int j = 0; j < 1; j++) {
            String nextAttrValue = "@@";
            if (j < nextGeneratedAttributes.size()) {
                nextAttrValue = nextGeneratedAttributes.get(j).trim();
            }
            generalFeatures.put("feature_nextAttrValue_" + j + "_" + nextAttrValue, 1.0);
        }
        String nextAttrValue = "@@";
        if (0 < nextGeneratedAttributes.size()) {
            nextAttrValue = nextGeneratedAttributes.get(0).trim();
        }
        String next2AttrValue = "@@";
        if (1 < nextGeneratedAttributes.size()) {
            next2AttrValue = nextGeneratedAttributes.get(1).trim();
        }
        String next3AttrValue = "@@";
        if (2 < nextGeneratedAttributes.size()) {
            next3AttrValue = nextGeneratedAttributes.get(2).trim();
        }
        String next4AttrValue = "@@";
        if (3 < nextGeneratedAttributes.size()) {
            next4AttrValue = nextGeneratedAttributes.get(3).trim();
        }
        String next5AttrValue = "@@";
        if (4 < nextGeneratedAttributes.size()) {
            next5AttrValue = nextGeneratedAttributes.get(4).trim();
        }

        String nextAttrValueBigram = nextAttrValue + "|" + next2AttrValue;
        String nextAttrValueTrigram = nextAttrValue + "|" + next2AttrValue + "|" + next3AttrValue;
        String nextAttrValue4gram = nextAttrValue + "|" + next2AttrValue + "|" + next3AttrValue + "|" + next4AttrValue;
        String nextAttrValue5gram = nextAttrValue + "|" + next2AttrValue + "|" + next3AttrValue + "|" + next4AttrValue + "|" + next5AttrValue;

        generalFeatures.put("feature_nextAttrValue_bigram_" + nextAttrValueBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_nextAttrValue_trigram_" + nextAttrValueTrigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_nextAttrValue_4gram_" + nextAttrValue4gram.toLowerCase(), 1.0);
        generalFeatures.put("feature_nextAttrValue_5gram_" + nextAttrValue5gram.toLowerCase(), 1.0);

        //If values have already been generated or not
        generalFeatures.put("feature_valueToBeMentioned_" + currentValue.toLowerCase(), 1.0);
        if (wasValueMentioned) {
            generalFeatures.put("feature_wasValueMentioned_true", 1.0);
        } else {
            //generalFeatures.put("feature_wasValueMentioned_false", 1.0);
        }
        HashSet<String> valuesThatFollow = new HashSet<>();
        for (String attrValue : attrValuesThatFollow) {
            if (!attrValue.endsWith("placetoeat")) {
                generalFeatures.put("feature_attrValuesThatFollow_" + attrValue.toLowerCase(), 1.0);
                if (attrValue.contains("=")) {
                    String v = attrValue.substring(attrValue.indexOf('=') + 1);
                    if (v.matches("[xX][0-9]+")) {
                        String attr = attrValue.substring(0, attrValue.indexOf('='));
                        valuesThatFollow.add(Action.TOKEN_X + attr + "_" + v.substring(1));
                    } else {
                        valuesThatFollow.add(v);
                    }
                    generalFeatures.put("feature_attrsThatFollow_" + attrValue.substring(0, attrValue.indexOf('=')).toLowerCase(), 1.0);
                } else {
                    generalFeatures.put("feature_attrsThatFollow_" + attrValue.toLowerCase(), 1.0);
                }
            }
        }
        if (valuesThatFollow.isEmpty()) {
            generalFeatures.put("feature_noAttrsFollow", 1.0);
        } else {
            generalFeatures.put("feature_noAttrsFollow", 0.0);
        }
        HashSet<String> mentionedValues = new HashSet<>();
        for (String attrValue : attrValuesAlreadyMentioned) {
            generalFeatures.put("feature_attrValuesAlreadyMentioned_" + attrValue.toLowerCase(), 1.0);
            if (attrValue.contains("=")) {
                generalFeatures.put("feature_attrsAlreadyMentioned_" + attrValue.substring(0, attrValue.indexOf('=')).toLowerCase(), 1.0);
                String v = attrValue.substring(attrValue.indexOf('=') + 1);
                if (v.matches("[xX][0-9]+")) {
                    String attr = attrValue.substring(0, attrValue.indexOf('='));
                    mentionedValues.add(Action.TOKEN_X + attr + "_" + v.substring(1));
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
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_sameAsPreviousWord", 1.0);
            } else {
                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_notSameAsPreviousWord", 1.0);
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_notSameAsPreviousWord", 1.0);
            }
            //Has word appeared in the same attrValue before
            for (Action previousAction : generatedWords) {
                if (previousAction.getWord().equals(action.getWord())
                        && previousAction.getAttribute().equals(currentAttrValue)) {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_appearedInSameAttrValue", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_appearedInSameAttrValue", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_notAppearedInSameAttrValue", 1.0);
                }
            }
            //Has word appeared before
            for (Action previousAction : generatedWords) {
                if (previousAction.getWord().equals(action.getWord())) {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_appeared", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_appeared", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getWord()).put("global_feature_specific_notAppeared", 1.0);
                }
            }
            HashSet<String> keys = new HashSet<>(valueSpecificFeatures.get(action.getAction()).keySet());
            for (String feature1 : keys) {
                for (String feature2 : keys) {
                    if (valueSpecificFeatures.get(action.getAction()).get(feature1) == 1.0
                            && valueSpecificFeatures.get(action.getAction()).get(feature2) == 1.0
                            && feature1.compareTo(feature2) < 0) {
                        valueSpecificFeatures.get(action.getAction()).put(feature1 + "&&" + feature2, 1.0);
                    }
                }
            }
            if (!action.getWord().startsWith(Action.TOKEN_X)) {
                for (String value : valueAlignments.keySet()) {
                    for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                        if (alignedStr.get(0).equals(action.getWord())) {
                            if (mentionedValues.contains(value)) {
                                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_beginsValue_alreadyMentioned", 1.0);
                                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_beginsValue_alreadyMentioned", 1.0);

                            } else if (currentValue.equals(value)) {
                                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_beginsValue_current", 1.0);
                                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_beginsValue_current", 1.0);

                            } else if (valuesThatFollow.contains(value)) {
                                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_beginsValue_thatFollows", 1.0);
                                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_beginsValue_thatFollows", 1.0);

                            } else {
                                //valueSpecificFeatures.get(action.getWord()).put("feature_specific_beginsValue_notInMR", 1.0);
                                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_beginsValue_notInMR", 1.0);

                            }
                        } else {
                            for (int i = 1; i < alignedStr.size(); i++) {
                                if (alignedStr.get(i).equals(action.getWord())) {
                                    if (endsWith(generatedPhrase, new ArrayList<String>(alignedStr.subList(0, i + 1)))) {
                                        if (mentionedValues.contains(value)) {
                                            //valueSpecificFeatures.get(action.getWord()).put("feature_specific_inValue_alreadyMentioned", 1.0);
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_inValue_alreadyMentioned", 1.0);

                                        } else if (currentValue.equals(value)) {
                                            //valueSpecificFeatures.get(action.getWord()).put("feature_specific_inValue_current", 1.0);
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_inValue_current", 1.0);

                                        } else if (valuesThatFollow.contains(value)) {
                                            //valueSpecificFeatures.get(action.getWord()).put("feature_specific_inValue_thatFollows", 1.0);
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_inValue_thatFollows", 1.0);

                                        } else {
                                            //valueSpecificFeatures.get(action.getWord()).put("feature_specific_inValue_notInMR", 1.0);
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_inValue_notInMR", 1.0);

                                        }
                                    } else {
                                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_outOfValue", 1.0);
                                        valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_outOfValue", 1.0);
                                    }
                                }
                            }
                        }
                    }
                }
                if (action.getWord().equals(Action.TOKEN_END)) {
                    if (generatedWordsInSameAttrValue.isEmpty()) {
                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_closingEmptyAttr", 1.0);
                        valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_closingEmptyAttr", 1.0);
                    }
                    if (!wasValueMentioned) {
                        //valueSpecificFeatures.get(action.getWord()).put("feature_specific_closingAttrWithValueNotMentioned", 1.0);
                        valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_closingAttrWithValueNotMentioned", 1.0);
                    }

                    if (!prevWord.equals("@@")) {
                        boolean alignmentIsOpen = false;
                        for (String value : valueAlignments.keySet()) {
                            for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                for (int i = 0; i < alignedStr.size() - 1; i++) {
                                    if (alignedStr.get(i).equals(prevWord)
                                            && endsWith(generatedPhrase, new ArrayList<String>(alignedStr.subList(0, i + 1)))) {
                                        alignmentIsOpen = true;
                                    }
                                }
                            }
                        }
                        if (alignmentIsOpen) {
                            // valueSpecificFeatures.get(action.getWord()).put("feature_specific_closingAttrWhileValueIsNotConcluded", 1.0);
                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_closingAttrWhileValueIsNotConcluded", 1.0);
                        }
                    }
                }
            } else if (currentValue.equals("no")
                    || currentValue.equals("yes")
                    || currentValue.equals("yes or no")
                    || currentValue.equals("none")
                    || currentValue.equals("empty")
                    || currentValue.equals("dont_care")) {
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_notInMR", 1.0);
            } else {
                String currentValueVariant = "";
                if (currentValue.matches("[xX][0-9]+")) {
                    currentValueVariant = Action.TOKEN_X + currentAttr + "_" + currentValue.substring(1);
                }

                if (mentionedValues.contains(action.getWord())) {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_XValue_alreadyMentioned", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_alreadyMentioned", 1.0);
                } else if (currentValueVariant.equals(action.getWord())
                        && !currentValueVariant.isEmpty()) {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_XValue_current", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_current", 1.0);

                } else if (valuesThatFollow.contains(action.getWord())) {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_XValue_thatFollows", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_thatFollows", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getWord()).put("feature_specific_XValue_notInMR", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_notInMR", 1.0);
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

            if (useLMs) {
                ArrayList<String> fullGramLM = new ArrayList<>();
                for (int i = 0; i < generatedWords.size(); i++) {
                    fullGramLM.add(generatedWords.get(i).getWord());
                }

                ArrayList<String> prev5wordGramLM = new ArrayList<>();
                int j = 0;
                for (int i = generatedWords.size() - 1; (i >= 0 && j < 5); i--) {
                    prev5wordGramLM.add(0, generatedWords.get(i).getWord());
                    j++;
                }
                prev5wordGramLM.add(action.getWord());
                if (prev5wordGramLM.size() < 3) {
                    prev5wordGramLM.add(0, "@@");
                }
                if (prev5wordGramLM.size() < 4) {
                    prev5wordGramLM.add(0, "@@");
                }

                double afterLMScorePerPred5Gram = wordLMsPerPredicate.get(predicate).getProbability(prev5wordGramLM);
                valueSpecificFeatures.get(action.getAction()).put("global_feature_LMWord_perPredicate_5gram_score", afterLMScorePerPred5Gram);
                //double afterLMScore5Gram = wordLM.getProbability(prev5wordGramLM);
                //valueSpecificFeatures.get(action.getAction()).put("global_feature_LMWord_5gram_score", afterLMScore5Gram);
                double afterLMScorePerPred = wordLMsPerPredicate.get(predicate).getProbability(fullGramLM);
                valueSpecificFeatures.get(action.getAction()).put("global_feature_LMWord_perPredicate_score", afterLMScorePerPred);
                //double afterLMScore = wordLM.getProbability(fullGramLM);
                //valueSpecificFeatures.get(action.getAction()).put("global_feature_LMWord_score", afterLMScore);
            }
        }

        return new Instance(generalFeatures, valueSpecificFeatures, costs);
    }

    /**
     *
     * @param di
     * @param wordSequence
     * @return
     */
    public String postProcessWordSequence(DatasetInstance di, ArrayList<Action> wordSequence) {
        HashSet<ArrayList<Action>> matched = new HashSet<>();
        ArrayList<Action> processedWordSequence = new ArrayList<>();
        for (Action act : wordSequence) {
            processedWordSequence.add(new Action(act));
        }
        if (!processedWordSequence.isEmpty()
                && processedWordSequence.get(processedWordSequence.size() - 1).getWord().equals(Action.TOKEN_END)
                && processedWordSequence.get(processedWordSequence.size() - 1).getAttribute().equals(Action.TOKEN_END)) {
            processedWordSequence.remove(processedWordSequence.size() - 1);

        }

        for (ArrayList<Action> surrounds : punctuationPatterns.keySet()) {
            int beforeNulls = 0;
            if (surrounds.get(0) == null) {
                beforeNulls++;
            }
            if (surrounds.get(1) == null) {
                beforeNulls++;
            }
            for (int i = 0 - beforeNulls; i < processedWordSequence.size(); i++) {
                boolean matches = true;
                int m = 0;
                for (int s = 0; s < surrounds.size(); s++) {
                    if (surrounds.get(s) != null) {
                        if (i + s < processedWordSequence.size()) {
                            if (!processedWordSequence.get(i + s).getWord().equals(surrounds.get(s).getWord()) /*|| !cleanActionList.get(i).getAttribute().equals(surrounds.get(s).getAttribute())*/) {
                                matches = false;
                                s = surrounds.size();
                            } else {
                                m++;
                            }
                        } else {
                            matches = false;
                            s = surrounds.size();
                        }
                    } else if (s < 2 && i + s >= 0) {
                        matches = false;
                        s = surrounds.size();
                    } else if (s >= 2 && i + s < processedWordSequence.size()) {
                        matches = false;
                        s = surrounds.size();
                    }
                }
                if (matches && m > 0) {
                    matched.add(surrounds);
                    processedWordSequence.add(i + 2, punctuationPatterns.get(surrounds));
                }
            }
        }
        boolean isLastPunct = true;
        if (processedWordSequence.contains(new Action("and", ""))) {
            for (int i = processedWordSequence.size() - 1; i > 0; i--) {
                if (processedWordSequence.get(i).getWord().equals(",")
                        && isLastPunct) {
                    isLastPunct = false;
                    processedWordSequence.get(i).setWord("and");
                } else if (processedWordSequence.get(i).getWord().equals("and")
                        && isLastPunct) {
                    isLastPunct = false;
                } else if (processedWordSequence.get(i).getWord().equals("and")
                        && !isLastPunct) {
                    processedWordSequence.get(i).setWord(",");
                }
            }
        }

        ArrayList<Action> cleanActionList = new ArrayList<>();
        for (Action action : processedWordSequence) {
            if (!action.getWord().equals(Action.TOKEN_START)
                    && !action.getWord().equals(Action.TOKEN_END)) {
                cleanActionList.add(action);
            }
        }

        String predictedWordSequence = " ";
        for (Action action : cleanActionList) {
            if (action.getWord().startsWith(Action.TOKEN_X)) {
                predictedWordSequence += "x ";
            } else {
                predictedWordSequence += action.getWord() + " ";
            }
        }

        predictedWordSequence = predictedWordSequence.trim();
        if (di.getMeaningRepresentation().getPredicate().startsWith("?")
                && !predictedWordSequence.endsWith("?")) {
            if (predictedWordSequence.endsWith(".")) {
                predictedWordSequence = predictedWordSequence.substring(0, predictedWordSequence.length() - 1);
            }
            predictedWordSequence = predictedWordSequence.trim() + "?";
        } else if (!predictedWordSequence.endsWith(".") && !predictedWordSequence.endsWith("?")) {
            /*if (predictedWordSequence.endsWith("?")) {
                predictedWordSequence = predictedWordSequence.substring(0, predictedWordSequence.length() - 1);
            }*/
            predictedWordSequence = predictedWordSequence.trim() + ".";
        }
        predictedWordSequence = predictedWordSequence.replaceAll(" the the ", " the ").replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
        predictedWordSequence = predictedWordSequence.replaceAll(" , \\. ", " \\. ").replaceAll(" and \\. ", " \\. ").replaceAll(" , \\? ", " \\? ").replaceAll(" and \\? ", " \\? ").replaceAll(" ,\\. ", " \\. ").replaceAll(" and\\. ", " \\. ").replaceAll(" ,\\? ", " \\? ").replaceAll(" and\\? ", " \\? ").trim();
        /*for (String comp : sillyCompositeWordsInData.keySet()) {
            predictedWordSequence = predictedWordSequence.replaceAll(comp, sillyCompositeWordsInData.get(comp));
        }*/
        if (predictedWordSequence.startsWith(",")
                || predictedWordSequence.startsWith(".")
                || predictedWordSequence.startsWith("?")) {
            predictedWordSequence = predictedWordSequence.substring(1).trim();
        }
        if (predictedWordSequence.startsWith(",")) {
            System.out.println(wordSequence);
            System.out.println(matched);
            System.out.println(predictedWordSequence);
        }
        return predictedWordSequence;
    }

    /**
     *
     * @param di
     * @param refSeq
     * @return
     */
    public String postProcessRef(DatasetInstance di, ArrayList<Action> refSeq) {
        String cleanedWords = "";
        for (Action action : refSeq) {
            if (!action.equals(new Action(Action.TOKEN_START, ""))
                    && !action.equals(new Action(Action.TOKEN_END, ""))) {
                if (action.getWord().startsWith(Action.TOKEN_X)) {
                    cleanedWords += "x ";
                } else {
                    cleanedWords += action.getWord() + " ";
                }
            }
        }
        cleanedWords = cleanedWords.trim();
        if (di.getMeaningRepresentation().getPredicate().startsWith("?")
                && !cleanedWords.endsWith("?")
                && !cleanedWords.endsWith(".")) {
            cleanedWords = cleanedWords.trim() + "?";
        } else if (!cleanedWords.endsWith("?")
                && !cleanedWords.endsWith(".")) {
            cleanedWords = cleanedWords.trim() + ".";
        }
        cleanedWords = cleanedWords.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
        /*for (String comp : sillyCompositeWordsInData.keySet()) {
            cleanedWords = cleanedWords.replaceAll(comp, sillyCompositeWordsInData.get(comp));
        }*/
        return cleanedWords.trim();
    }

    /*public  ActionSequence generateContentSequence(String predicate, DatasetInstance di, ActionSequence partialContentSequence, JAROW classifierAttrs, HashMap<String, HashSet<String>> availableContentActions) {
        ArrayList<Action> contentSequence = new ArrayList<>();

        String predictedAttr = "";
        ArrayList<String> predictedAttrValues = new ArrayList<>();
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
                attrValuesToBeMentioned.add(attribute + "=" + value);
            }
        }
        if (attrValuesToBeMentioned.isEmpty()) {
            attrValuesToBeMentioned.add("empty=empty");
        }

        ArrayList<String> sortedAlreadyMentioned = new ArrayList<>(attrValuesAlreadyMentioned);
        ArrayList<String> sortedToBeMentioned = new ArrayList<>(attrValuesToBeMentioned);
        Collections.sort(sortedAlreadyMentioned);
        Collections.sort(sortedToBeMentioned);
        String key = "CS|" + predicate + "|" + sortedAlreadyMentioned + "|" + sortedToBeMentioned + "|" + partialContentSequence.getSequence().toString();
        ActionSequence cachedSequence = JDAggerForBagel.wordSequenceCache.get(key);
        if (cachedSequence != null) {
            return cachedSequence;
        }

        while (!predictedAttr.equals(Action.TOKEN_END) && predictedAttrValues.size() < maxAttrRealizationSize) {
            if (contentSequence.size() < partialContentSequence.getSequence().size()) {
                predictedAttr = partialContentSequence.getSequence().get(contentSequence.size()).getAttribute();
                predictedAttrValues.add(predictedAttr);

                if (!predictedAttr.equals(Action.TOKEN_END)) {
                    contentSequence.add(new Action(Action.TOKEN_START, predictedAttr, ""));
                } else {
                    contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END, ""));
                }
                contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned);

                if (!predictedAttr.isEmpty()) {
                    attrValuesAlreadyMentioned.add(predictedAttr);
                    attrValuesToBeMentioned.remove(predictedAttr);
                }
            } else if (!attrValuesToBeMentioned.isEmpty()) {
                Instance attrTrainingVector = createAttrInstance(predicate, "@TOK@", predictedAttrValues, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableContentActions);

                if (attrTrainingVector != null) {
                    Prediction predictAttr = classifierAttrs.predict(attrTrainingVector);
                    predictedAttr = predictAttr.getLabel().trim();
                    String predictedValue = "";
                    if (!predictedAttr.equals(Action.TOKEN_END)) {
                        predictedValue = chooseNextValue(predictedAttr, attrValuesToBeMentioned);

                        HashSet<String> rejectedAttrs = new HashSet<>();
                        while (predictedValue.isEmpty() && (!predictedAttr.equals(Action.TOKEN_END) || predictedAttrValues.isEmpty())) {
                            rejectedAttrs.add(predictedAttr);

                            predictedAttr = Action.TOKEN_END;
                            double maxScore = -Double.MAX_VALUE;
                            for (String attr : predictAttr.getLabel2Score().keySet()) {
                                if (!rejectedAttrs.contains(attr)
                                        && (Double.compare(predictAttr.getLabel2Score().get(attr), maxScore) > 0)) {
                                    maxScore = predictAttr.getLabel2Score().get(attr);
                                    predictedAttr = attr;
                                }
                            }
                            if (!predictedAttr.equals(Action.TOKEN_END)) {
                                predictedValue = chooseNextValue(predictedAttr, attrValuesToBeMentioned);
                            }
                        }
                        predictedAttr += "=" + predictedValue;
                    }
                    predictedAttrValues.add(predictedAttr);

                    if (!predictedAttr.equals(Action.TOKEN_END)) {
                        String attribute = predictedAttr.split("=")[0];

                        if (!attribute.equals(Action.TOKEN_END)) {
                            contentSequence.add(new Action(Action.TOKEN_START, predictedAttr, ""));
                        } else {
                            contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END, ""));
                        }
                        contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned);
                    } else {
                        contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END, ""));
                        contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned);
                    }
                    if (!predictedAttr.isEmpty()) {
                        attrValuesAlreadyMentioned.add(predictedAttr);
                        attrValuesToBeMentioned.remove(predictedAttr);
                    }
                } else {
                    predictedAttr = Action.TOKEN_END;
                    contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END, ""));
                    contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned);
                }
            } else {
                predictedAttr = Action.TOKEN_END;
                contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END, ""));
                contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned);
            }
        }
        if (!contentSequence.get(contentSequence.size() - 1).getAttribute().equals(Action.TOKEN_END)) {
            //System.out.println("ATTR ROLL-IN IS UNENDING");
            //System.out.println(contentSequence);
            contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END, ""));
            contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned);
        }
        cachedSequence = new ActionSequence(contentSequence);
        JDAggerForBagel.wordSequenceCache.put(key, cachedSequence);
        return cachedSequence;
    }

    public ActionSequence generateWordSequence(String predicate, DatasetInstance di, ActionSequence contentSequence, ActionSequence partialWordSequence, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashSet<Action>> availableWordActions) {
        ArrayList<Action> predictedActionsList = new ArrayList<>();
        ArrayList<Action> predictedWordList = new ArrayList<>();

        String predictedAttr = "";
        HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
        HashSet<String> attrValuesToBeMentioned = new HashSet<>();
        ArrayList<String> predictedAttributes = new ArrayList<>();

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
                attrValuesToBeMentioned.add(attribute + "=" + value);
            }
            valuesToBeMentioned.put(attribute, new ArrayList<>(di.getMeaningRepresentation().getAttributes().get(attribute)));
        }
        if (attrValuesToBeMentioned.isEmpty()) {
            attrValuesToBeMentioned.add("empty=empty");
        }

        ArrayList<String> sortedAlreadyMentioned = new ArrayList<>(attrValuesAlreadyMentioned);
        ArrayList<String> sortedToBeMentioned = new ArrayList<>(attrValuesToBeMentioned);
        Collections.sort(sortedAlreadyMentioned);
        Collections.sort(sortedToBeMentioned);
        String key = "WS|" + predicate + "|" + sortedAlreadyMentioned + "|" + sortedToBeMentioned + "|" + contentSequence.getSequence().toString() + "|" + partialWordSequence.getSequence().toString();
        ActionSequence cachedSequence = JDAggerForBagel.wordSequenceCache.get(key);
        if (cachedSequence != null) {
            return cachedSequence;
        }

        int a = -1;
        ArrayList<String> predictedAttrValues = contentSequence.getAttributeSequence();
        for (String attrValue : predictedAttrValues) {
            a++;
            if (!attrValue.equals(Action.TOKEN_END)) {
                String attribute = attrValue.split("=")[0];
                predictedAttributes.add(attrValue);
                ArrayList<String> nextAttributesForInstance = new ArrayList<>(predictedAttrValues.subList(a + 1, predictedAttrValues.size()));

                if (!attribute.equals(Action.TOKEN_END)) {
                    if (classifierWords.containsKey(attribute)) {
                        String predictedWord = "";
                        String predictedPOS = "";

                        boolean isValueMentioned = false;
                        String valueTBM = "";
                        if (attrValue.contains("=")) {
                            valueTBM = attrValue.substring(attrValue.indexOf('=') + 1);
                        }
                        if (valueTBM.isEmpty()) {
                            isValueMentioned = true;
                        }
                        ArrayList<String> subPhrase = new ArrayList<>();
                        while (!predictedWord.equals(Action.TOKEN_END) && predictedWordList.size() < maxWordRealizationSize) {
                            if (predictedActionsList.size() < partialWordSequence.getSequence().size()) {
                                predictedWord = partialWordSequence.getSequence().get(predictedActionsList.size()).getWord();
                                predictedPOS = partialWordSequence.getSequence().get(predictedActionsList.size()).getPOS();
                                predictedActionsList.add(new Action(predictedWord, attrValue, predictedPOS));
                                if (!predictedWord.equals(Action.TOKEN_END)) {
                                    subPhrase.add(predictedWord);
                                }
                            } else {
                                ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                                for (int i = 0; i < predictedAttributes.size() - 1; i++) {
                                    predictedAttributesForInstance.add(predictedAttributes.get(i));
                                }
                                if (!predictedAttributes.get(predictedAttributes.size() - 1).equals(attrValue)) {
                                    predictedAttributesForInstance.add(predictedAttributes.get(predictedAttributes.size() - 1));
                                }
                                Instance wordTrainingVector = createWordInstance(predicate, new Action("@TOK@", attrValue, ""), predictedAttributesForInstance, predictedActionsList, nextAttributesForInstance, attrValuesAlreadyMentioned, attrValuesToBeMentioned, isValueMentioned, availableWordActions);

                                if (wordTrainingVector != null) {
                                    if (classifierWords.get(attribute) != null) {
                                        Prediction predictWord = classifierWords.get(attribute).predict(wordTrainingVector);

                                        if (predictWord.getLabel() != null) {
                                            predictedWord = predictWord.getLabel().trim();
                                            while (predictedWord.equals(Action.TOKEN_END) && predictedActionsList.get(predictedActionsList.size() - 1).getWord().equals(Action.TOKEN_END)) {
                                                double maxScore = -Double.MAX_VALUE;
                                                for (String word : predictWord.getLabel2Score().keySet()) {
                                                    if (!word.equals(Action.TOKEN_END)
                                                            && (Double.compare(predictWord.getLabel2Score().get(word), maxScore) > 0)) {
                                                        maxScore = predictWord.getLabel2Score().get(word);
                                                        predictedWord = word;
                                                    }
                                                }
                                            }

                                            predictedPOS = "";
                                            if (predictedWord.contains(Action.POS_DELIMITER)) {
                                                predictedWord = predictedWord.trim().substring(0, predictedWord.trim().indexOf(Action.POS_DELIMITER));
                                                predictedPOS = predictedWord.trim().substring(predictedWord.trim().indexOf(Action.POS_DELIMITER) + 1);
                                            }
                                        } else {
                                            predictedWord = Action.TOKEN_END;
                                            predictedPOS = "";
                                        }
                                        predictedActionsList.add(new Action(predictedWord, attrValue, predictedPOS));
                                        predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned, predictedAttributes, nextAttributesForInstance, isValueMentioned);
                                        if (!predictedWord.equals(Action.TOKEN_START)
                                                && !predictedWord.equals(Action.TOKEN_END)) {
                                            subPhrase.add(predictedWord);
                                            predictedWordList.add(new Action(predictedWord, attrValue, predictedPOS));
                                        }
                                    } else {
                                        predictedWord = Action.TOKEN_END;
                                        predictedActionsList.add(new Action(predictedWord, attrValue, ""));
                                        predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned, predictedAttributes, nextAttributesForInstance, isValueMentioned);
                                    }
                                }
                            }
                            if (!isValueMentioned) {
                                if (!predictedWord.equals(Action.TOKEN_END)) {
                                    if (predictedWord.startsWith(Action.TOKEN_X)
                                            && (valueTBM.matches("\"[xX][0-9]+\"")
                                            || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Action.TOKEN_X))) {
                                        isValueMentioned = true;
                                    } else if (!predictedWord.startsWith(Action.TOKEN_X)
                                            && !(valueTBM.matches("\"[xX][0-9]+\"")
                                            || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Action.TOKEN_X))) {
                                        for (ArrayList<String> alignedStr : valueAlignments.get(valueTBM).keySet()) {
                                            if (endsWith(subPhrase, alignedStr)) {
                                                isValueMentioned = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (isValueMentioned) {
                                    attrValuesAlreadyMentioned.add(predictedAttr);
                                    attrValuesToBeMentioned.remove(predictedAttr);
                                }
                            }
                            String mentionedAttrValue = "";
                            if (!predictedWord.startsWith(Action.TOKEN_X)) {
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
                        if (predictedWordList.size() >= maxWordRealizationSize
                                && !predictedActionsList.get(predictedActionsList.size() - 1).getWord().equals(Action.TOKEN_END)) {
                            predictedWord = Action.TOKEN_END;
                            predictedActionsList.add(new Action(predictedWord, attrValue, ""));
                            predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned, predictedAttributes, nextAttributesForInstance, isValueMentioned);
                        }
                    } else {
                        predictedActionsList.add(new Action(Action.TOKEN_END, attrValue, ""));
                        predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned, predictedAttributes, nextAttributesForInstance, true);
                    }
                } else {
                    predictedActionsList.add(new Action(Action.TOKEN_END, Action.TOKEN_END, ""));
                    predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesToBeMentioned, attrValuesAlreadyMentioned, predictedAttributes, nextAttributesForInstance, true);
                }
            }
        }

        Action previous = null;
        boolean open = false;
        for (int i = 0; i < predictedActionsList.size(); i++) {
            if (previous != null) {
                if (!predictedActionsList.get(i).getAttribute().equals(previous.getAttribute())
                        && !previous.getWord().equals(Action.TOKEN_END)) {
                    open = true;
                }
                if (predictedActionsList.get(i).getWord().equals(Action.TOKEN_END)
                        && previous.getWord().equals(Action.TOKEN_END)) {
                    open = true;
                }
            }
            previous = predictedActionsList.get(i);
        }
        if (open) {
            /*System.out.println("====!======ROLL IN END OR OPEN===========");
            System.out.println(predictedActionsList);
            System.out.println("Input Content: " + contentSequence);
            System.out.println("Input Words " + partialWordSequence);
            System.exit(0);*/
 /*}
        cachedSequence = new ActionSequence(predictedActionsList);
        JDAggerForBagel.wordSequenceCache.put(key, cachedSequence);
        return cachedSequence;
    }*/
    /**
     *
     * @param wordSequence
     * @return
     */
    public ArrayList<String> getPredictedAttrList(ArrayList<Action> wordSequence) {
        ArrayList<Action> cleanActionList = new ArrayList<>();
        for (Action action : wordSequence) {
            if (!action.getWord().equals(Action.TOKEN_START)
                    && !action.getWord().equals(Action.TOKEN_END)) {
                cleanActionList.add(action);
            }
        }

        ArrayList<String> predictedAttrList = new ArrayList<>();
        for (Action action : cleanActionList) {
            if (predictedAttrList.isEmpty()) {
                predictedAttrList.add(action.getAttribute());
            } else if (!predictedAttrList.get(predictedAttrList.size() - 1).equals(action.getAttribute())) {
                predictedAttrList.add(action.getAttribute());
            }
        }
        return predictedAttrList;
    }

    /**
     *
     * @return
     */
    public boolean loadLists() {
        String file1 = "predicates_Bagel";
        String file2 = "attributes_Bagel";
        String file3 = "attributeValuePairs_Bagel";
        String file4 = "valueAlignments_Bagel";
        String file5 = "datasetInstances_Bagel";
        String file6 = "maxLengths_Bagel";
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        FileInputStream fin3 = null;
        ObjectInputStream ois3 = null;
        FileInputStream fin4 = null;
        ObjectInputStream ois4 = null;
        FileInputStream fin5 = null;
        ObjectInputStream ois5 = null;
        FileInputStream fin6 = null;
        ObjectInputStream ois6 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()
                && (new File(file3)).exists()
                && (new File(file4)).exists()
                && (new File(file5)).exists()
                && (new File(file6)).exists()) {
            try {
                System.out.print("Load lists...");
                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                if (predicates == null) {
                    if (o1 instanceof ArrayList) {
                        predicates = new ArrayList<String>((ArrayList<String>) o1);
                    }
                } else if (o1 instanceof ArrayList) {
                    predicates.addAll((ArrayList<String>) o1);
                }
                ///////////////////
                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (availableAttributeActions == null) {
                    if (o2 instanceof HashMap) {
                        availableAttributeActions = new HashMap<String, HashSet<String>>((HashMap<String, HashSet<String>>) o2);
                    }
                } else if (o2 instanceof HashMap) {
                    availableAttributeActions.putAll((HashMap<String, HashSet<String>>) o2);
                }
                ///////////////////
                fin3 = new FileInputStream(file3);
                ois3 = new ObjectInputStream(fin3);
                Object o3 = ois3.readObject();
                if (attributeValuePairs == null) {
                    if (o3 instanceof HashMap) {
                        attributeValuePairs = new HashMap<String, HashSet<String>>((HashMap<String, HashSet<String>>) o3);
                    }
                } else if (o3 instanceof HashMap) {
                    attributeValuePairs.putAll((HashMap<String, HashSet<String>>) o3);
                }
                ///////////////////
                fin4 = new FileInputStream(file4);
                ois4 = new ObjectInputStream(fin4);
                Object o4 = ois4.readObject();
                if (valueAlignments == null) {
                    if (o4 instanceof HashMap) {
                        valueAlignments = new HashMap<String, HashMap<ArrayList<String>, Double>>((HashMap<String, HashMap<ArrayList<String>, Double>>) o4);
                    }
                } else if (o4 instanceof HashMap) {
                    valueAlignments.putAll((HashMap<String, HashMap<ArrayList<String>, Double>>) o4);
                }
                ///////////////////
                fin5 = new FileInputStream(file5);
                ois5 = new ObjectInputStream(fin5);
                Object o5 = ois5.readObject();
                if (datasetInstances == null) {
                    if (o5 instanceof HashMap) {
                        datasetInstances = new HashMap<String, ArrayList<DatasetInstance>>((HashMap<String, ArrayList<DatasetInstance>>) o5);
                    } else {
                        return false;
                    }
                } else if (o5 instanceof HashMap) {
                    datasetInstances.putAll((HashMap<String, ArrayList<DatasetInstance>>) o5);
                }
                ///////////////////
                fin6 = new FileInputStream(file6);
                ois6 = new ObjectInputStream(fin6);
                Object o6 = ois6.readObject();
                ArrayList<Integer> lengths = new ArrayList<Integer>((ArrayList<Integer>) o6);
                maxAttrRealizationSize = lengths.get(0);
                maxWordRealizationSize = lengths.get(1);
                System.out.println("done!");
            } catch (ClassNotFoundException ex) {
                ex.printStackTrace();
            } catch (IOException ex) {
                ex.printStackTrace();
            } finally {
                try {
                    fin1.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
                try {
                    ois1.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     *
     */
    public void writeLists() {
        String file1 = "predicates_Bagel";
        String file2 = "attributes_Bagel";
        String file3 = "attributeValuePairs_Bagel";
        String file4 = "valueAlignments_Bagel";
        String file5 = "datasetInstances_Bagel";
        String file6 = "maxLengths_Bagel";
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        FileOutputStream fout3 = null;
        ObjectOutputStream oos3 = null;
        FileOutputStream fout4 = null;
        ObjectOutputStream oos4 = null;
        FileOutputStream fout5 = null;
        ObjectOutputStream oos5 = null;
        FileOutputStream fout6 = null;
        ObjectOutputStream oos6 = null;
        try {
            System.out.print("Write lists...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(predicates);
            ///////////////////
            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(availableAttributeActions);
            ///////////////////
            fout3 = new FileOutputStream(file3);
            oos3 = new ObjectOutputStream(fout3);
            oos3.writeObject(attributeValuePairs);
            ///////////////////
            fout4 = new FileOutputStream(file4);
            oos4 = new ObjectOutputStream(fout4);
            oos4.writeObject(valueAlignments);
            ///////////////////
            fout5 = new FileOutputStream(file5);
            oos5 = new ObjectOutputStream(fout5);
            oos5.writeObject(datasetInstances);
            ///////////////////
            fout6 = new FileOutputStream(file6);
            oos6 = new ObjectOutputStream(fout6);
            ArrayList<Integer> lengths = new ArrayList<Integer>();
            lengths.add(maxAttrRealizationSize);
            lengths.add(maxWordRealizationSize);
            oos6.writeObject(lengths);
            System.out.println("done!");
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                fout1.close();
                fout2.close();
                fout3.close();
                fout4.close();
                fout5.close();
                fout6.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
            try {
                oos1.close();
                oos2.close();
                oos3.close();
                oos4.close();
                oos5.close();
                oos6.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    /**
     *
     * @return
     */
    public boolean loadLMs() {
        String file1 = "wordLM_Bagel_" + fold;
        String file2 = "wordLMs_Bagel_" + fold;
        String file3 = "attrLMs_Bagel_" + fold;
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        FileInputStream fin3 = null;
        ObjectInputStream ois3 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()
                && (new File(file3)).exists()) {
            try {
                System.out.print("Load language models...");
                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();

                wordLM = (SimpleLM) o1;

                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (wordLMsPerPredicate == null) {
                    if (o2 instanceof HashMap) {
                        wordLMsPerPredicate = new HashMap<String, SimpleLM>((HashMap<String, SimpleLM>) o2);
                    }
                } else if (o2 instanceof HashMap) {
                    wordLMsPerPredicate.putAll((HashMap<String, SimpleLM>) o2);
                }

                fin3 = new FileInputStream(file3);
                ois3 = new ObjectInputStream(fin3);
                Object o3 = ois3.readObject();

                if (attrLMsPerPredicate == null) {
                    if (o3 instanceof HashMap) {
                        attrLMsPerPredicate = new HashMap<String, SimpleLM>((HashMap<String, SimpleLM>) o3);
                    }
                } else if (o3 instanceof HashMap) {
                    attrLMsPerPredicate.putAll((HashMap<String, SimpleLM>) o3);
                }
                System.out.println("done!");
            } catch (ClassNotFoundException ex) {
                ex.printStackTrace();
            } catch (IOException ex) {
                ex.printStackTrace();
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                    fin3.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
                try {
                    ois1.close();
                    ois2.close();
                    ois3.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
        } else {
            return false;
        }
        return true;
    }

    /**
     *
     */
    public void writeLMs() {
        String file1 = "wordLM_Bagel_" + fold;
        String file2 = "wordLMs_Bagel_" + fold;
        String file3 = "attrLMs_Bagel_" + fold;
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        FileOutputStream fout3 = null;
        ObjectOutputStream oos3 = null;
        try {
            System.out.print("Write LMs...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(wordLM);

            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(wordLMsPerPredicate);

            fout3 = new FileOutputStream(file3);
            oos3 = new ObjectOutputStream(fout3);
            oos3.writeObject(attrLMsPerPredicate);
            System.out.println("done!");
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                fout1.close();
                fout2.close();
                fout3.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
            try {
                oos1.close();
                oos2.close();
                oos3.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    /**
     *
     * @return
     */
    public boolean loadTrainingData() {
        String file1 = "attrTrainingData_Bagel_" + fold + "_" + useSubsetData;
        String file2 = "wordTrainingData_Bagel_" + fold + "_" + useSubsetData;
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()) {
            try {
                System.out.print("Load training data...");
                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                if (predicateAttrTrainingData == null) {
                    if (o1 instanceof HashMap) {
                        predicateAttrTrainingData = new HashMap<String, ArrayList<Instance>>((HashMap<String, ArrayList<Instance>>) o1);
                    }
                } else if (o1 instanceof HashMap) {
                    predicateAttrTrainingData.putAll((HashMap<String, ArrayList<Instance>>) o1);
                }

                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (predicateWordTrainingData == null) {
                    if (o2 instanceof HashMap) {
                        predicateWordTrainingData = new HashMap<String, HashMap<String, ArrayList<Instance>>>((HashMap<String, HashMap<String, ArrayList<Instance>>>) o2);
                    }
                } else if (o2 instanceof HashMap) {
                    predicateWordTrainingData.putAll((HashMap<String, HashMap<String, ArrayList<Instance>>>) o2);
                }

                System.out.println("done!");
            } catch (ClassNotFoundException ex) {
                ex.printStackTrace();
            } catch (IOException ex) {
                ex.printStackTrace();
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
                try {
                    ois1.close();
                    ois2.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
        } else {
            return false;
        }
        return true;
    }

    /**
     *
     */
    public void writeTrainingData() {
        String file1 = "attrTrainingData_Bagel_" + fold + "_" + useSubsetData;
        String file2 = "wordTrainingData_Bagel_" + fold + "_" + useSubsetData;
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        try {
            System.out.print("Write Training Data...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(predicateAttrTrainingData);

            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(predicateWordTrainingData);

            System.out.println("done!");
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                fout1.close();
                fout2.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
            try {
                oos1.close();
                oos2.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    /**
     *
     * @param dataSize
     * @param trainedAttrClassifiers_0
     * @param trainedWordClassifiers_0
     * @return
     */
    public boolean loadInitClassifiers(int dataSize, HashMap<String, JAROW> trainedAttrClassifiers_0, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0) {
        String file1 = "attrInitClassifiers_BAGEL_" + fold + "_" + dataSize;
        String file2 = "wordInitClassifiers_BAGEL_" + fold + "_" + dataSize;
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()) {
            try {
                System.out.print("Load initial classifiers...");
                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                trainedAttrClassifiers_0 = (HashMap<String, JAROW>) o1;
                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (o2 instanceof HashMap) {
                    trainedWordClassifiers_0.putAll((HashMap<String, HashMap<String, JAROW>>) o2);
                }

                System.out.println("done!");
            } catch (ClassNotFoundException ex) {
                ex.printStackTrace();
            } catch (IOException ex) {
                ex.printStackTrace();
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
                try {
                    ois1.close();
                    ois2.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
        } else {
            return false;
        }
        return true;
    }

    /**
     *
     * @param dataSize
     * @param trainedAttrClassifiers_0
     * @param trainedWordClassifiers_0
     */
    public void writeInitClassifiers(int dataSize, HashMap<String, JAROW> trainedAttrClassifiers_0, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0) {
        String file1 = "attrInitClassifiers_BAGEL_" + fold + "_" + dataSize;
        String file2 = "wordInitClassifiers_BAGEL_" + fold + "_" + dataSize;
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        try {
            System.out.print("Write initial classifiers...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(trainedAttrClassifiers_0);

            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(trainedWordClassifiers_0);

            System.out.println("done!");
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                fout1.close();
                fout2.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
            try {
                oos1.close();
                oos2.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }
}
