/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uk.ac.ucl.jdagger;

import edu.stanford.nlp.pipeline.Annotation;
import gnu.trove.map.hash.THashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import similarity_measures.Levenshtein;
import uk.ac.ucl.jarow.Action;
import uk.ac.ucl.jarow.ActionSequence;
import uk.ac.ucl.jarow.Bagel;
import static uk.ac.ucl.jarow.Bagel.createWordInstance;
import static uk.ac.ucl.jarow.Bagel.endsWith;
import static uk.ac.ucl.jarow.Bagel.maxRealizationSize;
import uk.ac.ucl.jarow.DatasetInstance;
import uk.ac.ucl.jarow.Instance;
import uk.ac.ucl.jarow.JAROW;
import uk.ac.ucl.jarow.MeaningRepresentation;
import uk.ac.ucl.jarow.Prediction;

public class JDAggerForBagel {

    final static int threadsCount = Runtime.getRuntime().availableProcessors() * 2;
    public static int ep = 0;
    public static boolean train = false;

    public JDAggerForBagel() {
    }
    public static Random r = new Random();
    public static HashMap<ActionSequence, ActionSequence> rollOutCache = new HashMap<>();

    /*public JAROW runVDAggerForWords(String predicate, ArrayList<DatasetInstance> trainingDatasetInstances, ArrayList<Action> availableActions, int epochs, double beta) {
    JAROW classifierWords = null;//trainClassifier(trainingWordInstances);
    ArrayList<Instance> newTrainingInstances = new ArrayList();
    for (int i = 1; i <= epochs; i++) {
    ep = i;
    rollOutCache = new HashMap<>();
    System.out.println("Starting epoch " + i);
    long startTime = System.currentTimeMillis();
    
    double p = Math.pow(1.0 - beta, (double) i - 1);
    System.out.println("p = " + p);
    
    //ArrayList<Instance> newTrainingWordInstances = new ArrayList();
    //CHANGE
    for (DatasetInstance di : trainingDatasetInstances) {
    for (ArrayList<String> realization : di.getRealizationsToAlignments().keySet()) {
    ArrayList<String> alignment = di.getRealizationsToAlignments().get(realization);
    ArrayList<Action> as = new ArrayList<>();
    for (String s : realization) {
    as.add(new Action(s));
    }
    as.add(new Action(Bagel.TOKEN_END));
    ActionSequence ref = new ActionSequence(as, 0.0);
    
    //ROLL-IN
    double v = r.nextDouble();
    boolean useReferenceRollIn = false;
    if (v <= p) {
    useReferenceRollIn = true;
    }
    ActionSequence actSeq = getPolicyRollIn(predicate, di.getMeaningRepresentation(), classifierWords, useReferenceRollIn, ref);
    
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
    
    //FOR EVERY ACTION IN THE SEQUENCE
    for (int a = 0; a < actSeq.getSequence().size(); a++) {
    if (a < ref.getSequence().size() + 2) {
    TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
    availableActions.stream().forEach((action) -> {
    costs.put(action.getDecision(), 1.0);
    });
    
    //MODIFY IT TO EACH POSSIBLE AVAILABLE ACTION           
    ExecutorService executor = Executors.newFixedThreadPool(threadsCount);
    for (Action availableAction : availableActions) {
    executor.execute(new ReferenceRollOutThread(actSeq, costs, a, availableAction, ref));
    }
    executor.shutdown();
    while (!executor.isTerminated()) {
    }
    //GENERATE NEW TRAINING EXAMPLE
    ActionSequence rollOutSeq = new ActionSequence(actSeq);
    rollOutSeq.modifyAndShortenSequence(a, Bagel.TOKEN_END);
    rollOutSeq.getSequence().remove(rollOutSeq.getSequence().size() - 1);
    
    //train = true;
    newTrainingInstances.add(generateTrainingInstance(predicate, meaningRepr, availableActions, rollOutSeq, a, costs));
    //train = false;
    //System.exit(0);
    }
    }
    }
    //System.out.println("|-> " + modActSeq.getSequenceToString());
    //System.out.println("C " + modActSeq.getCost());
    }
    Collections.shuffle(newTrainingInstances);
    //if (classifierWords == null) {
    classifierWords = trainClassifier(newTrainingInstances);
    /long endTime = System.currentTimeMillis();
    long totalTime = endTime - startTime;
    SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss");
    Date resultdate = new Date(endTime);
    
    System.out.println("Epoch after: " + totalTime / 1000 / 60 + " mins, " + sdf.format(resultdate));
    }
    //multiTrainClassifier(newTrainingInstances);
    return classifierWords;
    }*/
    public static HashMap<String, JAROW> runLOLS(String predicate, HashSet<String> attributes, ArrayList<DatasetInstance> trainingDatasetInstances, HashMap<String, ArrayList<Instance>> trainingWordInstances, HashMap<String, HashMap<String, HashSet<Action>>> availableActions, HashMap<String, HashSet<ArrayList<String>>> valueAlignments, double beta) {

        ArrayList<HashMap<String, JAROW>> trainedClassifiers = new ArrayList();
        //INITIALIZE A POLICY P_0 (initializing on ref)
        HashMap<String, JAROW> trainedClassifiers_0 = new HashMap<>();
        for (String attribute : attributes) {
            if (!trainingWordInstances.get(attribute).isEmpty()) {
                trainedClassifiers_0.put(attribute, trainClassifier(trainingWordInstances.get(attribute)));
            } else {
                System.out.println("EMPTY " + attribute);
            }
        }
        trainedClassifiers.add(trainedClassifiers_0);
        //for (int i = 1; i <= epochs; i++) {
        HashMap<String, ArrayList<Instance>> newTrainingInstances = new HashMap<>();
        for (String attr : attributes) {
            newTrainingInstances.put(attr, new ArrayList<Instance>(trainingWordInstances.get(attr)));
        }
        int c = 0;
        for (DatasetInstance di : trainingDatasetInstances) {
            if (c < 1) {
                c++;
                HashMap<String, JAROW> trainedClassifiers_i = trainedClassifiers.get(trainedClassifiers.size() - 1);

                //Initialize new training set
                /*HashMap<String, ArrayList<Instance>> newTrainingInstances = new HashMap<>();
                for (String attr : trainedClassifiers_i.keySet()) {
                newTrainingInstances.put(attr, new ArrayList<Instance>());
                }*/
                //ROLL-IN
                ActionSequence actSeq = getLearnedPolicyRollIn(predicate, di, trainedClassifiers_i, valueAlignments);

                System.out.println(di.getMeaningRepresentation().getAttributes());
                System.out.println(actSeq.getWordSequenceToString());
                System.out.println(actSeq.getAttributeSequenceToString());
                System.out.println("==========================");

                //FOR EACH ACTION IN ROLL-IN SEQUENCE
                //The number of actions is not definite...might cause issues
                for (int index = 0; index < actSeq.getSequence().size(); index++) {
                    //FOR EACH POSSIBLE ALTERNATIVE ACTION
                    TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                    for (Action action : availableActions.get(predicate).get(actSeq.getSequence().get(index).getAttribute())) {
                        costs.put(action.getWord(), 1.0);
                    }

                    //Make the same decisions for all action substitutions
                    boolean useReferenceRollout = false;
                    double v = r.nextDouble();
                    if (v > beta) {
                        useReferenceRollout = true;
                    }

                    String previousWord = "";
                    if (index > 0) {
                        previousWord = actSeq.getSequence().get(index).getWord().trim().toLowerCase();
                    }
                    for (Action availableAction : availableActions.get(predicate).get(actSeq.getSequence().get(index).getAttribute())) {
                        if (!availableAction.getWord().trim().toLowerCase().equals(previousWord)) {
                            //if (!availableAction.getWord().equals(actSeq.getSequence().get(index).getWord())) {
                            ActionSequence modSeq = new ActionSequence(actSeq);
                            System.out.println("->| " + modSeq.getWordSequenceToString());
                            modSeq.modifyAndShortenSequence(index, availableAction.getWord());
                            System.out.println("A " + availableAction.getWord());
                            System.out.println("M " + modSeq.getWordSequenceToString());
                            //ROLL-OUT
                            costs.put(availableAction.getWord().trim().toLowerCase(), getPolicyRollOutCost(predicate, modSeq, di, trainedClassifiers_i, valueAlignments, useReferenceRollout));
                            System.out.println(costs.get(availableAction.getWord().trim().toLowerCase()));
                            //}
                        } else {
                            costs.put(availableAction.getWord().trim().toLowerCase(), 1.0);
                        }
                    }

                    //GENERATE NEW TRAINING EXAMPLE
                    newTrainingInstances.get(actSeq.getSequence().get(index).getAttribute()).add(generateTrainingInstance(predicate, di, actSeq, index, valueAlignments, costs));
                }

                //UPDATE CLASSIFIER            
                HashMap<String, JAROW> trainedClassifiers_ii = new HashMap<String, JAROW>();
                for (String attr : trainedClassifiers_i.keySet()) {
                    Collections.shuffle(newTrainingInstances.get(attr));
                    
                    //trainedClassifiers_ii.put(attr, new JAROW(trainedClassifiers_i.get(attr)));
                    //trainedClassifiers_ii.get(attr).trainAdditional(newTrainingInstances.get(attr), true, true, 10, 100.0, true);
                    //trainedClassifiers_ii.get(attr).train(newTrainingInstances.get(attr), true, true, 10, 100.0, true);
                    
                    Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
                    JAROW classifierWords = JAROW.trainOpt(newTrainingInstances.get(attr), Bagel.rounds, params, 0.1, true, false, 0);
                    trainedClassifiers_ii.put(attr, classifierWords);
                }

                trainedClassifiers.add(trainedClassifiers_ii);
            }
        }
        //}

        //FIRST NEED TO AVERAGE OVER ALL CLASSIFIERS
        HashMap<String, ArrayList<JAROW>> reorganizedClassifiersWords = new HashMap<>();
        for (String attribute : attributes) {
            reorganizedClassifiersWords.put(attribute, new ArrayList<JAROW>());
        }
        for (HashMap<String, JAROW> trainedClassifiers_i : trainedClassifiers) {
            for (String attribute : trainedClassifiers_i.keySet()) {
                reorganizedClassifiersWords.get(attribute).add(trainedClassifiers_i.get(attribute));
            }
        }
        HashMap<String, JAROW> avgClassifiersWords = new HashMap<>();
        for (String attribute : attributes) {
            JAROW avg = new JAROW();
            avg.averageOverClassifiers(reorganizedClassifiersWords.get(attribute));

            if (!reorganizedClassifiersWords.get(attribute).isEmpty()) {
                avgClassifiersWords.put(attribute, avg);
            }
        }

        return avgClassifiersWords;
    }

    /*public static ActionSequence getPolicyRollIn(String predicate, MeaningRepresentation mr, JAROW classifierWords, double p, ActionSequence ref) {
    double v = r.nextDouble();
    
    if (v <= p) {
    return getReferencePolicyRollIn(ref);
    } else {
    return getLearnedPolicyRollIn(predicate, mr, classifierWords, ref);
    }
    }
    
    public static ActionSequence getPolicyRollIn(String predicate, MeaningRepresentation mr, JAROW classifierWords, boolean useReferenceRollin, ActionSequence ref) {
    if (useReferenceRollin) {
    return getReferencePolicyRollIn(ref);
    } else {
    return getLearnedPolicyRollIn(predicate, mr, classifierWords, ref);
    }
    }*/
    public static Double getPolicyRollOutCost(String predicate, ActionSequence actSeq, DatasetInstance di, HashMap<String, JAROW> classifierWords, HashMap<String, HashSet<ArrayList<String>>> valueAlignments, double p) {
        double v = r.nextDouble();

        if (v <= p) {
            return getReferencePolicyRollOutCost(actSeq, di);
        } else {
            return getLearnedPolicyRollOut(actSeq, predicate, di, classifierWords, valueAlignments);
        }
    }

    public static Double getPolicyRollOutCost(String predicate, ActionSequence actSeq, DatasetInstance di, HashMap<String, JAROW> classifierWords, HashMap<String, HashSet<ArrayList<String>>> valueAlignments, boolean useReferenceRollout) {
        if (useReferenceRollout) {
            return getReferencePolicyRollOutCost(actSeq, di);
        } else {
            return getLearnedPolicyRollOut(actSeq, predicate, di, classifierWords, valueAlignments);
        }
    }

    public static ActionSequence getReferencePolicyRollIn(ActionSequence ref) {
        return new ActionSequence(ref);
    }

    public static Double getReferencePolicyRollOutCost(ActionSequence pAS, DatasetInstance di) {
        int maxRefLength = 0;
        HashSet<ActionSequence> refs = new HashSet<>();
        for (ArrayList<Action> ref : di.getRealizations()) {
            refs.add(new ActionSequence(ref, 0.0));
            if (ref.size() > maxRefLength) {
                maxRefLength = ref.size();
            }
        }

        pAS.recalculateCost(refs);

        if (pAS.getSequence().size() > 1 && pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(pAS.getSequence().get(pAS.getSequence().size() - 2).getWord())) {
            //Do not repeat the same word twice in a row
            return 1.0;
        } else if (pAS.getSequence().size() > maxRefLength) {
            //Do not exceed (or plan to exceed) the length of the reference
            //*** Perhaps this needs to be more elegant, too tired now
            ActionSequence newAS = new ActionSequence(pAS);
            if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(Bagel.TOKEN_END)) {
                newAS.recalculateCost(refs);
            } else {
                newAS.setCost(1.0);
            }
            return newAS.getCost();
        } else if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(Bagel.TOKEN_END)) {
            ActionSequence newAS = new ActionSequence(pAS);
            newAS.recalculateCost(refs);
            return newAS.getCost();
        } else {
            double minCost = Double.MAX_VALUE;
            for (ActionSequence minRAS : refs) {
                //Let;s assume for now that the only correct response is the particular sentence that corresponds to the meaning representation in the data
                //Lets also assume that a good cost estimation is comparing the rollin (plus mutation) to the correct responce cut to the same length
                //That is because we assume that all next words will be best and not make any impression to the score (reference rollout)
                ActionSequence newAS = new ActionSequence();
                for (int i = 0; i < pAS.getSequence().size(); i++) {
                    if (i < minRAS.getSequence().size()) {
                        newAS.getSequence().add(new Action(minRAS.getSequence().get(i).getWord(), minRAS.getSequence().get(i).getAttribute()));
                    }
                }
                newAS.setCost(Levenshtein.getNormDistance(pAS.getWordSequenceToString(), newAS.getWordSequenceToString(), minRAS.getSequence().size()));

                if (newAS.getCost() < minCost) {
                    minCost = newAS.getCost();
                }
            }
            return minCost;
        }
    }

    public static ActionSequence getLearnedPolicyRollIn(String predicate, DatasetInstance di, HashMap<String, JAROW> classifierWords, HashMap<String, HashSet<ArrayList<String>>> valueAlignments) {
        ArrayList<Action> predictedActionsList = new ArrayList<>();
        ArrayList<String> mentionedValueSequence = null;
        for (ArrayList<String> m : di.getMentionedValueSequences()) {
            mentionedValueSequence = m;
            break;
        }
        ArrayList<String> predictedAttributes = new ArrayList<>();
        HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
        for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
            valuesToBeMentioned.put(attribute, new ArrayList(di.getMeaningRepresentation().getAttributes().get(attribute)));
        }

        for (int a = 0; a < mentionedValueSequence.size(); a++) {
            String attribute = mentionedValueSequence.get(a).split("=")[0];
            predictedAttributes.add(attribute);

            if (!predictedAttributes.get(a).equals(Bagel.TOKEN_END)) {
                String predictedWord = "";
                int wW = predictedActionsList.size();

                boolean valueWasMentioned = false;
                ArrayList<String> subPhrase = new ArrayList<>();
                while (!predictedWord.equals(Bagel.TOKEN_END) && predictedActionsList.size() < maxRealizationSize) {
                    ArrayList<Action> tempList = new ArrayList(predictedActionsList);
                    tempList.add(new Action("@TOK@", predictedAttributes.get(a)));
                    Instance wordTrainingVector = createWordInstance(predicate, predictedAttributes, a, tempList, wW, valuesToBeMentioned.get(attribute), di.getMeaningRepresentation(), false);

                    if (wordTrainingVector != null) {
                        if (classifierWords.get(attribute) != null) {
                            Prediction predict = classifierWords.get(attribute).predict(wordTrainingVector);
                            predictedWord = predict.getLabel().trim();
                            predictedActionsList.add(new Action(predictedWord, attribute));
                            if (!predictedWord.equals(Bagel.TOKEN_END)) {
                                subPhrase.add(predictedWord);
                            }
                        } else {
                            predictedWord = Bagel.TOKEN_END;
                            predictedActionsList.add(new Action(predictedWord, attribute));
                        }
                    }
                    if (!valuesToBeMentioned.get(attribute).isEmpty()) {
                        if (predictedWord.equals(Bagel.TOKEN_X)) {
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
                                    for (ArrayList<String> alignedStr : valueAlignments.get(value)) {
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
                predictedActionsList.remove(predictedActionsList.size() - 1);
            }
        }
        HashSet<ActionSequence> refs = new HashSet<>();
        for (ArrayList<Action> ref : di.getRealizations()) {
            refs.add(new ActionSequence(ref, 0.0));
        }

        return new ActionSequence(predictedActionsList, refs);
    }

    public static double getLearnedPolicyRollOut(ActionSequence pAS, String predicate, DatasetInstance di, HashMap<String, JAROW> classifierWords, HashMap<String, HashSet<ArrayList<String>>> valueAlignments) {
        ArrayList<Action> predictedActionsList = new ArrayList<>();
        ArrayList<String> mentionedValueSequence = null;
        for (ArrayList<String> m : di.getMentionedValueSequences()) {
            mentionedValueSequence = m;
            break;
        }
        ArrayList<String> predictedAttributes = new ArrayList<>();
        HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
        for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
            valuesToBeMentioned.put(attribute, new ArrayList(di.getMeaningRepresentation().getAttributes().get(attribute)));
        }

        int s = 0;
        for (int a = 0; a < mentionedValueSequence.size(); a++) {
            String attribute = mentionedValueSequence.get(a).split("=")[0];
            predictedAttributes.add(attribute);

            if (!predictedAttributes.get(a).equals(Bagel.TOKEN_END)) {
                String predictedWord = "";
                int wW = predictedActionsList.size();

                boolean valueWasMentioned = false;
                ArrayList<String> subPhrase = new ArrayList<>();
                while (!predictedWord.equals(Bagel.TOKEN_END) && predictedActionsList.size() < maxRealizationSize) {
                    ArrayList<Action> tempList = new ArrayList(predictedActionsList);
                    tempList.add(new Action("@TOK@", predictedAttributes.get(a)));
                    Instance wordTrainingVector = createWordInstance(predicate, predictedAttributes, a, tempList, wW, valuesToBeMentioned.get(attribute), di.getMeaningRepresentation(), false);

                    if (wordTrainingVector != null) {
                        if (classifierWords.get(attribute) != null) {
                            if (s < pAS.getSequence().size()) {
                                predictedWord = pAS.getSequence().get(s).getWord();
                            } else {
                                Prediction predict = classifierWords.get(attribute).predict(wordTrainingVector);
                                predictedWord = predict.getLabel().trim();
                            }
                            predictedActionsList.add(new Action(predictedWord, attribute));
                            if (!predictedWord.equals(Bagel.TOKEN_END)) {
                                subPhrase.add(predictedWord);
                            }
                        } else {
                            predictedWord = Bagel.TOKEN_END;
                            predictedActionsList.add(new Action(predictedWord, attribute));
                        }
                    }
                    if (!valuesToBeMentioned.get(attribute).isEmpty()) {
                        if (predictedWord.equals(Bagel.TOKEN_X)) {
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
                                    for (ArrayList<String> alignedStr : valueAlignments.get(value)) {
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
                predictedActionsList.remove(predictedActionsList.size() - 1);
            }
        }
        HashSet<ActionSequence> refs = new HashSet<>();
        for (ArrayList<Action> ref : di.getRealizations()) {
            refs.add(new ActionSequence(ref, 0.0));
        }

        return new ActionSequence(predictedActionsList, refs).getCost();
    }

    public static Instance generateTrainingInstance(String predicate, DatasetInstance di, ActionSequence modActSeq, int index, HashMap<String, HashSet<ArrayList<String>>> valueAlignments, TObjectDoubleHashMap<String> costs) {
        ArrayList<Action> predictedWordsList = new ArrayList<>();
        HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
        for (String attribute : di.getMeaningRepresentation().getAttributes().keySet()) {
            valuesToBeMentioned.put(attribute, new ArrayList(di.getMeaningRepresentation().getAttributes().get(attribute)));
        }

        for (int i = 0; i <= index; i++) {
            String attribute = modActSeq.getSequence().get(i).getAttribute();

            boolean valueWasMentioned = false;
            ArrayList<String> subPhrase = new ArrayList<>();
            if (!attribute.equals(Bagel.TOKEN_END)) {
                String predictedWord = modActSeq.getSequence().get(i).getWord();
                if (!predictedWord.equals(Bagel.TOKEN_END)) {
                    subPhrase.add(predictedWord);
                }

                predictedWordsList.add(new Action(predictedWord, attribute));
                if (!valuesToBeMentioned.get(attribute).isEmpty()) {
                    if (predictedWord.equals(Bagel.TOKEN_X)) {
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
                                for (ArrayList<String> alignedStr : valueAlignments.get(value)) {
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
            }
        }
        ArrayList<String> predictedAttributes = new ArrayList<>();
        for (int i = 0; i < index; i++) {
            Action act = predictedWordsList.get(i);
            if (predictedAttributes.isEmpty()) {
                if (!act.getAttribute().equals(Bagel.TOKEN_END)) {
                    predictedAttributes.add(act.getAttribute());
                }
            } else {
                if (!act.getAttribute().equals(Bagel.TOKEN_END) && !act.getAttribute().equals(predictedAttributes.get(predictedAttributes.size() - 1))) {
                    predictedAttributes.add(act.getAttribute());
                }
            }
        }
        return Bagel.createWordInstance(predicate, predictedAttributes, predictedAttributes.size(), predictedWordsList, index, costs, valuesToBeMentioned.get(modActSeq.getSequence().get(index).getAttribute()), di.getMeaningRepresentation());
    }

    /*public static Instance generateTrainingInstance(String predicate, MeaningRepresentation meaningRepr, ArrayList<Action> availableActions, ActionSequence modActSeq, int index, TObjectDoubleHashMap<String> costs) {
    HashMap<String, Boolean> argumentsToBeMentioned = new HashMap<>();
    for (String argument : meaningRepr.getAttributes().keySet()) {
    argumentsToBeMentioned.put(argument, true);
    }
    ArrayList<String> predictedWordsList = new ArrayList<>();
    for (int i = 0; i <= index - 1; i++) {
    String predictedWord = modActSeq.getSequence().get(i).getWord();
    
    predictedWordsList.add(predictedWord);
    for (String arg : argumentsToBeMentioned.keySet()) {
    if (predictedWord.equals(arg)) {
    argumentsToBeMentioned.put(arg, false);
    }
    }
    }
    return Bagel.createWordInstance(predicate, predictedWordsList, index, costs, argumentsToBeMentioned);
    }*/
    public static JAROW trainClassifier(ArrayList<Instance> trainingWordInstances) {
        /*JAROW classifierWords = new JAROW();
        classifierWords.train(trainingWordInstances, true, true, 10, 100.0, true);
                */

        Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
        JAROW classifierWords = JAROW.trainOpt(trainingWordInstances, Bagel.rounds, params, 0.1, true, false, 0);
        return classifierWords;
    }

    public static void multiTrainClassifier(ArrayList<Instance> trainingWordInstances) {
        System.out.println("Start Multi");

        for (int i = 0; i < 10; i++) {
            JAROW classifierWords = new JAROW();
            Collections.shuffle(trainingWordInstances);
            System.out.println(classifierWords.train(trainingWordInstances, true, true, Bagel.rounds, 0.1, true));
        }

        System.out.println("End Multi");

    }
}
