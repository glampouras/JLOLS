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
package jdagger;

import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import similarity_measures.Levenshtein;
import imitationNLG.Action;
import imitationNLG.ActionSequence;
import imitationNLG.Bagel;
import imitationNLG.DatasetInstance;
import jarow.Instance;
import jarow.JAROW;
import jarow.Prediction;

public class JDAggerForBagel {

    final static int threadsCount = 4;

    public JDAggerForBagel(Bagel bagel) {
        this.bagel = bagel;
    }
    boolean print = false;
    static boolean adapt = false;
    public static double p = 0.0;
    public static double param = 0.0;
    public static double earlyStopMaxFurtherSteps = 0;
    static int rollOutWindowSize = 5;
    public static int checkIndex = -1;
    Bagel bagel = null;

    public Object[] runLOLS(String predicate, HashSet<String> attributes, ArrayList<DatasetInstance> trainingData, ArrayList<Instance> trainingAttrInstances, HashMap<String, ArrayList<Instance>> trainingWordInstances, HashMap<String, HashSet<Action>> availableWordActions, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, double beta, ArrayList<DatasetInstance> testingData, HashMap<Integer, HashSet<String>> nGrams) {
        //param = 1000;
        param = 100.0;

        ArrayList<JAROW> trainedAttrClassifiers = new ArrayList<>();
        ArrayList<HashMap<String, JAROW>> trainedWordClassifiers = new ArrayList<>();
        //INITIALIZE A POLICY P_0 (initializing on ref)
        JAROW trainedAttrClassifiers_0 = null;
        HashMap<String, JAROW> trainedWordClassifiers_0 = new HashMap<>();

        ArrayList<Instance> totalTrainingAttrInstances = new ArrayList<Instance>();
        HashMap<String, ArrayList<Instance>> totalTrainingWordInstances = new HashMap<String, ArrayList<Instance>>();

        trainedAttrClassifiers_0 = trainClassifier(trainingAttrInstances, param, adapt);
        totalTrainingAttrInstances.addAll(trainingAttrInstances);
        trainedAttrClassifiers.add(trainedAttrClassifiers_0);
        for (String attribute : attributes) {
            if (trainingWordInstances.containsKey(attribute) && !trainingWordInstances.get(attribute).isEmpty()) {
                trainedWordClassifiers_0.put(attribute, trainClassifier(trainingWordInstances.get(attribute), param, adapt));

                if (!totalTrainingWordInstances.containsKey(attribute)) {
                    totalTrainingWordInstances.put(attribute, new ArrayList<Instance>());
                }
                totalTrainingWordInstances.get(attribute).addAll(trainingWordInstances.get(attribute));

            } else {
                System.out.println("EMPTY " + attribute);
            }
        }
        trainedWordClassifiers.add(trainedWordClassifiers_0);        
        bagel.evaluateGeneration(trainedAttrClassifiers_0, trainedWordClassifiers_0, trainingData, testingData, availableWordActions, nGrams, predicate, true, -1);
        
        System.out.println("**************LOLS COMMENCING**************");
        int epochs = 3;
        for (int e = 0; e < epochs; e++) {
            if (e == 0) {
                beta = 1.0;
            } else {
                beta = Math.pow(1.0 - p, e);
            }

            System.out.println("beta = " + beta + " , p = " + p + " , early = " + earlyStopMaxFurtherSteps);

            JAROW trainedAttrClassifier_i = trainedAttrClassifiers.get(trainedAttrClassifiers.size() - 1);
            HashMap<String, JAROW> trainedWordClassifiers_i = trainedWordClassifiers.get(trainedWordClassifiers.size() - 1);

            ConcurrentHashMap<DatasetInstance, CopyOnWriteArrayList<Instance>> newAttrTrainingInstances = new ConcurrentHashMap<>();
            ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>> newWordTrainingInstances = new ConcurrentHashMap<>();
            for (DatasetInstance di : trainingData) {
                newAttrTrainingInstances.put(di, new CopyOnWriteArrayList<Instance>());
                newWordTrainingInstances.put(di, new ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>());
                for (String attr : attributes) {
                    newWordTrainingInstances.get(di).put(attr, new CopyOnWriteArrayList<Instance>());
                }
            }

            ExecutorService executor = Executors.newFixedThreadPool(threadsCount);
            for (DatasetInstance di : trainingData) {
                executor.execute(new runBAGELLOLSOnInstance(bagel, beta, di, attributes, trainedAttrClassifier_i, trainedWordClassifiers_i, valueAlignments, availableWordActions, trainingData, nGrams, newAttrTrainingInstances, newWordTrainingInstances));
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }

            ArrayList<Instance> totalNewAttrTrainingInstances = new ArrayList<Instance>();
            HashMap<String, ArrayList<Instance>> totalNewWordTrainingInstances = new HashMap<>();
            for (DatasetInstance di : trainingData) {
                totalNewAttrTrainingInstances.addAll(newAttrTrainingInstances.get(di));
                for (String attr : attributes) {
                    if (!totalNewWordTrainingInstances.containsKey(attr)) {
                        totalNewWordTrainingInstances.put(attr, new ArrayList<Instance>());
                    }
                    totalNewWordTrainingInstances.get(attr).addAll(newWordTrainingInstances.get(di).get(attr));
                }
            }

            //UPDATE CLASSIFIERS
            JAROW trainedAttrClassifier_ii = new JAROW(trainedAttrClassifier_i);
            trainedAttrClassifier_ii.trainAdditional(new ArrayList<Instance>(totalNewAttrTrainingInstances), true, false, 10, adapt, 1000);
            totalTrainingAttrInstances.addAll(totalNewAttrTrainingInstances);
            trainedAttrClassifiers.add(trainedAttrClassifier_ii);

            HashMap<String, JAROW> trainedWordClassifiers_ii = new HashMap<String, JAROW>();
            for (String attr : trainedWordClassifiers_i.keySet()) {
                trainedWordClassifiers_ii.put(attr, new JAROW(trainedWordClassifiers_i.get(attr)));
            }
            for (String attr : trainedWordClassifiers_i.keySet()) {
                if (!totalNewWordTrainingInstances.get(attr).isEmpty()) {
                    trainedWordClassifiers_ii.get(attr).trainAdditional(totalNewWordTrainingInstances.get(attr), true, false, 10, adapt, 1000);
                    totalTrainingWordInstances.get(attr).addAll(totalNewWordTrainingInstances.get(attr));
                }
            }
            trainedWordClassifiers.add(trainedWordClassifiers_ii);

            //FIRST NEED TO AVERAGE OVER ALL CLASSIFIERS
            HashMap<String, ArrayList<JAROW>> reorganizedClassifiersWords = new HashMap<>();
            for (String attribute : attributes) {
                reorganizedClassifiersWords.put(attribute, new ArrayList<JAROW>());
            }
            for (HashMap<String, JAROW> trainedClassifiers_i : trainedWordClassifiers) {
                for (String attribute : trainedClassifiers_i.keySet()) {
                    reorganizedClassifiersWords.get(attribute).add(trainedClassifiers_i.get(attribute));
                }
            }

            JAROW avgClassifiersAttrs = new JAROW();

            avgClassifiersAttrs.averageOverClassifiers(trainedAttrClassifiers);

            HashMap<String, JAROW> avgClassifiersWords = new HashMap<>();
            for (String attribute : attributes) {
                JAROW avg = new JAROW();
                avg.averageOverClassifiers(reorganizedClassifiersWords.get(attribute));

                if (!reorganizedClassifiersWords.get(attribute).isEmpty()) {
                    avgClassifiersWords.put(attribute, avg);
                }
            }

            System.out.println("AVERAGE CLASSIFIER at epoch = " + e);
            bagel.evaluateGeneration(avgClassifiersAttrs, avgClassifiersWords, trainingData, testingData, availableWordActions, nGrams, predicate, true, e);
        }

        //FIRST NEED TO AVERAGE OVER ALL CLASSIFIERS
        HashMap<String, ArrayList<JAROW>> reorganizedClassifiersWords = new HashMap<>();
        for (String attribute : attributes) {
            reorganizedClassifiersWords.put(attribute, new ArrayList<JAROW>());
        }
        for (HashMap<String, JAROW> trainedClassifiers_i : trainedWordClassifiers) {
            for (String attribute : trainedClassifiers_i.keySet()) {
                reorganizedClassifiersWords.get(attribute).add(trainedClassifiers_i.get(attribute));
            }
        }

        JAROW avgClassifiersAttrs = new JAROW();

        avgClassifiersAttrs.averageOverClassifiers(trainedAttrClassifiers);

        HashMap<String, JAROW> avgClassifiersWords = new HashMap<>();
        for (String attribute : attributes) {
            JAROW avg = new JAROW();
            avg.averageOverClassifiers(reorganizedClassifiersWords.get(attribute));

            if (!reorganizedClassifiersWords.get(attribute).isEmpty()) {
                avgClassifiersWords.put(attribute, avg);
            }
        }
        System.out.println("AVERAGE CLASSIFIER");
        bagel.evaluateGeneration(avgClassifiersAttrs, avgClassifiersWords, trainingData, testingData, availableWordActions, nGrams, predicate, true, 99);

        JAROW trainedAttrClassifiers_retrain = trainClassifier(totalTrainingAttrInstances, adapt);
        HashMap<String, JAROW> trainedWordClassifiers_retrain = new HashMap<>();
        for (String attribute : attributes) {
            if (trainingWordInstances.containsKey(attribute) && !trainingWordInstances.get(attribute).isEmpty()) {
                trainedWordClassifiers_retrain.put(attribute, trainClassifier(totalTrainingWordInstances.get(attribute), adapt));
            }
        }
        System.out.println("TOTAL CLASSIFIER");
        bagel.evaluateGeneration(trainedAttrClassifiers_retrain, trainedWordClassifiers_retrain, trainingData, testingData, availableWordActions, nGrams, predicate, true, 150);

        JAROW trainedAttrClassifiers_retrain2 = trainClassifier(totalTrainingAttrInstances, avgClassifiersAttrs.getParam(), adapt);
        HashMap<String, JAROW> trainedWordClassifiers_retrain2 = new HashMap<>();
        for (String attribute : attributes) {
            if (trainingWordInstances.containsKey(attribute) && !trainingWordInstances.get(attribute).isEmpty()) {
                trainedWordClassifiers_retrain2.put(attribute, trainClassifier(totalTrainingWordInstances.get(attribute), avgClassifiersWords.get(attribute).getParam(), adapt));
            }
        }
        System.out.println("TOTAL (NON OPT) CLASSIFIER");
        bagel.evaluateGeneration(trainedAttrClassifiers_retrain2, trainedWordClassifiers_retrain2, trainingData, testingData, availableWordActions, nGrams, predicate, true, 200);

        Object[] results = new Object[2];
        results[0] = avgClassifiersAttrs;
        results[1] = avgClassifiersWords;
        return results;
    }

    public JAROW trainClassifier(ArrayList<Instance> trainingWordInstances, boolean adapt) {
        Double[] params = {0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
        JAROW classifierWords = JAROW.trainOpt(trainingWordInstances, bagel.rounds, params, 0.1, adapt, false);
        return classifierWords;
    }

    public JAROW trainClassifier(ArrayList<Instance> trainingWordInstances, Double param, boolean adapt) {
        JAROW classifierWords = new JAROW();
        if (param == null) {
            classifierWords.train(trainingWordInstances, true, false, 10, JDAggerForBagel.param, adapt);
        } else {
            classifierWords.train(trainingWordInstances, true, false, 10, param, adapt);
        }

        return classifierWords;
    }

    public void multiTrainClassifier(ArrayList<Instance> trainingWordInstances, boolean adapt) {
        System.out.println("Start Multi");

        for (int i = 0; i < 10; i++) {
            JAROW classifierWords = new JAROW();
            Collections.shuffle(trainingWordInstances);
            System.out.println(classifierWords.train(trainingWordInstances, true, true, bagel.rounds, 0.1, adapt));
        }

        System.out.println("End Multi");

    }
}

class runBAGELLOLSOnInstance extends Thread {

    Bagel bagel;
    String predicate;
    double beta;
    DatasetInstance di;
    HashSet<String> attributes;
    JAROW trainedAttrClassifier_i;
    HashMap<String, JAROW> trainedWordClassifiers_i;
    HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments;
    HashMap<String, HashSet<Action>> availableWordActions;
    ArrayList<DatasetInstance> trainingData;
    HashMap<Integer, HashSet<String>> nGrams;
    ConcurrentHashMap<DatasetInstance, CopyOnWriteArrayList<Instance>> newAttrTrainingInstances;
    ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>> newWordTrainingInstances;

    public runBAGELLOLSOnInstance(Bagel bagel, double beta, DatasetInstance di, HashSet<String> attributes, JAROW trainedAttrClassifier_i, HashMap<String, JAROW> trainedWordClassifiers_i, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashSet<Action>> availableWordActions, ArrayList<DatasetInstance> trainingData, HashMap<Integer, HashSet<String>> nGrams, ConcurrentHashMap<DatasetInstance, CopyOnWriteArrayList<Instance>> newAttrTrainingInstances, ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>> newWordTrainingInstances) {
        this.predicate = di.getMeaningRepresentation().getPredicate();
        this.bagel = bagel;
        this.beta = beta;

        this.di = di;

        this.attributes = attributes;
        this.trainedAttrClassifier_i = trainedAttrClassifier_i;
        this.trainedWordClassifiers_i = trainedWordClassifiers_i;
        this.valueAlignments = valueAlignments;

        this.availableWordActions = availableWordActions;
        this.trainingData = trainingData;
        this.nGrams = nGrams;

        this.newAttrTrainingInstances = newAttrTrainingInstances;
        this.newWordTrainingInstances = newWordTrainingInstances;
    }

    public void run() {
        //ROLL-IN
        ActionSequence actSeq = getLearnedPolicyRollIn(predicate, di, trainedAttrClassifier_i, trainedWordClassifiers_i, valueAlignments, availableWordActions, trainingData, nGrams);

        boolean earlyStop = false;
        int earlyStopSteps = 0;
        //FOR EACH ACTION IN ROLL-IN SEQUENCE
        HashSet<String> encounteredXValues = new HashSet<String>();
        for (int index = 0; index < actSeq.getSequence().size(); index++) {
            //FOR EACH POSSIBLE ALTERNATIVE ACTION

            //Make the same decisions for all action substitutions
            boolean useReferenceRollout = false;
            double v = Bagel.r.nextDouble();
            if (v < beta) {
                useReferenceRollout = true;
            }
            calculateSubReferences(di, new ArrayList<Action>(actSeq.getSequence().subList(0, index)));

            String correctRealizations = "";
            for (ArrayList<Action> acList : di.getEvalRealizations()) {
                String l = "";
                for (Action a : acList) {
                    if (!a.getWord().equals(Bagel.TOKEN_START)
                            && !a.getWord().equals(Bagel.TOKEN_END)) {
                        l += a.getWord() + " ";
                    }
                }
                correctRealizations += l.trim() + " , ";
            }

            if (!actSeq.getSequence().get(index).getWord().equals(Bagel.TOKEN_START)
                    && !actSeq.getSequence().get(index).getAttribute().equals(Bagel.TOKEN_END)) {
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                String attrValue = actSeq.getSequence().get(index).getAttribute();
                String attr = attrValue.substring(0, attrValue.indexOf('='));
                int modLength = 0;
                HashSet<String> wrongActions = new HashSet<>();
                if (trainedWordClassifiers_i.containsKey(attr)) {
                    for (Action action : availableWordActions.get(attr)) {
                        costs.put(action.getWord().toLowerCase().trim(), 1.0);
                    }
                    costs.put(Bagel.TOKEN_END.toLowerCase().trim(), 1.0);

                    for (Action availableAction : availableWordActions.get(attr)) {
                        availableAction.setAttribute(attrValue);

                        boolean eligibleWord = true;
                        if (availableAction.getWord().trim().toLowerCase().startsWith(Bagel.TOKEN_X)) {
                            if (encounteredXValues.contains(availableAction.getWord().trim().toLowerCase())) {
                                eligibleWord = false;
                            } else {
                                int xIndex = Integer.parseInt(availableAction.getWord().trim().toLowerCase().substring(availableAction.getWord().trim().toLowerCase().indexOf("_") + 1));
                                for (int x = 0; x < xIndex; x++) {
                                    if (!encounteredXValues.contains(Bagel.TOKEN_X + attr + "_" + x)) {
                                        eligibleWord = false;
                                    }
                                }
                            }
                        }
                        if (availableAction.getWord().equals(Bagel.TOKEN_END)
                                && actSeq.getSequence().get(index - 1).getWord().equals(Bagel.TOKEN_START)) {
                            eligibleWord = false;
                        } else if (!availableAction.getWord().equals(Bagel.TOKEN_END)) {
                            ArrayList<Action> cleanActSeq = new ArrayList<Action>();
                            for (Action a : actSeq.getSequence().subList(0, index)) {
                                if (!a.getWord().equals(Bagel.TOKEN_START)
                                        && !a.getWord().equals(Bagel.TOKEN_END)) {
                                    cleanActSeq.add(a);
                                }
                            }
                            cleanActSeq.add(availableAction);
                            for (int j = 1; j <= Math.floor(cleanActSeq.size() / 2); j++) {
                                String followingStr = " " + (new ActionSequence(new ArrayList<Action>(cleanActSeq.subList(cleanActSeq.size() - j, cleanActSeq.size())), 0.0)).getWordSequenceToNoPunctString().trim();
                                String previousStr = " " + (new ActionSequence(new ArrayList<Action>(cleanActSeq.subList(0, cleanActSeq.size() - j)), 0.0)).getWordSequenceToNoPunctString().trim();

                                if (previousStr.endsWith(followingStr)) {
                                    eligibleWord = false;
                                }
                            }
                        }
                        if (eligibleWord) {
                            ActionSequence modSeq = new ActionSequence(actSeq);
                            modSeq.modifyAndShortenSequence(index, new Action(availableAction.getWord(), availableAction.getAttribute()));
                            modLength = modSeq.getNoEndLength();
                            //ROLL-OUT
                            costs.put(availableAction.getWord().trim().toLowerCase(), getPolicyRollOutCost(predicate, modSeq, di, trainedAttrClassifier_i, trainedWordClassifiers_i, valueAlignments, useReferenceRollout, trainingData, availableWordActions, nGrams));
                        } else {
                            wrongActions.add(availableAction.getWord().trim().toLowerCase());
                        }
                    }
                    if (!useReferenceRollout) {
                        double maxCost = 0.0;
                        for (String s : costs.keySet()) {
                            if (costs.get(s) != 100000000000000000.0
                                    && costs.get(s) > maxCost) {
                                maxCost = costs.get(s);
                            }
                        }
                        if (maxCost != 0.0) {
                            for (String s : costs.keySet()) {
                                if (costs.get(s) == 100000000000000000.0
                                        || wrongActions.contains(s)) {
                                    costs.put(s, 1.0);
                                } else {
                                    costs.put(s, costs.get(s) / maxCost);
                                }
                            }
                        }
                    }
                    Double bestActionCost = Double.POSITIVE_INFINITY;
                    for (String s : costs.keySet()) {
                        if (costs.get(s) < bestActionCost) {
                            bestActionCost = costs.get(s);
                        }
                    }
                    for (String s : costs.keySet()) {
                        if (costs.get(s) != 1.0) {
                            costs.put(s, costs.get(s) - bestActionCost);
                        }
                    }
                    Instance in = generateWordTrainingInstance(predicate, attrValue, di, new ArrayList<Action>(actSeq.getSequence().subList(0, index)), valueAlignments, costs, availableWordActions, nGrams);

                    in.setAlignedSubRealizations(di.getAlignedSubRealizations());

                    newWordTrainingInstances.get(di).get(actSeq.getSequence().get(index).getAttribute().substring(0, actSeq.getSequence().get(index).getAttribute().indexOf('='))).add(in);
                    if (trainedWordClassifiers_i.get(actSeq.getSequence().get(index).getAttribute().substring(0, actSeq.getSequence().get(index).getAttribute().indexOf('='))).isInstanceLeadingToFix(in)) {
                        earlyStop = true;
                        index = actSeq.getSequence().size() + 1;
                    }
                    if (earlyStop) {
                        if (earlyStopSteps >= JDAggerForBagel.earlyStopMaxFurtherSteps) {
                            index = actSeq.getSequence().size() + 1;
                        } else {
                            earlyStopSteps++;
                        }
                    }
                }
                if (index < actSeq.getSequence().size()) {
                    if (actSeq.getSequence().get(index).getWord().startsWith(Bagel.TOKEN_X)) {
                        encounteredXValues.add(actSeq.getSequence().get(index).getWord());
                    }
                }
            } else {
                //ALTERNATIVE ATTRS
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                for (String a : attributes) {
                    costs.put(a.toLowerCase().trim(), 1.0);
                }
                costs.put(Bagel.TOKEN_END.toLowerCase().trim(), 1.0);

                HashSet<String> wrongActions = new HashSet<>();
                for (String attr : costs.keySet()) {
                    String value = bagel.chooseNextValue(attr, actSeq.getSequence().get(index).getAttrValuesStillToBeMentionedAtThisPoint(), trainingData);

                    boolean eligibleWord = true;
                    if (value.isEmpty()
                            && !attr.equals(Bagel.TOKEN_END)) {
                        eligibleWord = false;
                    } else if (!di.getMeaningRepresentation().getAttributes().containsKey(attr)
                            && !attr.equals(Bagel.TOKEN_END)) {
                        eligibleWord = false;
                    } else if (!attr.equals(Bagel.TOKEN_END)) {
                        ArrayList<String> cleanAttrValuesSeq = new ArrayList<String>();
                        String previousAttrValue = "";
                        for (Action a : actSeq.getSequence().subList(0, index)) {
                            if (a.getWord().equals(Bagel.TOKEN_START)) {
                                if (previousAttrValue.isEmpty()) {
                                    cleanAttrValuesSeq.add(a.getAttribute());
                                } else if (!a.getAttribute().equals(previousAttrValue)) {
                                    cleanAttrValuesSeq.add(a.getAttribute());
                                }
                            }
                        }
                        cleanAttrValuesSeq.add(attr + "=" + value);
                        for (int j = 1; j <= Math.floor(cleanAttrValuesSeq.size() / 2); j++) {
                            String followingStr = "";
                            for (int i = cleanAttrValuesSeq.size() - j; i < cleanAttrValuesSeq.size(); i++) {
                                followingStr += " " + cleanAttrValuesSeq.get(i).trim();
                            }
                            String previousStr = "";
                            for (int i = 0; i < cleanAttrValuesSeq.size() - j; i++) {
                                previousStr += " " + cleanAttrValuesSeq.get(i).trim();
                            }
                            if (previousStr.endsWith(followingStr)) {
                                eligibleWord = false;
                            }
                        }
                    }
                    if (eligibleWord) {
                        ActionSequence modSeq = new ActionSequence(actSeq);
                        modSeq.modifyAndShortenSequence(index, new Action(Bagel.TOKEN_START, attr + "=" + value));
                        //ROLL-OUT
                        costs.put(attr.trim().toLowerCase(), getPolicyRollOutCost(predicate, modSeq, di, trainedAttrClassifier_i, trainedWordClassifiers_i, valueAlignments, useReferenceRollout, trainingData, availableWordActions, nGrams));
                    } else {
                        wrongActions.add(attr.trim().toLowerCase());
                    }
                }
                if (!useReferenceRollout) {
                    double maxCost = 0.0;
                    for (String s : costs.keySet()) {
                        if (costs.get(s) != 100000000000000000.0
                                && costs.get(s) > maxCost) {
                            maxCost = costs.get(s);
                        }
                    }
                    if (maxCost != 0.0) {
                        for (String s : costs.keySet()) {
                            if (costs.get(s) == 100000000000000000.0
                                    || wrongActions.contains(s)) {
                                costs.put(s, 1.0);
                            } else {
                                costs.put(s, costs.get(s) / maxCost);
                            }
                        }
                    }
                }
                Double bestActionCost = Double.POSITIVE_INFINITY;
                for (String s : costs.keySet()) {
                    if (costs.get(s) < bestActionCost) {
                        bestActionCost = costs.get(s);
                    }
                }
                if (bestActionCost == 1.0) {
                    costs.put(Bagel.TOKEN_END, 0.0);
                    bestActionCost = 0.0;
                }
                for (String s : costs.keySet()) {
                    if (costs.get(s) != 1.0) {
                        costs.put(s, costs.get(s) - bestActionCost);
                    }
                }
                Instance in = generateAttrTrainingInstance(predicate, di, new ArrayList<Action>(actSeq.getSequence().subList(0, index)), valueAlignments, costs);
                newAttrTrainingInstances.get(di).add(in);
                if (trainedAttrClassifier_i.isInstanceLeadingToFix(in)) {
                    index = actSeq.getSequence().size() + 1;
                    earlyStop = true;
                }
                if (earlyStop) {
                    if (earlyStopSteps >= JDAggerForBagel.earlyStopMaxFurtherSteps) {
                        index = actSeq.getSequence().size() + 1;
                    } else {
                        earlyStopSteps++;
                    }
                }
            }
        }
    }

    public void calculateSubReferences(DatasetInstance di, ArrayList<Action> actSeq) {
        ArrayList<Action> cleanActListPrev5Gram = new ArrayList<Action>();
        for (int i = actSeq.size() - 1; (i >= 0 && cleanActListPrev5Gram.size() < JDAggerForBagel.rollOutWindowSize); i--) {
            if (!actSeq.get(i).getWord().equals(Bagel.TOKEN_START)
                    && !actSeq.get(i).getWord().equals(Bagel.TOKEN_END)) {
                cleanActListPrev5Gram.add(0, actSeq.get(i));
            }
        }
        String actLastAttr = "";
        int lengthAfterAction = JDAggerForBagel.rollOutWindowSize + 1;
        boolean isForAttr = false;
        if (!actSeq.isEmpty()) {
            actLastAttr = actSeq.get(actSeq.size() - 1).getAttribute();
            if (actSeq.get(actSeq.size() - 1).getWord().equals(Bagel.TOKEN_END)) {
                isForAttr = true;
            }
        } else {
            isForAttr = true;
        }
        ActionSequence cleanActSeqPrev5Gram = new ActionSequence(cleanActListPrev5Gram, 0.0, true);
        for (ArrayList<Action> ref : di.getEvalRealizations()) {
            ArrayList<Action> cleanRef = new ArrayList<Action>();
            for (Action a : ref) {
                if (!a.getWord().equals(Bagel.TOKEN_START)
                        && !a.getWord().equals(Bagel.TOKEN_END)) {
                    cleanRef.add(a);
                }
            }
            di.setAlignedSubRealization(ref, null);
            di.setRealizationCorrectAction(ref, null);
            if (cleanActSeqPrev5Gram.getSequence().size() < JDAggerForBagel.rollOutWindowSize) {
                int start = cleanActSeqPrev5Gram.getSequence().size() - JDAggerForBagel.rollOutWindowSize;
                if (start < 0) {
                    start = 0;
                }
                ArrayList<Action> subRefGram = null;
                if (cleanActSeqPrev5Gram.getSequence().size() + JDAggerForBagel.rollOutWindowSize + 1 < cleanRef.size()) {
                    subRefGram = new ArrayList<Action>(cleanRef.subList(start, cleanActSeqPrev5Gram.getSequence().size() + JDAggerForBagel.rollOutWindowSize + 1));
                    if (isForAttr && !subRefGram.isEmpty()) {
                        int changeIndex = -1;
                        String newAttr = "";
                        boolean afterAttr = false;
                        for (int i = 0; i < subRefGram.size(); i++) {
                            if (!afterAttr) {
                                if (subRefGram.get(i).getAttribute().equals(actLastAttr)) {
                                    afterAttr = true;
                                }
                            } else {
                                if (newAttr.isEmpty()) {
                                    if (!subRefGram.get(i).getAttribute().equals(actLastAttr)) {
                                        newAttr = subRefGram.get(i).getAttribute();
                                    }
                                } else {
                                    if (!subRefGram.get(i).getAttribute().equals(newAttr)) {
                                        changeIndex = i;
                                        i = subRefGram.size();
                                    }
                                }
                            }
                        }
                        if (changeIndex != -1) {
                            subRefGram = new ArrayList<Action>(subRefGram.subList(0, changeIndex));
                        }
                    }
                    di.setAlignedSubRealization(ref, new ActionSequence(new ArrayList<Action>(subRefGram), 0.0, true).getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}", "").replaceAll("  ", " ").trim());
                } else {
                    subRefGram = new ArrayList<Action>(cleanRef.subList(start, cleanRef.size()));
                    if (isForAttr && !subRefGram.isEmpty()) {
                        int changeIndex = -1;
                        String newAttr = "";
                        boolean afterAttr = false;
                        for (int i = 0; i < subRefGram.size(); i++) {
                            if (!afterAttr) {
                                if (subRefGram.get(i).getAttribute().equals(actLastAttr)) {
                                    afterAttr = true;
                                }
                            } else {
                                if (newAttr.isEmpty()) {
                                    if (!subRefGram.get(i).getAttribute().equals(actLastAttr)) {
                                        newAttr = subRefGram.get(i).getAttribute();
                                    }
                                } else {
                                    if (!subRefGram.get(i).getAttribute().equals(newAttr)) {
                                        changeIndex = i;
                                        i = subRefGram.size();
                                    }
                                }
                            }
                        }
                        if (changeIndex != -1) {
                            subRefGram = new ArrayList<Action>(subRefGram.subList(0, changeIndex));
                        }
                    }
                    di.setAlignedSubRealization(ref, new ActionSequence(subRefGram, 0.0, true).getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}", "").replaceAll("  ", " ").trim());
                }
                if (cleanActListPrev5Gram.size() < cleanRef.size()) {
                    Action lastAction = cleanRef.get(cleanActListPrev5Gram.size());
                    if (!lastAction.getAttribute().equals(actLastAttr)
                            && !actLastAttr.isEmpty()
                            && !actSeq.get(actSeq.size() - 1).getWord().equals(Bagel.TOKEN_END)) {
                        di.setRealizationCorrectAction(ref, Bagel.TOKEN_END);
                    } else {
                        di.setRealizationCorrectAction(ref, lastAction.getWord());
                    }
                } else {
                    di.setRealizationCorrectAction(ref, Bagel.TOKEN_END);
                }
            } else {
                Double minCost = Double.MAX_VALUE;
                int bestAlignment = -1;
                int win = JDAggerForBagel.rollOutWindowSize;
                while (bestAlignment == -1) {
                    for (int i = 0; i <= cleanRef.size() - win; i++) {
                        ArrayList<Action> subRefGram = new ArrayList<Action>(cleanRef.subList(i, i + win));
                        ActionSequence subRef5Gram = new ActionSequence(subRefGram, 0.0, true);
                        String subRef5GramStr = subRef5Gram.getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}", "").replaceAll("  ", " ").trim();

                        ArrayList<String> cleanRefs = new ArrayList<>();
                        cleanRefs.add(subRef5GramStr);
                        double refCost = ActionSequence.getBLEU(cleanActSeqPrev5Gram.getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}", "").replaceAll("  ", " ").trim(), cleanRefs);

                        if (refCost < minCost) {
                            minCost = refCost;
                            bestAlignment = i;
                        }
                    }
                    win--;
                    if (win == 0) {
                        bestAlignment = -2;
                    }
                }
                if (bestAlignment >= 0) {
                    int start = bestAlignment + win + 1 - JDAggerForBagel.rollOutWindowSize;
                    if (start < 0) {
                        start = 0;
                    }
                    ArrayList<Action> subRefGram = null;
                    if (start + JDAggerForBagel.rollOutWindowSize + lengthAfterAction < cleanRef.size()) {
                        subRefGram = new ArrayList<Action>(cleanRef.subList(start, start + JDAggerForBagel.rollOutWindowSize + lengthAfterAction));
                        if (isForAttr && !subRefGram.isEmpty()) {
                            int changeIndex = -1;
                            String newAttr = "";
                            boolean afterAttr = false;
                            for (int i = 0; i < subRefGram.size(); i++) {
                                if (!afterAttr) {
                                    if (subRefGram.get(i).getAttribute().equals(actLastAttr)) {
                                        afterAttr = true;
                                    }
                                } else {
                                    if (newAttr.isEmpty()) {
                                        if (!subRefGram.get(i).getAttribute().equals(actLastAttr)) {
                                            newAttr = subRefGram.get(i).getAttribute();
                                        }
                                    } else {
                                        if (!subRefGram.get(i).getAttribute().equals(newAttr)) {
                                            changeIndex = i;
                                            i = subRefGram.size();
                                        }
                                    }
                                }
                            }
                            if (changeIndex != -1) {
                                subRefGram = new ArrayList<Action>(subRefGram.subList(0, changeIndex));
                            }
                        }
                        di.setAlignedSubRealization(ref, new ActionSequence(subRefGram, 0.0, true).getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}", "").replaceAll("  ", " ").trim());
                    } else {
                        subRefGram = new ArrayList<Action>(cleanRef.subList(start, cleanRef.size()));
                        if (isForAttr && !subRefGram.isEmpty()) {
                            int changeIndex = -1;
                            String newAttr = "";
                            boolean afterAttr = false;
                            for (int i = 0; i < subRefGram.size(); i++) {
                                if (!afterAttr) {
                                    if (subRefGram.get(i).getAttribute().equals(actLastAttr)) {
                                        afterAttr = true;
                                    }
                                } else {
                                    if (newAttr.isEmpty()) {
                                        if (!subRefGram.get(i).getAttribute().equals(actLastAttr)) {
                                            newAttr = subRefGram.get(i).getAttribute();
                                        }
                                    } else {
                                        if (!subRefGram.get(i).getAttribute().equals(newAttr)) {
                                            changeIndex = i;
                                            i = subRefGram.size();
                                        }
                                    }
                                }
                            }
                            if (changeIndex != -1) {
                                subRefGram = new ArrayList<Action>(subRefGram.subList(0, changeIndex));
                            }
                        }
                        di.setAlignedSubRealization(ref, new ActionSequence(subRefGram, 0.0, true).getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}", "").replaceAll("  ", " ").trim());
                    }

                    if (bestAlignment + win + 1 < cleanRef.size()) {
                        Action lastAction = cleanRef.get(bestAlignment + win + 1);
                        if (!lastAction.getAttribute().equals(actLastAttr)
                                && !actLastAttr.isEmpty()
                                && !actSeq.get(actSeq.size() - 1).getWord().equals(Bagel.TOKEN_END)) {
                            di.setRealizationCorrectAction(ref, Bagel.TOKEN_END);
                        } else {
                            di.setRealizationCorrectAction(ref, lastAction.getWord());
                        }
                    } else {
                        di.setRealizationCorrectAction(ref, Bagel.TOKEN_END);
                    }
                }
            }
        }
        int maxSize = -1;
        for (ArrayList<Action> ref : di.getEvalRealizations()) {
            if (di.getAlignedSubRealization(ref).split(" ").length > maxSize) {
                maxSize = di.getAlignedSubRealization(ref).split(" ").length;
            }
        }
        for (ArrayList<Action> ref : di.getEvalRealizations()) {
            if (di.getAlignedSubRealization(ref).split(" ").length < maxSize) {
                di.setAlignedSubRealization(ref, null);
            }
        }
    }

    public Double getPolicyRollOutCost(String predicate, ActionSequence actSeq, DatasetInstance di, JAROW classifierAttrs, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, double p, ArrayList<DatasetInstance> trainingData, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams) {
        double v = Bagel.r.nextDouble();

        if (v <= p) {
            return getReferencePolicyRollOutCost(actSeq, di, availableWordActions);
        } else {
            return getLearnedPolicyRollOutCost(actSeq, predicate, di, classifierAttrs, classifierWords, valueAlignments, availableWordActions, trainingData, nGrams);
        }
    }

    public Double getPolicyRollOutCost(String predicate, ActionSequence actSeq, DatasetInstance di, JAROW classifierAttrs, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, boolean useReferenceRollout, ArrayList<DatasetInstance> trainingData, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams) {
        if (useReferenceRollout) {
            return getReferencePolicyRollOutCost(actSeq, di, availableWordActions);
        } else {
            return getLearnedPolicyRollOutCost(actSeq, predicate, di, classifierAttrs, classifierWords, valueAlignments, availableWordActions, trainingData, nGrams);
        }
    }

    public ActionSequence getReferencePolicyRollIn(ActionSequence ref) {
        return new ActionSequence(ref);
    }

    public Double getReferencePolicyRollOutCost(ActionSequence pAS, DatasetInstance di, HashMap<String, HashSet<Action>> availableWordActions) {
        HashMap<ActionSequence, ArrayList<Action>> refs = new HashMap<>();
        for (ArrayList<Action> ref : di.getEvalRealizations()) {
            refs.put(new ActionSequence(ref, 0.0, true), ref);
        }

        if (pAS.getNoPunctLength() > 1 && pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(pAS.getSequence().get(pAS.getSequence().size() - 2).getWord())) {
            //Do not repeat the same word twice in a row
            return 1.0;
        } else {
            double minCost = 1.0;
            for (ActionSequence minRAS : refs.keySet()) {
                boolean resolved = false;
                if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(Bagel.TOKEN_END)) {
                    if (pAS.getNoPunctLength() < minRAS.getSequence().size()) {
                        //If last action is the end of the attr and ref says we should continue with the same attr
                        //In other words, we should not have ended expressing that attr!
                        if (pAS.getSequence().get(pAS.getSequence().size() - 1).getAttribute().equals(minRAS.getSequence().get(pAS.getNoPunctLength()).getAttribute())) {
                            if (1.0 <= minCost) {
                                minCost = 1.0;
                                resolved = true;
                            }
                        }
                    } else {
                        minCost = 0.0;
                        resolved = true;
                    }
                }
                if (!resolved) {
                    if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(Bagel.TOKEN_START)) {
                        String currentAttr = pAS.getSequence().get(pAS.getSequence().size() - 1).getAttribute().substring(0, pAS.getSequence().get(pAS.getSequence().size() - 1).getAttribute().indexOf('='));

                        if (availableWordActions.containsKey(currentAttr)
                                || currentAttr.equals(Bagel.TOKEN_END)) {
                            HashSet<Action> dict = new HashSet<>();
                            ArrayList<Action> cleanMinRAS = new ArrayList<>();
                            for (Action a : minRAS.getSequence()) {
                                if (!a.getWord().equals(Bagel.TOKEN_START)
                                        && !a.getWord().equals(Bagel.TOKEN_END)
                                        && !a.getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                                    cleanMinRAS.add(a);
                                }
                            }
                            if (pAS.getNoPunctLength() < cleanMinRAS.size()) {
                                String followingAttr = cleanMinRAS.get(pAS.getNoPunctLength()).getAttribute();
                                for (int i = pAS.getNoPunctLength(); i < cleanMinRAS.size(); i++) {
                                    Action a = cleanMinRAS.get(i);
                                    if (a.getAttribute().equals(followingAttr)) {
                                        dict.add(a);
                                    }
                                }
                                boolean allIncluded = true;
                                if (availableWordActions.containsKey(currentAttr)
                                        && di.getMeaningRepresentation().getAttributes().containsKey(currentAttr)) {
                                    for (Action a : dict) {
                                        if (!availableWordActions.get(currentAttr).contains(a)) {
                                            allIncluded = false;
                                        }
                                    }
                                } else {
                                    allIncluded = false;
                                }
                                if (allIncluded) {
                                    minCost = 0.0;
                                } else {
                                    minCost = 1.0;
                                }
                            } else {
                                minCost = 1.0;
                            }
                        } else {
                            minCost = 1.0;
                        }
                    } else {
                        //Let;s assume for now that the only correct response is the particular sentence that corresponds to the meaning representation in the data
                        //Shortening both ref and generated sequences to a 5 + 5 + 1 word window centered around the same index
                        double refCost = 1.0;
                        if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(di.getRealizationCorrectAction(refs.get(minRAS)))) {
                            refCost = 0.0;
                        } else {
                            refCost = ((double) JDAggerForBagel.rollOutWindowSize) / (JDAggerForBagel.rollOutWindowSize + JDAggerForBagel.rollOutWindowSize + 1.0);
                        }
                        if (refCost < minCost) {
                            minCost = refCost;
                        }
                    }
                }
            }
            return minCost;
        }
    }

    public ActionSequence getLearnedPolicyRollIn(String predicate, DatasetInstance di, JAROW classifierAttrs, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashSet<Action>> availableWordActions, ArrayList<DatasetInstance> trainingData, HashMap<Integer, HashSet<String>> nGrams) {
        ArrayList<Action> predictedActionsList = new ArrayList<>();
        ArrayList<Action> predictedWordList = new ArrayList<>();

        String predictedAttr = "";
        ArrayList<String> predictedAttrValues = new ArrayList<>();
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
        while (!predictedAttr.equals(Bagel.TOKEN_END) && predictedAttrValues.size() < bagel.maxAttrRealizationSize) {
            if (!predictedAttr.isEmpty()) {
                attrValuesToBeMentioned.remove(predictedAttr);
            }
            Instance attrTrainingVector = bagel.createAttrInstance(predicate, "@TOK@", predictedAttrValues, predictedActionsList, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation());

            if (attrTrainingVector != null) {
                Prediction predictAttr = classifierAttrs.predict(attrTrainingVector);
                predictedAttr = predictAttr.getLabel().trim();
                String predictedValue = "";
                if (!predictedAttr.equals(Bagel.TOKEN_END)) {
                    predictedValue = bagel.chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);

                    HashSet<String> rejectedAttrs = new HashSet<String>();
                    while (predictedValue.isEmpty() && !predictedAttr.equals(Bagel.TOKEN_END)) {
                        rejectedAttrs.add(predictedAttr);

                        predictedAttr = Bagel.TOKEN_END;
                        double maxScore = -Double.MAX_VALUE;
                        for (String attr : predictAttr.getLabel2Score().keySet()) {
                            if (!rejectedAttrs.contains(attr)
                                    && (Double.compare(predictAttr.getLabel2Score().get(attr), maxScore) > 0)) {
                                maxScore = predictAttr.getLabel2Score().get(attr);
                                predictedAttr = attr;
                            }
                        }                                
                        if (!predictedAttr.equals(Bagel.TOKEN_END)) {
                            predictedValue = bagel.chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);
                        }
                    }
                    predictedAttr += "=" + predictedValue;
                }

                predictedAttrValues.add(predictedAttr);

                String attribute = predictedAttrValues.get(predictedAttrValues.size() - 1).split("=")[0];
                String attrValue = predictedAttrValues.get(predictedAttrValues.size() - 1);
                predictedAttributes.add(attrValue);

                if (!attribute.equals(Bagel.TOKEN_END)) {
                    predictedActionsList.add(new Action(Bagel.TOKEN_START, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                    predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
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
                        while (!predictedWord.equals(Bagel.TOKEN_END) && predictedWordList.size() < bagel.maxWordRealizationSize) {

                            ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                            for (int i = 0; i < predictedAttributes.size() - 1; i++) {
                                predictedAttributesForInstance.add(predictedAttributes.get(i));
                            }
                            if (!predictedAttributes.get(predictedAttributes.size() - 1).equals(attrValue)) {
                                predictedAttributesForInstance.add(predictedAttributes.get(predictedAttributes.size() - 1));
                            }
                            Instance wordTrainingVector = bagel.createWordInstance(predicate, new Action("@TOK@", predictedAttrValues.get(predictedAttrValues.size() - 1)), predictedAttributesForInstance, predictedActionsList, isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions, nGrams, false);
                           
                            if (wordTrainingVector != null) {
                                if (classifierWords.get(attribute) != null) {
                                    Prediction predictWord = classifierWords.get(attribute).predict(wordTrainingVector);

                                    if (predictWord.getLabel() != null) {
                                        predictedWord = predictWord.getLabel().trim();
                                    } else {
                                        predictedWord = Bagel.TOKEN_END;
                                    }
                                    predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                    predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                                    if (!predictedWord.equals(Bagel.TOKEN_START)
                                            && !predictedWord.equals(Bagel.TOKEN_END)) {
                                        subPhrase.add(predictedWord);
                                        predictedWordList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                    }
                                } else {
                                    predictedWord = Bagel.TOKEN_END;
                                    predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                    predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                                }
                            }
                            if (!isValueMentioned) {
                                if (!predictedWord.equals(Bagel.TOKEN_END)) {
                                    if (predictedWord.startsWith(Bagel.TOKEN_X)
                                            && (valueTBM.matches("\"[xX][0-9]+\"")
                                            || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Bagel.TOKEN_X))) {
                                        isValueMentioned = true;
                                    } else if (!predictedWord.startsWith(Bagel.TOKEN_X)
                                            && !(valueTBM.matches("\"[xX][0-9]+\"")
                                            || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Bagel.TOKEN_X))) {
                                        for (ArrayList<String> alignedStr : valueAlignments.get(valueTBM).keySet()) {
                                            if (bagel.endsWith(subPhrase, alignedStr)) {
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
                            if (!predictedWord.startsWith(Bagel.TOKEN_X)) {
                                for (String attrValueTBM : attrValuesToBeMentioned) {
                                    if (attrValueTBM.contains("=")) {
                                        String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                        if (!(value.matches("\"[xX][0-9]+\"")
                                                || value.matches("[xX][0-9]+")
                                                || value.startsWith(Bagel.TOKEN_X))) {
                                            for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                                if (bagel.endsWith(subPhrase, alignedStr)) {
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
                        if (predictedWordList.size() >= bagel.maxWordRealizationSize
                                && !predictedActionsList.get(predictedActionsList.size() - 1).getWord().equals(Bagel.TOKEN_END)) {
                            predictedWord = Bagel.TOKEN_END;
                            predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                            predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                        }
                    } else {
                        predictedActionsList.add(new Action(Bagel.TOKEN_END, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                        predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                    }
                } else {
                    predictedActionsList.add(new Action(Bagel.TOKEN_END, Bagel.TOKEN_END));
                    predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                }
            }
        }
        if (predictedAttrValues.size() >= bagel.maxAttrRealizationSize
                && !predictedActionsList.get(predictedActionsList.size() - 1).getAttribute().equals(Bagel.TOKEN_END)) {
            predictedActionsList.add(new Action(Bagel.TOKEN_END, Bagel.TOKEN_END));
            predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
        }
        HashSet<ActionSequence> refs = new HashSet<>();
        for (ArrayList<Action> ref : di.getEvalRealizations()) {
            refs.add(new ActionSequence(ref, 0.0, true));
        }

        String previous = "";
        boolean end = false;
        boolean open = false;
        for (int i = 0; i < predictedActionsList.size(); i++) {
            if (previous.equals(Bagel.TOKEN_START)
                    && predictedActionsList.get(i).getWord().equals(Bagel.TOKEN_START)) {
                end = true;
            }
            if (predictedActionsList.get(i).getWord().equals(Bagel.TOKEN_START)) {
                if (open) {
                    end = true;
                }
                open = true;
            }
            if (predictedActionsList.get(i).getWord().equals(Bagel.TOKEN_END)
                    && !predictedActionsList.get(i).getAttribute().equals(Bagel.TOKEN_END)) {
                if (!open) {
                    end = true;
                }
                open = false;
            }
            previous = predictedActionsList.get(i).getWord();
        }
        if (end || open) {
            System.out.println("==========ROLL IN END OR OPEN===========");
            System.exit(0);
        }
        return new ActionSequence(predictedActionsList, refs);
    }

    public double getLearnedPolicyRollOutCost(ActionSequence pAS, String predicate, DatasetInstance di, JAROW classifierAttrs, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashSet<Action>> availableWordActions, ArrayList<DatasetInstance> trainingData, HashMap<Integer, HashSet<String>> nGrams) {

        Action examinedAction = pAS.getSequence().get(pAS.getSequence().size() - 1);

        if (pAS.getNoPunctLength() > 1 && examinedAction.getWord().equals(pAS.getSequence().get(pAS.getSequence().size() - 2).getWord())) {
            //Do not repeat the same word twice in a row
            return 1.0;
        } else {
            ArrayList<Action> predictedActionsList = new ArrayList<>();
            ArrayList<Action> predictedWordList = new ArrayList<>();
            ArrayList<Action> newPredictedActionsList = new ArrayList<>();
            String predictedAttr = "";
            ArrayList<String> predictedAttrValues = new ArrayList<>();
            ArrayList<String> predictedAttributes = new ArrayList<>();

            HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
            HashSet<String> attrValuesToBeMentioned = new HashSet<>();
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
            int generatedWords = 0;
            while (!predictedAttr.equals(Bagel.TOKEN_END) && predictedAttrValues.size() < bagel.maxAttrRealizationSize && generatedWords < JDAggerForBagel.rollOutWindowSize) {
                predictedAttr = "";
                if (predictedActionsList.size() < pAS.getSequence().size()) {
                    if (!pAS.getSequence().get(predictedActionsList.size()).getWord().equals(Bagel.TOKEN_START)) {
                        System.out.println("NO TOKEN START AFTER TOKEN END");
                        System.out.println(pAS.getSequence().get(predictedActionsList.size()).getWord());
                        System.exit(0);
                    }
                    predictedAttr = pAS.getSequence().get(predictedActionsList.size()).getAttribute();
                    predictedAttrValues.add(predictedAttr);
                } else {
                    if (!predictedAttr.isEmpty()) {
                        attrValuesToBeMentioned.remove(predictedAttr);
                    }
                    Instance attrTrainingVector = bagel.createAttrInstance(predicate, "@TOK@", predictedAttrValues, predictedActionsList, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation());

                    if (attrTrainingVector != null) {
                        Prediction predictAttr = classifierAttrs.predict(attrTrainingVector);
                        predictedAttr = predictAttr.getLabel().trim();
                        String predictedValue = "";
                        if (!predictedAttr.equals(Bagel.TOKEN_END)) {
                            predictedValue = bagel.chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);

                            HashSet<String> rejectedAttrs = new HashSet<String>();
                            while (predictedValue.isEmpty() && !predictedAttr.equals(Bagel.TOKEN_END)) {
                                rejectedAttrs.add(predictedAttr);

                                predictedAttr = Bagel.TOKEN_END;
                                double maxScore = -Double.MAX_VALUE;
                                for (String attr : predictAttr.getLabel2Score().keySet()) {
                                    if (!rejectedAttrs.contains(attr)
                                            && (Double.compare(predictAttr.getLabel2Score().get(attr), maxScore) > 0)) {
                                        maxScore = predictAttr.getLabel2Score().get(attr);
                                        predictedAttr = attr;
                                    }
                                }                                
                                if (!predictedAttr.equals(Bagel.TOKEN_END)) {
                                    predictedValue = bagel.chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);
                                }
                            }
                            predictedAttr += "=" + predictedValue;
                        }
                        predictedAttrValues.add(predictedAttr);
                    }
                }
                String attribute = predictedAttrValues.get(predictedAttrValues.size() - 1).split("=")[0];
                String attrValue = predictedAttrValues.get(predictedAttrValues.size() - 1);
                predictedAttributes.add(attrValue);

                if (!attribute.equals(Bagel.TOKEN_END)) {
                    predictedActionsList.add(new Action(Bagel.TOKEN_START, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                    if (classifierWords.containsKey(attribute)) {
                        String predictedWord = "";
                        int wW = predictedActionsList.size();

                        boolean isValueMentioned = false;
                        String valueTBM = "";
                        if (attrValue.contains("=")) {
                            valueTBM = attrValue.substring(attrValue.indexOf('=') + 1);
                        }
                        if (valueTBM.isEmpty()) {
                            isValueMentioned = true;
                        }
                        ArrayList<String> subPhrase = new ArrayList<>();
                        while (!predictedWord.equals(Bagel.TOKEN_END) && predictedWordList.size() < bagel.maxWordRealizationSize && generatedWords < JDAggerForBagel.rollOutWindowSize) {
                            if (predictedActionsList.size() < pAS.getSequence().size()) {
                                predictedWord = pAS.getSequence().get(predictedActionsList.size()).getWord();
                                predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                if (!predictedWord.equals(Bagel.TOKEN_START)
                                        && !predictedWord.equals(Bagel.TOKEN_END)) {
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
                                Instance wordTrainingVector = bagel.createWordInstance(predicate, new Action("@TOK@", predictedAttrValues.get(predictedAttrValues.size() - 1)), predictedAttributesForInstance, predictedActionsList, isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions, nGrams, false);

                                if (wordTrainingVector != null) {
                                    if (classifierWords.get(attribute) != null) {
                                        Prediction predictWord = classifierWords.get(attribute).predict(wordTrainingVector);
                                        if (predictWord.getLabel() != null) {
                                            predictedWord = predictWord.getLabel().trim();
                                        } else {
                                            predictedWord = Bagel.TOKEN_END;
                                        }
                                        predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                        if (!predictedWord.equals(Bagel.TOKEN_START)
                                                && !predictedWord.equals(Bagel.TOKEN_END)) {
                                            generatedWords++;
                                            subPhrase.add(predictedWord);
                                            newPredictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                            predictedWordList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                        }
                                    } else {
                                        predictedWord = Bagel.TOKEN_END;
                                        predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                    }
                                }
                            }
                            if (!isValueMentioned) {
                                if (!predictedWord.equals(Bagel.TOKEN_END)) {
                                    if (predictedWord.startsWith(Bagel.TOKEN_X)
                                            && (valueTBM.matches("\"[xX][0-9]+\"")
                                            || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Bagel.TOKEN_X))) {
                                        isValueMentioned = true;
                                    } else if (!predictedWord.startsWith(Bagel.TOKEN_X)
                                            && !(valueTBM.matches("\"[xX][0-9]+\"")
                                            || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Bagel.TOKEN_X))) {
                                        for (ArrayList<String> alignedStr : valueAlignments.get(valueTBM).keySet()) {
                                            if (bagel.endsWith(subPhrase, alignedStr)) {
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
                            if (!predictedWord.startsWith(Bagel.TOKEN_X)) {
                                for (String attrValueTBM : attrValuesToBeMentioned) {
                                    if (attrValueTBM.contains("=")) {
                                        String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                        if (!(value.matches("\"[xX][0-9]+\"")
                                                || value.matches("[xX][0-9]+"))) {
                                            for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                                if (bagel.endsWith(subPhrase, alignedStr)) {
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
                            wW++;
                        }
                        if (predictedWordList.size() >= bagel.maxWordRealizationSize
                                && !predictedActionsList.get(predictedActionsList.size() - 1).getWord().equals(Bagel.TOKEN_END)) {
                            predictedWord = Bagel.TOKEN_END;
                            predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                            predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                        }
                    } else {
                        predictedActionsList.add(new Action(Bagel.TOKEN_END, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                    }
                } else {
                    predictedActionsList.add(new Action(Bagel.TOKEN_END, Bagel.TOKEN_END));
                }
            }  
            ArrayList<Action> newCleanPredictedActionsList = new ArrayList<>();
            for (Action a : newPredictedActionsList) {
                //Essentially skip punctuation
                if (!a.getAttribute().equals(Bagel.TOKEN_PUNCT)
                        && !a.getWord().equals(Bagel.TOKEN_START)
                        && !a.getWord().equals(Bagel.TOKEN_END)) {
                    newCleanPredictedActionsList.add(a);
                }
            }
            int added = 0;
            for (int i = pAS.getSequence().size() - 1; added <= JDAggerForBagel.rollOutWindowSize && i >= 0; i--) {
                //Essentially skip punctuation
                if (!pAS.getSequence().get(i).getAttribute().equals(Bagel.TOKEN_PUNCT)
                        && !pAS.getSequence().get(i).getWord().equals(Bagel.TOKEN_START)
                        && !pAS.getSequence().get(i).getWord().equals(Bagel.TOKEN_END)) {
                    newCleanPredictedActionsList.add(0, pAS.getSequence().get(i));
                    added++;
                }
            }

            double minCost = 1.0;
            //If the end of the ATTR
            if (examinedAction.getWord().equals(Bagel.TOKEN_END)) {
                if (pAS.getSequence().size() < newCleanPredictedActionsList.size()) {
                    //If last action is the end of the attr and rollout says we should continue with the same attr
                    //In other words, we should not have ended expressing that attr!
                    if (!newPredictedActionsList.isEmpty()
                            && examinedAction.getAttribute().equals(newPredictedActionsList.get(0).getAttribute())) {
                        return 100000000000000000.0;
                    }
                }
            }
            ActionSequence newAS = new ActionSequence(newCleanPredictedActionsList, 0.0);

            ArrayList<String> refWindows = new ArrayList<>();
            for (ArrayList<Action> ref : di.getEvalRealizations()) {
                if (di.getAlignedSubRealization(ref) != null) {
                    refWindows.add(di.getAlignedSubRealization(ref));
                }
            }
            double refCost = 1.0;
            if (!refWindows.isEmpty()) {
                refCost = ActionSequence.getBLEU(newAS.getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}", "").replaceAll("  ", " ").trim(), refWindows);
            }

            if (minCost != 0.0) {
                minCost = refCost;
            }
            return minCost;
        }
    }

    public Instance generateWordTrainingInstance(String predicate, String currectAttrValue, DatasetInstance di, ArrayList<Action> generatedActions, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, TObjectDoubleHashMap<String> costs, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams) {
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

        ArrayList<Action> predictedWordsList = new ArrayList<>();
        boolean isValueMentioned = false;
        String valueTBM = "";
        String previousAlignment = "";
        ArrayList<String> subPhrase = new ArrayList<>();
        for (int i = 0; i < generatedActions.size(); i++) {
            String attribute = generatedActions.get(i).getAttribute();
            if (!attribute.equals(previousAlignment)) {
                previousAlignment = attribute;
                subPhrase = new ArrayList<>();
                isValueMentioned = false;
                valueTBM = "";
                if (previousAlignment.contains("=")) {
                    valueTBM = previousAlignment.substring(previousAlignment.indexOf('=') + 1);
                }
                if (valueTBM.isEmpty()) {
                    isValueMentioned = true;
                }
            }
            if (!attribute.equals(Bagel.TOKEN_END)) {
                String predictedWord = generatedActions.get(i).getWord();
                if (!predictedWord.equals(Bagel.TOKEN_START)
                        && !predictedWord.equals(Bagel.TOKEN_END)) {
                    subPhrase.add(predictedWord);
                }

                predictedWordsList.add(new Action(predictedWord, attribute));
                if (!isValueMentioned) {
                    if (predictedWordsList.get(i).getWord().startsWith(Bagel.TOKEN_X)
                            && (valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Bagel.TOKEN_X))) {
                        isValueMentioned = true;
                    } else if (!predictedWordsList.get(i).getWord().startsWith(Bagel.TOKEN_X)
                            && !(valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Bagel.TOKEN_X))) {
                        for (ArrayList<String> alignedStr : valueAlignments.get(valueTBM).keySet()) {
                            if (bagel.endsWith(subPhrase, alignedStr)) {
                                isValueMentioned = true;
                                break;
                            }
                        }
                    }
                    if (isValueMentioned) {
                        attrValuesAlreadyMentioned.add(attribute);
                        attrValuesToBeMentioned.remove(attribute);
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
                                    if (bagel.endsWith(subPhrase, alignedStr)) {
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
        ArrayList<String> predictedAttributes = new ArrayList<>();
        for (int i = 0; i < generatedActions.size(); i++) {
            Action act = predictedWordsList.get(i);
            if (predictedAttributes.isEmpty()) {
                if (!act.getAttribute().equals(Bagel.TOKEN_END) && !act.getAttribute().equals(currectAttrValue)) {
                    predictedAttributes.add(act.getAttribute());
                }
            } else {
                if (!act.getAttribute().equals(Bagel.TOKEN_END)
                        && !act.getAttribute().equals(predictedAttributes.get(predictedAttributes.size() - 1))
                        && !act.getAttribute().equals(currectAttrValue)) {
                    predictedAttributes.add(act.getAttribute());
                }
            }
        }
        return bagel.createWordInstance(predicate, currectAttrValue, predictedAttributes, predictedWordsList, costs, isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions, nGrams);
    }

    public Instance generateAttrTrainingInstance(String predicate, DatasetInstance di, ArrayList<Action> generatedActions, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, TObjectDoubleHashMap<String> costs) {
        HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
        HashSet<String> attrValuesToBeMentioned = new HashSet<>();
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

        ArrayList<Action> predictedActionsList = new ArrayList<>();
        ArrayList<String> predictedAttrValues = new ArrayList<>();

        boolean isValueMentioned = false;
        String valueTBM = "";
        String previousAlignment = "";
        ArrayList<String> subPhrase = new ArrayList<>();
        for (int i = 0; i < generatedActions.size(); i++) {
            String attrValue = generatedActions.get(i).getAttribute();
            if (!generatedActions.get(i).getAttribute().equals(previousAlignment)) {
                if (predictedAttrValues.isEmpty()) {
                    predictedAttrValues.add(attrValue);
                } else if (!predictedAttrValues.get(predictedAttrValues.size() - 1).equals(attrValue)) {
                    predictedAttrValues.add(attrValue);
                }
                previousAlignment = attrValue;
                subPhrase = new ArrayList<>();
                isValueMentioned = false;
                valueTBM = "";
                if (previousAlignment.contains("=")) {
                    valueTBM = previousAlignment.substring(previousAlignment.indexOf('=') + 1);
                }
                if (valueTBM.isEmpty()) {
                    isValueMentioned = true;
                }
            }
            if (!attrValue.equals(Bagel.TOKEN_END)) {
                String predictedWord = generatedActions.get(i).getWord();
                if (!predictedWord.equals(Bagel.TOKEN_START)
                        && !predictedWord.equals(Bagel.TOKEN_END)) {
                    subPhrase.add(predictedWord);
                }

                predictedActionsList.add(new Action(predictedWord, attrValue));
                if (!isValueMentioned) {
                    if (predictedActionsList.get(i).getWord().startsWith(Bagel.TOKEN_X)
                            && (valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Bagel.TOKEN_X))) {
                        isValueMentioned = true;
                    } else if (!predictedActionsList.get(i).getWord().startsWith(Bagel.TOKEN_X)
                            && !(valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Bagel.TOKEN_X))) {
                        for (ArrayList<String> alignedStr : valueAlignments.get(valueTBM).keySet()) {
                            if (bagel.endsWith(subPhrase, alignedStr)) {
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
                if (!predictedWord.startsWith(Bagel.TOKEN_X)) {
                    for (String attrValueTBM : attrValuesToBeMentioned) {
                        if (attrValueTBM.contains("=")) {
                            String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                            if (!(value.matches("\"[xX][0-9]+\"")
                                    || value.matches("[xX][0-9]+"))) {
                                for (ArrayList<String> alignedStr : valueAlignments.get(value).keySet()) {
                                    if (bagel.endsWith(subPhrase, alignedStr)) {
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
        return bagel.createAttrInstance(predicate, predictedAttrValues, predictedActionsList, costs, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation());
    }
}
