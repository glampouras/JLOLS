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
import imitationNLG.Action;
import imitationNLG.ActionSequence;
import imitationNLG.DatasetInstance;
import imitationNLG.SFX;
import jarow.Instance;
import jarow.JAROW;
import jarow.Prediction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class JDAggerForSFX {

    final static int threadsCount = 4;

    public JDAggerForSFX(SFX SFX) {
        this.SFX = SFX;
    }
    boolean print = false;
    static boolean adapt = false;
    static double param = 0.0;
    static double additionalParam = 0.0;
    public static double earlyStopMaxFurtherSteps = 0;
    public static double p = 0.2;
    static int rollOutWindowSize = 5;
    public static int checkIndex = -1;
    SFX SFX = null;

    public Object[] runLOLS(HashMap<String, HashSet<String>> availableAttributeActions, ArrayList<DatasetInstance> trainingData, HashMap<String, ArrayList<Instance>> trainingAttrInstances, HashMap<String, HashMap<String, ArrayList<Instance>>> trainingWordInstances, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, double beta, ArrayList<DatasetInstance> testingData, HashMap<Integer, HashSet<String>> nGrams) {
        param = 1000.0;
        additionalParam = 1000.0;

        ArrayList<HashMap<String, JAROW>> trainedAttrClassifiers = new ArrayList<>();
        ArrayList<HashMap<String, HashMap<String, JAROW>>> trainedWordClassifiers = new ArrayList<>();
        //INITIALIZE A POLICY P_0 (initializing on ref)
        HashMap<String, JAROW> trainedAttrClassifiers_0 = new HashMap<>();
        HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0 = new HashMap<>();

        HashMap<String, ArrayList<Instance>> totalTrainingAttrInstances = new HashMap<>();
        HashMap<String, HashMap<String, ArrayList<Instance>>> totalTrainingWordInstances = new HashMap<>();

        for (String predicate : trainingAttrInstances.keySet()) {
            trainedAttrClassifiers_0.put(predicate, trainClassifier(trainingAttrInstances.get(predicate), param, adapt));
            //trainedAttrClassifiers_0.put(predicate, trainClassifier(trainingAttrInstances.get(predicate), adapt));
            if (!totalTrainingAttrInstances.containsKey(predicate)) {
                totalTrainingAttrInstances.put(predicate, new ArrayList<Instance>());
            }
            totalTrainingAttrInstances.get(predicate).addAll(trainingAttrInstances.get(predicate));
            for (String attribute : availableAttributeActions.get(predicate)) {
                if (trainingWordInstances.get(predicate).containsKey(attribute) && !trainingWordInstances.get(predicate).get(attribute).isEmpty()) {
                    if (!trainedWordClassifiers_0.containsKey(predicate)) {
                        trainedWordClassifiers_0.put(predicate, new HashMap<String, JAROW>());
                    }
                    trainedWordClassifiers_0.get(predicate).put(attribute, trainClassifier(trainingWordInstances.get(predicate).get(attribute), param, adapt));

                    if (!totalTrainingWordInstances.containsKey(predicate)) {
                        totalTrainingWordInstances.put(predicate, new HashMap<String, ArrayList<Instance>>());
                    }
                    if (!totalTrainingWordInstances.get(predicate).containsKey(attribute)) {
                        totalTrainingWordInstances.get(predicate).put(attribute, new ArrayList<Instance>());
                    }
                    totalTrainingWordInstances.get(predicate).get(attribute).addAll(trainingWordInstances.get(predicate).get(attribute));
                } else {
                    System.out.println("EMPTY " + attribute);
                }
            }
        }
        trainedAttrClassifiers.add(trainedAttrClassifiers_0);
        trainedWordClassifiers.add(trainedWordClassifiers_0);
        Double BLEU = SFX.evaluateGeneration(trainedAttrClassifiers_0, trainedWordClassifiers_0, trainingData, testingData, availableAttributeActions, availableWordActions, nGrams, true, -1);
        System.out.println("**************LOLS COMMENCING**************");

        checkIndex = -1;
        int epochs = 5;
        for (int e = 0; e < epochs; e++) {
            if (e == 0) {
                beta = 1.0;
            } else {
                beta = Math.pow(1.0 - p, e);
            }
            //beta = 1.0 - p;
            System.out.println("beta = " + beta + " , p = " + p + " , early = " + earlyStopMaxFurtherSteps);

            HashMap<String, JAROW> trainedAttrClassifier_i = trainedAttrClassifiers.get(trainedAttrClassifiers.size() - 1);
            HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_i = trainedWordClassifiers.get(trainedWordClassifiers.size() - 1);

            ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>> newAttrTrainingInstances = new ConcurrentHashMap<>();
            ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>>> newWordTrainingInstances = new ConcurrentHashMap<>();
            for (DatasetInstance di : trainingData) {
                newAttrTrainingInstances.put(di, new ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>());
                newWordTrainingInstances.put(di, new ConcurrentHashMap<String, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>>());

                for (String predicate : trainingAttrInstances.keySet()) {
                    newAttrTrainingInstances.get(di).put(predicate, new CopyOnWriteArrayList<Instance>());
                    newWordTrainingInstances.get(di).put(predicate, new ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>());
                    for (String attr : availableAttributeActions.get(predicate)) {
                        newWordTrainingInstances.get(di).get(predicate).put(attr, new CopyOnWriteArrayList<Instance>());
                    }
                }
            }

            ExecutorService executor = Executors.newFixedThreadPool(threadsCount);
            for (DatasetInstance di : trainingData) {
                executor.execute(new runSFXLOLSOnInstance(SFX, beta, di, availableAttributeActions, trainedAttrClassifier_i, trainedWordClassifiers_i, valueAlignments, availableWordActions, trainingData, nGrams, newAttrTrainingInstances, newWordTrainingInstances));
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }

            HashMap<String, ArrayList<Instance>> totalNewAttrTrainingInstances = new HashMap<String, ArrayList<Instance>>();
            HashMap<String, HashMap<String, ArrayList<Instance>>> totalNewWordTrainingInstances = new HashMap<>();
            for (DatasetInstance di : trainingData) {
                for (String predicate : newAttrTrainingInstances.get(di).keySet()) {
                    if (!totalNewAttrTrainingInstances.containsKey(predicate)) {
                        totalNewAttrTrainingInstances.put(predicate, new ArrayList<Instance>());
                    }
                    if (!totalNewWordTrainingInstances.containsKey(predicate)) {
                        totalNewWordTrainingInstances.put(predicate, new HashMap<String, ArrayList<Instance>>());
                    }
                    totalNewAttrTrainingInstances.get(predicate).addAll(newAttrTrainingInstances.get(di).get(predicate));
                }
                for (String predicate : totalNewWordTrainingInstances.keySet()) {
                    for (String attr : availableAttributeActions.get(predicate)) {
                        if (!totalNewWordTrainingInstances.get(predicate).containsKey(attr)) {
                            totalNewWordTrainingInstances.get(predicate).put(attr, new ArrayList<Instance>());
                        }
                        totalNewWordTrainingInstances.get(predicate).get(attr).addAll(newWordTrainingInstances.get(di).get(predicate).get(attr));
                    }
                }
            }

            //UPDATE CLASSIFIERS
            HashMap<String, JAROW> trainedAttrClassifier_ii = new HashMap<String, JAROW>();
            HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_ii = new HashMap<String, HashMap<String, JAROW>>();
            for (String predicate : trainingAttrInstances.keySet()) {
                trainedAttrClassifier_ii.put(predicate, new JAROW(trainedAttrClassifier_i.get(predicate)));

                if (trainedWordClassifiers_i.containsKey(predicate)) {
                    trainedWordClassifiers_ii.put(predicate, new HashMap<String, JAROW>());
                    for (String attr : trainedWordClassifiers_i.get(predicate).keySet()) {
                        trainedWordClassifiers_ii.get(predicate).put(attr, new JAROW(trainedWordClassifiers_i.get(predicate).get(attr)));
                    }
                }
            }
            for (String predicate : trainingAttrInstances.keySet()) {
                trainedAttrClassifier_ii.get(predicate).trainAdditional(totalNewAttrTrainingInstances.get(predicate), true, false, 10, adapt, additionalParam);
                totalTrainingAttrInstances.get(predicate).addAll(totalNewAttrTrainingInstances.get(predicate));

                if (trainedWordClassifiers_i.containsKey(predicate)) {
                    for (String attr : trainedWordClassifiers_i.get(predicate).keySet()) {
                        if (!totalNewWordTrainingInstances.get(predicate).get(attr).isEmpty()) {
                            trainedWordClassifiers_ii.get(predicate).get(attr).trainAdditional(totalNewWordTrainingInstances.get(predicate).get(attr), true, false, 10, adapt, additionalParam);
                            totalTrainingWordInstances.get(predicate).get(attr).addAll(totalNewWordTrainingInstances.get(predicate).get(attr));
                        }
                    }
                }
            }
            trainedAttrClassifiers.add(trainedAttrClassifier_ii);
            trainedWordClassifiers.add(trainedWordClassifiers_ii);

            //FIRST NEED TO AVERAGE OVER ALL CLASSIFIERS
            HashMap<String, ArrayList<JAROW>> reorganizedClassifiersAttrs = new HashMap<>();
            for (String predicate : trainingAttrInstances.keySet()) {
                reorganizedClassifiersAttrs.put(predicate, new ArrayList<JAROW>());
            }
            for (HashMap<String, JAROW> trainedClassifiers_i : trainedAttrClassifiers) {
                for (String predicate : trainedClassifiers_i.keySet()) {
                    reorganizedClassifiersAttrs.get(predicate).add(trainedClassifiers_i.get(predicate));
                }
            }

            HashMap<String, HashMap<String, ArrayList<JAROW>>> reorganizedClassifiersWords = new HashMap<>();
            for (String predicate : trainingAttrInstances.keySet()) {
                reorganizedClassifiersWords.put(predicate, new HashMap<String, ArrayList<JAROW>>());
                for (String attribute : availableAttributeActions.get(predicate)) {
                    reorganizedClassifiersWords.get(predicate).put(attribute, new ArrayList<JAROW>());
                }
            }
            for (HashMap<String, HashMap<String, JAROW>> trainedClassifiers_i : trainedWordClassifiers) {
                for (String predicate : trainedClassifiers_i.keySet()) {
                    for (String attribute : trainedClassifiers_i.get(predicate).keySet()) {
                        reorganizedClassifiersWords.get(predicate).get(attribute).add(trainedClassifiers_i.get(predicate).get(attribute));
                    }
                }
            }

            HashMap<String, JAROW> avgClassifiersAttrs = new HashMap<>();
            for (String predicate : trainingAttrInstances.keySet()) {
                JAROW avg = new JAROW();
                avg.averageOverClassifiers(reorganizedClassifiersAttrs.get(predicate));

                avgClassifiersAttrs.put(predicate, avg);
            }

            HashMap<String, HashMap<String, JAROW>> avgClassifiersWords = new HashMap<>();
            for (String predicate : trainingWordInstances.keySet()) {
                for (String attribute : availableAttributeActions.get(predicate)) {
                    if (!avgClassifiersWords.containsKey(predicate)) {
                        avgClassifiersWords.put(predicate, new HashMap<String, JAROW>());
                    }
                    if (!reorganizedClassifiersWords.get(predicate).get(attribute).isEmpty()) {
                        JAROW avg = new JAROW();
                        avg.averageOverClassifiers(reorganizedClassifiersWords.get(predicate).get(attribute));

                        avgClassifiersWords.get(predicate).put(attribute, avg);
                    }
                }
            }

            System.out.println("AVERAGE CLASSIFIER at epoch = " + e);
            SFX.evaluateGeneration(avgClassifiersAttrs, avgClassifiersWords, trainingData, testingData, availableAttributeActions, availableWordActions, nGrams, true, e + 1);
        }

        HashMap<String, ArrayList<JAROW>> reorganizedClassifiersAttrs = new HashMap<>();
        for (String predicate : trainingAttrInstances.keySet()) {
            reorganizedClassifiersAttrs.put(predicate, new ArrayList<JAROW>());
        }
        for (HashMap<String, JAROW> trainedClassifiers_i : trainedAttrClassifiers) {
            for (String predicate : trainedClassifiers_i.keySet()) {
                reorganizedClassifiersAttrs.get(predicate).add(trainedClassifiers_i.get(predicate));
            }
        }

        HashMap<String, HashMap<String, ArrayList<JAROW>>> reorganizedClassifiersWords = new HashMap<>();
        for (String predicate : trainingAttrInstances.keySet()) {
            reorganizedClassifiersWords.put(predicate, new HashMap<String, ArrayList<JAROW>>());
            for (String attribute : availableAttributeActions.get(predicate)) {
                reorganizedClassifiersWords.get(predicate).put(attribute, new ArrayList<JAROW>());
            }
        }
        for (HashMap<String, HashMap<String, JAROW>> trainedClassifiers_i : trainedWordClassifiers) {
            for (String predicate : trainedClassifiers_i.keySet()) {
                for (String attribute : trainedClassifiers_i.get(predicate).keySet()) {
                    reorganizedClassifiersWords.get(predicate).get(attribute).add(trainedClassifiers_i.get(predicate).get(attribute));
                }
            }
        }

        HashMap<String, JAROW> avgClassifiersAttrs = new HashMap<>();
        for (String predicate : trainingAttrInstances.keySet()) {
            JAROW avg = new JAROW();
            avg.averageOverClassifiers(reorganizedClassifiersAttrs.get(predicate));

            avgClassifiersAttrs.put(predicate, avg);
        }

        HashMap<String, HashMap<String, JAROW>> avgClassifiersWords = new HashMap<>();
        for (String predicate : trainingWordInstances.keySet()) {
            for (String attribute : availableAttributeActions.get(predicate)) {
                JAROW avg = new JAROW();
                avg.averageOverClassifiers(reorganizedClassifiersWords.get(predicate).get(attribute));

                if (!avgClassifiersWords.containsKey(predicate)) {
                    avgClassifiersWords.put(predicate, new HashMap<String, JAROW>());
                }
                if (!reorganizedClassifiersWords.get(predicate).get(attribute).isEmpty()) {
                    avgClassifiersWords.get(predicate).put(attribute, avg);
                }
            }
        }
        System.out.println("AVERAGE CLASSIFIER");
        SFX.evaluateGeneration(avgClassifiersAttrs, avgClassifiersWords, trainingData, testingData, availableAttributeActions, availableWordActions, nGrams, true, 150);

        HashMap<String, JAROW> trainedAttrClassifiers_retrain2 = new HashMap<>();
        HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_retrain2 = new HashMap<>();
        for (String predicate : trainingAttrInstances.keySet()) {
            trainedAttrClassifiers_retrain2.put(predicate, trainClassifier(totalTrainingAttrInstances.get(predicate), avgClassifiersAttrs.get(predicate).getParam(), adapt));
            trainedWordClassifiers_retrain2.put(predicate, new HashMap<String, JAROW>());
            for (String attribute : availableAttributeActions.get(predicate)) {
                if (trainingWordInstances.get(predicate).containsKey(attribute) && !trainingWordInstances.get(predicate).get(attribute).isEmpty()) {
                    trainedWordClassifiers_retrain2.get(predicate).put(attribute, trainClassifier(totalTrainingWordInstances.get(predicate).get(attribute), avgClassifiersWords.get(predicate).get(attribute).getParam(), adapt));
                }
            }
        }
        System.out.println("TOTAL (NON OPT) CLASSIFIER");
        SFX.evaluateGeneration(trainedAttrClassifiers_retrain2, trainedWordClassifiers_retrain2, trainingData, testingData, availableAttributeActions, availableWordActions, nGrams, true, 200);

        Object[] results = new Object[2];
        results[0] = avgClassifiersAttrs;
        results[1] = avgClassifiersWords;
        return results;
    }

    public JAROW trainClassifier(ArrayList<Instance> trainingWordInstances, boolean adapt) {
        Double[] params = {0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
        JAROW classifierWords = JAROW.trainOpt(trainingWordInstances, SFX.rounds, params, 0.1, adapt, false);
        return classifierWords;
    }

    public JAROW trainClassifier(ArrayList<Instance> trainingWordInstances, Double param, boolean adapt) {
        JAROW classifierWords = new JAROW();
        if (param == null) {
            classifierWords.train(trainingWordInstances, true, false, 10, param, adapt);
        } else {
            classifierWords.train(trainingWordInstances, true, false, 10, param, adapt);
        }

        return classifierWords;
    }
}

class runSFXLOLSOnInstance extends Thread {

    SFX SFX;
    double beta;
    DatasetInstance di;
    HashMap<String, HashSet<String>> availableAttributeActions;
    HashMap<String, JAROW> trainedAttrClassifier_i;
    HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_i;
    HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments;
    HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions;
    ArrayList<DatasetInstance> trainingData;
    HashMap<Integer, HashSet<String>> nGrams;
    ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>> newAttrTrainingInstances;
    ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>>> newWordTrainingInstances;

    public runSFXLOLSOnInstance(SFX SFX, double beta, DatasetInstance di, HashMap<String, HashSet<String>> availableAttributeActions, HashMap<String, JAROW> trainedAttrClassifier_i, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_i, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions, ArrayList<DatasetInstance> trainingData, HashMap<Integer, HashSet<String>> nGrams, ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>> newAttrTrainingInstances, ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>>> newWordTrainingInstances) {
        this.SFX = SFX;
        this.beta = beta;

        this.di = di;

        this.availableAttributeActions = availableAttributeActions;
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
        String predicate = di.getMeaningRepresentation().getPredicate();
        //ROLL-IN
        ActionSequence actSeq = getLearnedPolicyRollIn(predicate, di, trainedAttrClassifier_i.get(predicate), trainedWordClassifiers_i.get(predicate), valueAlignments, availableWordActions.get(predicate), trainingData, nGrams);

        boolean earlyStop = false;
        int earlyStopSteps = 0;
        //FOR EACH ACTION IN ROLL-IN SEQUENCE
        HashSet<String> encounteredXValues = new HashSet<String>();
        for (int index = 0; index < actSeq.getSequence().size(); index++) {
            //FOR EACH POSSIBLE ALTERNATIVE ACTION

            //Make the same decisions for all action substitutions
            boolean useReferenceRollout = false;
            double v = SFX.r.nextDouble();
            if (v < beta) {
                useReferenceRollout = true;
            }
            calculateSubReferences(di, new ArrayList<Action>(actSeq.getSequence().subList(0, index)));

            String correctRealizations = "";
            String l = "";
            for (Action a : di.getTrainRealization()) {
                if (!a.getWord().equals(SFX.TOKEN_START)
                        && !a.getWord().equals(SFX.TOKEN_END)) {
                    l += a.getWord() + " ";
                }
            }
            correctRealizations += l.trim() + " , ";

            if (!actSeq.getSequence().get(index).getWord().equals(SFX.TOKEN_START)
                    && !actSeq.getSequence().get(index).getAttribute().equals(SFX.TOKEN_END)) {
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                String attrValue = actSeq.getSequence().get(index).getAttribute();
                String attr = attrValue.substring(0, attrValue.indexOf('='));
                int modLength = 0;
                HashSet<String> wrongActions = new HashSet<>();
                if (trainedWordClassifiers_i.get(predicate).containsKey(attr)) {
                    for (Action action : availableWordActions.get(predicate).get(attr)) {
                        costs.put(action.getWord().toLowerCase().trim(), 1.0);
                    }
                    costs.put(SFX.TOKEN_END.toLowerCase().trim(), 1.0);

                    String value = SFX.chooseNextValue(attr, actSeq.getSequence().get(index).getAttrValuesStillToBeMentionedAtThisPoint(), trainingData);
                    for (Action availableAction : availableWordActions.get(predicate).get(attr)) {
                        availableAction.setAttribute(attrValue);

                        boolean eligibleWord = true;
                        if (availableAction.getWord().trim().toLowerCase().startsWith(SFX.TOKEN_X)) {
                            if (value.isEmpty()) {
                                eligibleWord = false;
                            } else if (value.equals("no")
                                    || value.equals("yes")
                                    || value.equals("yes or no")
                                    || value.equals("none")
                                    || value.equals("empty")
                                    || value.equals("dont_care")) {
                                eligibleWord = false;
                            } else {
                                if (encounteredXValues.contains(availableAction.getWord().trim().toLowerCase())) {
                                    eligibleWord = false;
                                } else {
                                    int xIndex = Integer.parseInt(availableAction.getWord().trim().toLowerCase().substring(availableAction.getWord().trim().toLowerCase().indexOf("_") + 1));
                                    for (int x = 0; x < xIndex; x++) {
                                        if (!encounteredXValues.contains(SFX.TOKEN_X + attr + "_" + x)) {
                                            eligibleWord = false;
                                        }
                                    }
                                }
                            }
                        }
                        if (availableAction.getWord().equals(SFX.TOKEN_END)
                                && (actSeq.getSequence().get(index - 1).getWord().equals(SFX.TOKEN_START)
                                || index == 0)) {
                            eligibleWord = false;
                        } else if (!availableAction.getWord().equals(SFX.TOKEN_END)) {
                            ArrayList<Action> cleanActSeq = new ArrayList<Action>();
                            for (Action a : actSeq.getSequence().subList(0, index)) {
                                if (!a.getWord().equals(SFX.TOKEN_START)
                                        && !a.getWord().equals(SFX.TOKEN_END)) {
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
                            costs.put(availableAction.getWord().trim().toLowerCase(), getPolicyRollOutCost(predicate, modSeq, di, trainedAttrClassifier_i.get(predicate), trainedWordClassifiers_i.get(predicate), valueAlignments, useReferenceRollout, trainingData, availableWordActions.get(predicate), nGrams));
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
                    Instance in = generateWordTrainingInstance(predicate, attrValue, value, di, new ArrayList<Action>(actSeq.getSequence().subList(0, index)), valueAlignments, costs, availableWordActions.get(predicate), nGrams);

                    in.setAlignedSubRealizations(di.getAlignedSubRealizations());

                    newWordTrainingInstances.get(di).get(predicate).get(actSeq.getSequence().get(index).getAttribute().substring(0, actSeq.getSequence().get(index).getAttribute().indexOf('='))).add(in);
                    if (trainedWordClassifiers_i.get(predicate).get(actSeq.getSequence().get(index).getAttribute().substring(0, actSeq.getSequence().get(index).getAttribute().indexOf('='))).isInstanceLeadingToFix(in)) {
                        earlyStop = true;
                        index = actSeq.getSequence().size() + 1;
                    }
                    if (earlyStop) {
                        if (earlyStopSteps >= JDAggerForSFX.earlyStopMaxFurtherSteps) {
                            index = actSeq.getSequence().size() + 1;
                        } else {
                            earlyStopSteps++;
                        }
                    }
                }
                if (index < actSeq.getSequence().size()) {
                    if (actSeq.getSequence().get(index).getWord().startsWith(SFX.TOKEN_X)) {
                        encounteredXValues.add(actSeq.getSequence().get(index).getWord());
                    }
                }
            } else {
                //ALTERNATIVE ATTRS
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                for (String a : availableAttributeActions.get(predicate)) {
                    costs.put(a.toLowerCase().trim(), 1.0);
                }
                costs.put(SFX.TOKEN_END.toLowerCase().trim(), 1.0);

                HashSet<String> wrongActions = new HashSet<>();
                for (String attr : costs.keySet()) {
                    String value = SFX.chooseNextValue(attr, actSeq.getSequence().get(index).getAttrValuesStillToBeMentionedAtThisPoint(), trainingData);

                    boolean eligibleWord = true;
                    if (value.isEmpty()
                            && !attr.equals("empty")
                            && !attr.equals(SFX.TOKEN_END)) {
                        eligibleWord = false;
                    } else if (!di.getMeaningRepresentation().getAttributes().containsKey(attr)
                            && !attr.equals(SFX.TOKEN_END)
                            && !(value.equals("empty") && attr.equals("empty"))) {
                        eligibleWord = false;
                    } else if (!attr.equals(SFX.TOKEN_END)) {
                        ArrayList<String> cleanAttrValuesSeq = new ArrayList<String>();
                        String previousAttrValue = "";
                        for (Action a : actSeq.getSequence().subList(0, index)) {
                            if (a.getWord().equals(SFX.TOKEN_START)) {
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
                        modSeq.modifyAndShortenSequence(index, new Action(SFX.TOKEN_START, attr + "=" + value));
                        //ROLL-OUT
                        costs.put(attr.trim().toLowerCase(), getPolicyRollOutCost(predicate, modSeq, di, trainedAttrClassifier_i.get(predicate), trainedWordClassifiers_i.get(predicate), valueAlignments, useReferenceRollout, trainingData, availableWordActions.get(predicate), nGrams));
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
                    costs.put(SFX.TOKEN_END, 0.0);
                    bestActionCost = 0.0;
                }
                for (String s : costs.keySet()) {
                    if (costs.get(s) != 1.0) {
                        costs.put(s, costs.get(s) - bestActionCost);
                    }
                }
                Instance in = generateAttrTrainingInstance(predicate, di, new ArrayList<Action>(actSeq.getSequence().subList(0, index)), valueAlignments, costs);

                if (in.getCorrectLabels().isEmpty()) {
                    System.out.println("NO COR");
                    System.out.println(predicate);
                    System.out.println(costs);
                    System.exit(0);
                }
                if (trainedAttrClassifier_i.get(predicate).isInstanceLeadingToFix(in)) {
                    earlyStop = true;
                    index = actSeq.getSequence().size() + 1;
                }
                if (earlyStop) {
                    if (earlyStopSteps >= JDAggerForSFX.earlyStopMaxFurtherSteps) {
                        index = actSeq.getSequence().size() + 1;
                    } else {
                        earlyStopSteps++;
                    }
                }
                newAttrTrainingInstances.get(di).get(predicate).add(in);
            }
        }

    }

    public void calculateSubReferences(DatasetInstance di, ArrayList<Action> actSeq) {
        ArrayList<Action> cleanActListPrev5Gram = new ArrayList<Action>();
        for (int i = actSeq.size() - 1; (i >= 0 && cleanActListPrev5Gram.size() < JDAggerForSFX.rollOutWindowSize); i--) {
            if (!actSeq.get(i).getWord().equals(SFX.TOKEN_START)
                    && !actSeq.get(i).getWord().equals(SFX.TOKEN_END)) {
                cleanActListPrev5Gram.add(0, actSeq.get(i));
            }
        }
        String actLastAttr = "";
        int lengthAfterAction = JDAggerForSFX.rollOutWindowSize + 1;
        boolean isForAttr = false;
        if (!actSeq.isEmpty()) {
            actLastAttr = actSeq.get(actSeq.size() - 1).getAttribute();
            if (actSeq.get(actSeq.size() - 1).getWord().equals(SFX.TOKEN_END)) {
                isForAttr = true;
            }
        } else {
            isForAttr = true;
        }
        ActionSequence cleanActSeqPrev5Gram = new ActionSequence(cleanActListPrev5Gram, 0.0, true);

        ArrayList<Action> ref = di.getTrainRealization();
        ArrayList<Action> cleanRef = new ArrayList<Action>();
        for (Action a : ref) {
            if (!a.getWord().equals(SFX.TOKEN_START)
                    && !a.getWord().equals(SFX.TOKEN_END)) {
                cleanRef.add(a);
            }
        }
        di.setAlignedSubRealization(ref, null);
        di.setRealizationCorrectAction(ref, null);
        String best5Gram = null;
        if (cleanActSeqPrev5Gram.getSequence().size() < JDAggerForSFX.rollOutWindowSize) {
            int start = cleanActSeqPrev5Gram.getSequence().size() - JDAggerForSFX.rollOutWindowSize;
            if (start < 0) {
                start = 0;
            }
            ArrayList<Action> subRefGram = null;
            if (cleanActSeqPrev5Gram.getSequence().size() + JDAggerForSFX.rollOutWindowSize + 1 < cleanRef.size()) {
                subRefGram = new ArrayList<Action>(cleanRef.subList(start, cleanActSeqPrev5Gram.getSequence().size() + JDAggerForSFX.rollOutWindowSize + 1));
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
                        //&& isForAttr
                        && !actLastAttr.isEmpty()
                        && !actSeq.get(actSeq.size() - 1).getWord().equals(SFX.TOKEN_END)) {
                    di.setRealizationCorrectAction(ref, SFX.TOKEN_END);
                } else {
                    di.setRealizationCorrectAction(ref, lastAction.getWord());
                }
            } else {
                di.setRealizationCorrectAction(ref, SFX.TOKEN_END);
            }
            best5Gram = "";
            for (Action a : cleanActSeqPrev5Gram.getSequence()) {
                best5Gram += a.getWord() + " ";
            }
            best5Gram = best5Gram.trim();
        } else {
            Double minCost = Double.MAX_VALUE;
            int bestAlignment = -1;
            int win = JDAggerForSFX.rollOutWindowSize;
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
                        best5Gram = subRef5GramStr;
                    }
                }
                win--;
                if (win == 0) {
                    bestAlignment = -2;
                }
            }
            if (bestAlignment >= 0) {
                int start = bestAlignment + win + 1 - JDAggerForSFX.rollOutWindowSize;
                if (start < 0) {
                    start = 0;
                }
                ArrayList<Action> subRefGram = null;
                if (start + JDAggerForSFX.rollOutWindowSize + lengthAfterAction < cleanRef.size()) {
                    subRefGram = new ArrayList<Action>(cleanRef.subList(start, start + JDAggerForSFX.rollOutWindowSize + lengthAfterAction));
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
                            && !actSeq.get(actSeq.size() - 1).getWord().equals(SFX.TOKEN_END)) {
                        di.setRealizationCorrectAction(ref, SFX.TOKEN_END);
                    } else {
                        di.setRealizationCorrectAction(ref, lastAction.getWord());
                    }
                } else {
                    di.setRealizationCorrectAction(ref, SFX.TOKEN_END);
                }
            }
        }
        int maxSize = -1;
        if (di.getAlignedSubRealization(ref).split(" ").length > maxSize) {
            maxSize = di.getAlignedSubRealization(ref).split(" ").length;
        }
        if (di.getAlignedSubRealization(ref).split(" ").length < maxSize) {
            di.setAlignedSubRealization(ref, null);
        }
    }

    public Double getPolicyRollOutCost(String predicate, ActionSequence actSeq, DatasetInstance di, JAROW classifierAttrs, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, double p, ArrayList<DatasetInstance> trainingData, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams) {
        double v = SFX.r.nextDouble();

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
        refs.put(new ActionSequence(di.getTrainRealization(), 0.0, true), di.getTrainRealization());

        if (pAS.getNoPunctLength() > 1 && pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(pAS.getSequence().get(pAS.getSequence().size() - 2).getWord())) {
            //Do not repeat the same word twice in a row
            return 1.0;
        } else {
            double minCost = 1.0;
            for (ActionSequence minRAS : refs.keySet()) {
                //If the end of the ATTR
                boolean resolved = false;
                if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(SFX.TOKEN_END)) {
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
                    if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(SFX.TOKEN_START)) {
                        String currentAttr = pAS.getSequence().get(pAS.getSequence().size() - 1).getAttribute().substring(0, pAS.getSequence().get(pAS.getSequence().size() - 1).getAttribute().indexOf('='));

                        if (availableWordActions.containsKey(currentAttr)
                                || currentAttr.equals(SFX.TOKEN_END)) {
                            HashSet<Action> dict = new HashSet<>();
                            ArrayList<Action> cleanMinRAS = new ArrayList<>();
                            for (Action a : minRAS.getSequence()) {
                                if (!a.getWord().equals(SFX.TOKEN_START)
                                        && !a.getWord().equals(SFX.TOKEN_END)
                                        && !a.getAttribute().equals(SFX.TOKEN_PUNCT)) {
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
                                        && (di.getMeaningRepresentation().getAttributes().containsKey(currentAttr)
                                        || currentAttr.equals("empty"))) {
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
                        double refCost = 1.0;
                        if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(di.getRealizationCorrectAction(refs.get(minRAS)))) {
                            refCost = 0.0;
                        } else {
                            refCost = ((double) JDAggerForSFX.rollOutWindowSize) / (JDAggerForSFX.rollOutWindowSize + JDAggerForSFX.rollOutWindowSize + 1.0);
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
        if (attrValuesToBeMentioned.isEmpty()) {
            attrValuesToBeMentioned.add("empty=empty");
        }
        while (!predictedAttr.equals(SFX.TOKEN_END) && predictedAttrValues.size() < SFX.maxAttrRealizationSize) {
            if (!predictedAttr.isEmpty()) {
                attrValuesToBeMentioned.remove(predictedAttr);
            }
            Instance attrTrainingVector = SFX.createAttrInstance(predicate, "@TOK@", predictedAttrValues, predictedActionsList, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableAttributeActions);

            if (attrTrainingVector != null) {
                Prediction predictAttr = classifierAttrs.predict(attrTrainingVector);
                predictedAttr = predictAttr.getLabel().trim();
                String predictedValue = "";
                if (!predictedAttr.equals(SFX.TOKEN_END)) {
                    predictedValue = SFX.chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);

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
                            predictedValue = SFX.chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);
                        }
                    }
                    predictedAttr += "=" + predictedValue;
                }

                predictedAttrValues.add(predictedAttr);

                String attribute = predictedAttrValues.get(predictedAttrValues.size() - 1).split("=")[0];
                String attrValue = predictedAttrValues.get(predictedAttrValues.size() - 1);
                predictedAttributes.add(attrValue);

                if (!attribute.equals(SFX.TOKEN_END)) {
                    predictedActionsList.add(new Action(SFX.TOKEN_START, predictedAttrValues.get(predictedAttrValues.size() - 1)));
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
                        while (!predictedWord.equals(SFX.TOKEN_END) && predictedWordList.size() < SFX.maxWordRealizationSize) {

                            ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                            for (int i = 0; i < predictedAttributes.size() - 1; i++) {
                                predictedAttributesForInstance.add(predictedAttributes.get(i));
                            }
                            if (!predictedAttributes.get(predictedAttributes.size() - 1).equals(attrValue)) {
                                predictedAttributesForInstance.add(predictedAttributes.get(predictedAttributes.size() - 1));
                            }
                            Instance wordTrainingVector = SFX.createWordInstance(predicate, new Action("@TOK@", predictedAttrValues.get(predictedAttrValues.size() - 1)), predictedAttributesForInstance, predictedActionsList, isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions, nGrams, false);

                            if (wordTrainingVector != null) {
                                if (classifierWords.get(attribute) != null) {
                                    Prediction predictWord = classifierWords.get(attribute).predict(wordTrainingVector);

                                    if (predictWord.getLabel() != null) {
                                        predictedWord = predictWord.getLabel().trim();
                                    } else {
                                        predictedWord = SFX.TOKEN_END;
                                    }
                                    predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                    predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                                    if (!predictedWord.equals(SFX.TOKEN_START)
                                            && !predictedWord.equals(SFX.TOKEN_END)) {
                                        subPhrase.add(predictedWord);
                                        predictedWordList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                    }
                                } else {
                                    predictedWord = SFX.TOKEN_END;
                                    predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                    predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
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
                                                || valueToCheck.equals("empty")
                                                || valueToCheck.equals("dont_care")) {
                                            valueToCheck = attribute + ":" + valueTBM;
                                        }
                                        if (!valueToCheck.equals("empty:empty")
                                                && valueAlignments.containsKey(valueToCheck)) {
                                            for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                                if (SFX.endsWith(subPhrase, alignedStr)) {
                                                    isValueMentioned = true;
                                                    break;
                                                }
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
                                                    || valueToCheck.equals("empty")
                                                    || valueToCheck.equals("dont_care")) {
                                                valueToCheck = attrValueTBM.replace("=", ":");
                                            }
                                            if (!valueToCheck.equals("empty:empty")
                                                    && valueAlignments.containsKey(valueToCheck)) {
                                                for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                                    if (SFX.endsWith(subPhrase, alignedStr)) {
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
                        if (predictedWordList.size() >= SFX.maxWordRealizationSize
                                && !predictedActionsList.get(predictedActionsList.size() - 1).getWord().equals(SFX.TOKEN_END)) {
                            predictedWord = SFX.TOKEN_END;
                            predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                            predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                        }
                    } else {
                        predictedActionsList.add(new Action(SFX.TOKEN_END, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                        predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                    }
                } else {
                    predictedActionsList.add(new Action(SFX.TOKEN_END, SFX.TOKEN_END));
                    predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                }
            }
        }
        if (predictedAttrValues.size() >= SFX.maxAttrRealizationSize
                && !predictedActionsList.get(predictedActionsList.size() - 1).getAttribute().equals(SFX.TOKEN_END)) {
            predictedActionsList.add(new Action(SFX.TOKEN_END, SFX.TOKEN_END));
            predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
        }
        HashSet<ActionSequence> refs = new HashSet<>();
        refs.add(new ActionSequence(di.getTrainRealization(), 0.0, true));

        String previous = "";
        boolean end = false;
        boolean open = false;
        for (int i = 0; i < predictedActionsList.size(); i++) {
            if (previous.equals(SFX.TOKEN_START)
                    && predictedActionsList.get(i).getWord().equals(SFX.TOKEN_START)) {
                end = true;
            }
            if (predictedActionsList.get(i).getWord().equals(SFX.TOKEN_START)) {
                if (open) {
                    end = true;
                }
                open = true;
            }
            if (predictedActionsList.get(i).getWord().equals(SFX.TOKEN_END)
                    && !predictedActionsList.get(i).getAttribute().equals(SFX.TOKEN_END)) {
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
        //System.out.println("===========START===========" + pAS.getSequence());

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
            if (attrValuesToBeMentioned.isEmpty()) {
                attrValuesToBeMentioned.add("empty=empty");
            }
            int generatedWords = 0;
            while (!predictedAttr.equals(SFX.TOKEN_END) && predictedAttrValues.size() < SFX.maxAttrRealizationSize && generatedWords < JDAggerForSFX.rollOutWindowSize) {
                predictedAttr = "";
                if (predictedActionsList.size() < pAS.getSequence().size()) {
                    if (!pAS.getSequence().get(predictedActionsList.size()).getWord().equals(SFX.TOKEN_START)) {
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
                    Instance attrTrainingVector = SFX.createAttrInstance(predicate, "@TOK@", predictedAttrValues, predictedActionsList, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableAttributeActions);

                    if (attrTrainingVector != null) {
                        Prediction predictAttr = classifierAttrs.predict(attrTrainingVector);
                        predictedAttr = predictAttr.getLabel().trim();
                        String predictedValue = "";
                        if (!predictedAttr.equals(SFX.TOKEN_END)) {
                            predictedValue = SFX.chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);

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
                                    predictedValue = SFX.chooseNextValue(predictedAttr, attrValuesToBeMentioned, trainingData);
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

                if (!attribute.equals(SFX.TOKEN_END)) {
                    predictedActionsList.add(new Action(SFX.TOKEN_START, predictedAttrValues.get(predictedAttrValues.size() - 1)));
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
                        while (!predictedWord.equals(SFX.TOKEN_END) && predictedWordList.size() < SFX.maxWordRealizationSize && generatedWords < JDAggerForSFX.rollOutWindowSize) {
                            if (predictedActionsList.size() < pAS.getSequence().size()) {
                                predictedWord = pAS.getSequence().get(predictedActionsList.size()).getWord();
                                predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                if (!predictedWord.equals(SFX.TOKEN_START)
                                        && !predictedWord.equals(SFX.TOKEN_END)) {
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
                                Instance wordTrainingVector = SFX.createWordInstance(predicate, new Action("@TOK@", predictedAttrValues.get(predictedAttrValues.size() - 1)), predictedAttributesForInstance, predictedActionsList, isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions, nGrams, false);

                                if (wordTrainingVector != null) {
                                    if (classifierWords.get(attribute) != null) {
                                        Prediction predictWord = classifierWords.get(attribute).predict(wordTrainingVector);
                                        if (predictWord.getLabel() != null) {
                                            predictedWord = predictWord.getLabel().trim();
                                        } else {
                                            predictedWord = SFX.TOKEN_END;
                                        }
                                        predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                        if (!predictedWord.equals(SFX.TOKEN_START)
                                                && !predictedWord.equals(SFX.TOKEN_END)) {
                                            generatedWords++;
                                            subPhrase.add(predictedWord);
                                            newPredictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                            predictedWordList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                                        }
                                    } else {
                                        predictedWord = SFX.TOKEN_END;
                                        predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
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
                                                || valueToCheck.equals("empty")
                                                || valueToCheck.equals("dont_care")) {
                                            valueToCheck = attribute + ":" + valueTBM;
                                        }
                                        if (!valueToCheck.equals("empty:empty")
                                                && valueAlignments.containsKey(valueToCheck)) {
                                            for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                                if (SFX.endsWith(subPhrase, alignedStr)) {
                                                    isValueMentioned = true;
                                                    break;
                                                }
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
                                                    || valueToCheck.equals("empty")
                                                    || valueToCheck.equals("dont_care")) {
                                                valueToCheck = attrValueTBM.replace("=", ":");
                                            }
                                            if (!valueToCheck.equals("empty:empty")
                                                    && valueAlignments.containsKey(valueToCheck)) {
                                                for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                                    if (SFX.endsWith(subPhrase, alignedStr)) {
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
                        if (predictedWordList.size() >= SFX.maxWordRealizationSize
                                && !predictedActionsList.get(predictedActionsList.size() - 1).getWord().equals(SFX.TOKEN_END)) {
                            predictedWord = SFX.TOKEN_END;
                            predictedActionsList.add(new Action(predictedWord, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                            predictedActionsList.get(predictedActionsList.size() - 1).setAttrValuesStillToBeMentionedAtThisPoint(attrValuesToBeMentioned);
                        }
                    } else {
                        predictedActionsList.add(new Action(SFX.TOKEN_END, predictedAttrValues.get(predictedAttrValues.size() - 1)));
                    }
                } else {
                    predictedActionsList.add(new Action(SFX.TOKEN_END, SFX.TOKEN_END));
                }
            }
            ArrayList<Action> newCleanPredictedActionsList = new ArrayList<>();
            for (Action a : newPredictedActionsList) {
                //Essentially skip punctuation
                if (!a.getAttribute().equals(SFX.TOKEN_PUNCT)
                        && !a.getWord().equals(SFX.TOKEN_START)
                        && !a.getWord().equals(SFX.TOKEN_END)) {
                    newCleanPredictedActionsList.add(a);
                }
            }
            int added = 0;
            for (int i = pAS.getSequence().size() - 1; added <= JDAggerForSFX.rollOutWindowSize && i >= 0; i--) {
                //Essentially skip punctuation
                if (!pAS.getSequence().get(i).getAttribute().equals(SFX.TOKEN_PUNCT)
                        && !pAS.getSequence().get(i).getWord().equals(SFX.TOKEN_START)
                        && !pAS.getSequence().get(i).getWord().equals(SFX.TOKEN_END)) {
                    newCleanPredictedActionsList.add(0, pAS.getSequence().get(i));
                    added++;
                }
            }

            double minCost = 1.0;
            //If the end of the ATTR
            if (examinedAction.getWord().equals(SFX.TOKEN_END)) {
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
            if (di.getAlignedSubRealization(di.getTrainRealization()) != null) {
                refWindows.add(di.getAlignedSubRealization(di.getTrainRealization()));
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

    public Instance generateWordTrainingInstance(String predicate, String currectAttrValue, String nextValue, DatasetInstance di, ArrayList<Action> generatedActions, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, TObjectDoubleHashMap<String> costs, HashMap<String, HashSet<Action>> availableWordActions, HashMap<Integer, HashSet<String>> nGrams) {
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
        if (attrValuesToBeMentioned.isEmpty()) {
            attrValuesToBeMentioned.add("empty=empty");
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
            if (!attribute.equals(SFX.TOKEN_END)) {
                String predictedWord = generatedActions.get(i).getWord();
                if (!predictedWord.equals(SFX.TOKEN_START)
                        && !predictedWord.equals(SFX.TOKEN_END)) {
                    subPhrase.add(predictedWord);
                }

                predictedWordsList.add(new Action(predictedWord, attribute));
                if (!isValueMentioned) {
                    if (predictedWordsList.get(i).getWord().startsWith(SFX.TOKEN_X)
                            && (valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+")
                            || valueTBM.startsWith(SFX.TOKEN_X))) {
                        isValueMentioned = true;
                    } else if (!predictedWordsList.get(i).getWord().startsWith(SFX.TOKEN_X)
                            && !(valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+")
                            || valueTBM.startsWith(SFX.TOKEN_X))) {
                        String valueToCheck = valueTBM;
                        if (valueToCheck.equals("no")
                                || valueToCheck.equals("yes")
                                || valueToCheck.equals("yes or no")
                                || valueToCheck.equals("none")
                                || valueToCheck.equals("empty")
                                || valueToCheck.equals("dont_care")) {
                            valueToCheck = attribute.replace("=", ":");
                        }
                        if (!valueToCheck.equals("empty:empty")
                                && valueAlignments.containsKey(valueToCheck)) {
                            for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                if (SFX.endsWith(subPhrase, alignedStr)) {
                                    isValueMentioned = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (isValueMentioned) {
                        attrValuesAlreadyMentioned.add(attribute);
                        attrValuesToBeMentioned.remove(attribute);
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
                                        || valueToCheck.equals("empty")
                                        || valueToCheck.equals("dont_care")) {
                                    valueToCheck = attrValueTBM.substring(0, attrValueTBM.indexOf('=')) + ":" + value;
                                }
                                if (!valueToCheck.equals("empty:empty")
                                        && valueAlignments.containsKey(valueToCheck)) {
                                    for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                        if (SFX.endsWith(subPhrase, alignedStr)) {
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
        ArrayList<String> predictedAttributes = new ArrayList<>();
        for (int i = 0; i < generatedActions.size(); i++) {
            Action act = predictedWordsList.get(i);
            if (predictedAttributes.isEmpty()) {
                if (!act.getAttribute().equals(SFX.TOKEN_END) && !act.getAttribute().equals(currectAttrValue)) {
                    predictedAttributes.add(act.getAttribute());
                }
            } else {
                if (!act.getAttribute().equals(SFX.TOKEN_END)
                        && !act.getAttribute().equals(predictedAttributes.get(predictedAttributes.size() - 1))
                        && !act.getAttribute().equals(currectAttrValue)) {
                    predictedAttributes.add(act.getAttribute());
                }
            }
        }
        return SFX.createWordInstance(predicate, currectAttrValue, predictedAttributes, predictedWordsList, costs, isValueMentioned, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableWordActions, nGrams);
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
        if (attrValuesToBeMentioned.isEmpty()) {
            attrValuesToBeMentioned.add("empty=empty");
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
            if (!attrValue.equals(SFX.TOKEN_END)) {
                String predictedWord = generatedActions.get(i).getWord();
                if (!predictedWord.equals(SFX.TOKEN_START)
                        && !predictedWord.equals(SFX.TOKEN_END)) {
                    subPhrase.add(predictedWord);
                }

                predictedActionsList.add(new Action(predictedWord, attrValue));
                if (!isValueMentioned) {
                    if (predictedActionsList.get(i).getWord().startsWith(SFX.TOKEN_X)
                            && (valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+")
                            || valueTBM.startsWith(SFX.TOKEN_X))) {
                        isValueMentioned = true;
                    } else if (!predictedActionsList.get(i).getWord().startsWith(SFX.TOKEN_X)
                            && !(valueTBM.matches("\"[xX][0-9]+\"") || valueTBM.matches("[xX][0-9]+")
                            || valueTBM.startsWith(SFX.TOKEN_X))) {
                        String valueToCheck = valueTBM;
                        if (valueToCheck.equals("no")
                                || valueToCheck.equals("yes")
                                || valueToCheck.equals("yes or no")
                                || valueToCheck.equals("none")
                                || valueToCheck.equals("empty")
                                || valueToCheck.equals("dont_care")) {
                            String attribute = attrValue;
                            if (attribute.contains("=")) {
                                attribute = attrValue.substring(0, attrValue.indexOf('='));
                            }
                            valueToCheck = attribute + ":" + valueTBM;
                        }
                        if (!valueToCheck.equals("empty:empty")
                                && valueAlignments.containsKey(valueToCheck)) {
                            for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                if (SFX.endsWith(subPhrase, alignedStr)) {
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
                                        || valueToCheck.equals("empty")
                                        || valueToCheck.equals("dont_care")) {
                                    valueToCheck = attrValueTBM.substring(0, attrValueTBM.indexOf('=')) + ":" + value;
                                }
                                if (!valueToCheck.equals("empty:empty")
                                        && valueAlignments.containsKey(valueToCheck)) {
                                    for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                        if (SFX.endsWith(subPhrase, alignedStr)) {
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
        return SFX.createAttrInstance(predicate, predictedAttrValues, predictedActionsList, costs, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableAttributeActions);
    }
}
