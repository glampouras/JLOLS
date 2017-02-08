
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
import imitationNLG.DatasetParser;
import jarow.Instance;
import jarow.JAROW;
import jarow.Prediction;
import jarow.TrainAdditionalThread;
import jarow.TrainThread;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 *
 * @author Gerasimos Lampouras
 */
public class JLOLS {

    final static int THREADS_COUNT = Runtime.getRuntime().availableProcessors();

    boolean print = false;

    /**
     *
     */
    public static double earlyStopMaxFurtherSteps = 0;

    /**
     *
     */
    public static double p = 0.2;
    static int rollOutWindowSize = 5;

    /**
     *
     */
    public static int checkIndex = -1;
    DatasetParser datasetParser = null;

    HashMap<String, JAROW> trainedAttrClassifiers_0 = new HashMap<>();
    HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0 = new HashMap<>();

    /**
     *
     */
    public static WordSequenceCache<String, ActionSequence> wordSequenceCache = new WordSequenceCache<>(2000000, 500000, 10000);

    /**
     *
     */
    public static WordSequenceCache<String, Double> costCache = new WordSequenceCache<>(20000000, 5000000, 10000);

    /**
     *
     * @param dataset
     */
    public JLOLS(DatasetParser dataset) {
        this.datasetParser = dataset;
    }

    /**
     *
     * @param availableAttributeActions
     * @param trainingData
     * @param trainingAttrInstances
     * @param trainingWordInstances
     * @param availableWordActions
     * @param valueAlignments
     * @param beta
     * @param testingData
     * @param detailedResults
     * @return
     */
    public Object[] runLOLS(HashMap<String, HashSet<String>> availableAttributeActions, ArrayList<DatasetInstance> trainingData, HashMap<String, ArrayList<Instance>> trainingAttrInstances, HashMap<String, HashMap<String, ArrayList<Instance>>> trainingWordInstances, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, double beta, ArrayList<DatasetInstance> testingData, boolean detailedResults) {
        ArrayList<HashMap<String, JAROW>> trainedAttrClassifiers = new ArrayList<>();
        ArrayList<HashMap<String, HashMap<String, JAROW>>> trainedWordClassifiers = new ArrayList<>();
        //INITIALIZE A POLICY P_0 (initializing on ref)

        HashMap<String, ArrayList<Instance>> totalTrainingAttrInstances = new HashMap<>();
        HashMap<String, HashMap<String, ArrayList<Instance>>> totalTrainingWordInstances = new HashMap<>();

        for (String predicate : trainingAttrInstances.keySet()) {
            trainedAttrClassifiers_0.put(predicate, new JAROW());
            if (!totalTrainingAttrInstances.containsKey(predicate)) {
                totalTrainingAttrInstances.put(predicate, new ArrayList<Instance>());
            }
            totalTrainingAttrInstances.get(predicate).addAll(trainingAttrInstances.get(predicate));
            if (availableAttributeActions.containsKey(predicate)) {
                for (String attribute : availableAttributeActions.get(predicate)) {
                    if (!attribute.equals(Action.TOKEN_END)) {
                        if (trainingWordInstances.get(predicate).containsKey(attribute) && !trainingWordInstances.get(predicate).get(attribute).isEmpty()) {
                            if (!trainedWordClassifiers_0.containsKey(predicate)) {
                                trainedWordClassifiers_0.put(predicate, new HashMap<String, JAROW>());
                            }
                            trainedWordClassifiers_0.get(predicate).put(attribute, new JAROW());

                            if (!totalTrainingWordInstances.containsKey(predicate)) {
                                totalTrainingWordInstances.put(predicate, new HashMap<String, ArrayList<Instance>>());
                            }
                            if (!totalTrainingWordInstances.get(predicate).containsKey(attribute)) {
                                totalTrainingWordInstances.get(predicate).put(attribute, new ArrayList<Instance>());
                            }
                            totalTrainingWordInstances.get(predicate).get(attribute).addAll(trainingWordInstances.get(predicate).get(attribute));
                        } else {
                            System.out.println("EMPTY {" + predicate + ": " + attribute + "}");
                        }
                    }
                }
            }
        }

        if (datasetParser.resetLists || !datasetParser.loadInitClassifiers(trainingData.size(), trainedAttrClassifiers_0, trainedWordClassifiers_0)) {
            System.out.print("Initial training...");
            ExecutorService executorTrain = Executors.newFixedThreadPool(THREADS_COUNT);
            for (String predicate : trainingAttrInstances.keySet()) {
                executorTrain.execute(new TrainThread(trainedAttrClassifiers_0.get(predicate), trainingAttrInstances.get(predicate), datasetParser.averaging, datasetParser.shuffling, datasetParser.rounds, datasetParser.initialTrainingParam, datasetParser.adapt));
                //trainedAttrClassifiers_0.put(predicate, trainClassifier(trainingAttrInstances.get(predicate), param, adapt));       
                if (availableAttributeActions.containsKey(predicate)) {
                    for (String attribute : availableAttributeActions.get(predicate)) {
                        if (!attribute.equals(Action.TOKEN_END)
                                && trainingWordInstances.get(predicate).containsKey(attribute) && !trainingWordInstances.get(predicate).get(attribute).isEmpty()) {
                            executorTrain.execute(new TrainThread(trainedWordClassifiers_0.get(predicate).get(attribute), trainingWordInstances.get(predicate).get(attribute), datasetParser.averaging, datasetParser.shuffling, datasetParser.rounds, datasetParser.initialTrainingParam, datasetParser.adapt));
                            //trainedWordClassifiers_0.get(predicate).put(attribute, trainClassifier(trainingWordInstances.get(predicate).get(attribute), param, adapt));
                        }
                    }
                }
            }
            executorTrain.shutdown();
            while (!executorTrain.isTerminated()) {
            }
            System.out.println("done!");
            datasetParser.writeInitClassifiers(trainingData.size(), trainedAttrClassifiers_0, trainedWordClassifiers_0);
        }

        trainedAttrClassifiers.add(trainedAttrClassifiers_0);
        trainedWordClassifiers.add(trainedWordClassifiers_0);
        datasetParser.evaluateGeneration(trainedAttrClassifiers_0, trainedWordClassifiers_0, testingData, availableAttributeActions, availableWordActions, true, -1, detailedResults);

        System.out.println("**************LOLS COMMENCING**************");
        checkIndex = -1;
        int epochs = 10;
        HashMap<String, ArrayList<DatasetInstance>> trainingDataPerPredicate = new HashMap<>();
        for (String predicate : trainingAttrInstances.keySet()) {
            trainingDataPerPredicate.put(predicate, new ArrayList<DatasetInstance>());
        }
        for (DatasetInstance di : trainingData) {
            trainingDataPerPredicate.get(di.getMeaningRepresentation().getPredicate()).add(di);
        }

        for (int e = 0; e < epochs; e++) {
            wordSequenceCache = new WordSequenceCache<>(2000000, 500000, 10000);
            runSFXLOLSOnInstance.avgContentErrors = 0;
            runSFXLOLSOnInstance.avgWordErrors = 0;
            if (e == 0) {
                beta = 1.0;
            } else {
                beta = Math.pow(1.0 - p, e);
            }
            System.out.println("beta = " + beta + " , p = " + p + " , early = " + earlyStopMaxFurtherSteps);

            HashMap<String, JAROW> trainedAttrClassifier_i = trainedAttrClassifiers.get(trainedAttrClassifiers.size() - 1);
            HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_i = trainedWordClassifiers.get(trainedWordClassifiers.size() - 1);

            ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>> newAttrTrainingInstances = new ConcurrentHashMap<>();
            ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>>> newWordTrainingInstances = new ConcurrentHashMap<>();

            ArrayList<DatasetInstance> subTrainingData = new ArrayList<>();
            if (datasetParser.useSubsetData) {
                for (String predicate : trainingDataPerPredicate.keySet()) {
                    int to = (int) Math.round((((e + 1.0) * 20.0) / 100.0) * trainingDataPerPredicate.get(predicate).size());
                    if (to >= trainingDataPerPredicate.get(predicate).size()) {
                        subTrainingData.addAll(trainingDataPerPredicate.get(predicate));
                    } else {
                        subTrainingData.addAll(trainingDataPerPredicate.get(predicate).subList(0, to));
                    }
                }
            } else {
                subTrainingData.addAll(trainingData);
            }
            for (DatasetInstance di : subTrainingData) {
                newAttrTrainingInstances.put(di, new ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>());
                newWordTrainingInstances.put(di, new ConcurrentHashMap<String, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>>());

                for (String predicate : trainingAttrInstances.keySet()) {
                    newAttrTrainingInstances.get(di).put(predicate, new CopyOnWriteArrayList<Instance>());
                    newWordTrainingInstances.get(di).put(predicate, new ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>());
                    if (availableAttributeActions.containsKey(predicate)) {
                        for (String attr : availableAttributeActions.get(predicate)) {
                            newWordTrainingInstances.get(di).get(predicate).put(attr, new CopyOnWriteArrayList<Instance>());
                        }
                    }
                }
            }

            System.out.print("Run LOLS..." + THREADS_COUNT + "...");
            ExecutorService executor = Executors.newFixedThreadPool(THREADS_COUNT);
            for (DatasetInstance di : subTrainingData) {
                //if (di.getMeaningRepresentation().getPredicate().equals("inform_no_match")) {
                executor.execute(new runSFXLOLSOnInstance(datasetParser, beta, di, availableAttributeActions, trainedAttrClassifier_i, trainedWordClassifiers_i, valueAlignments, availableWordActions, subTrainingData, newAttrTrainingInstances, newWordTrainingInstances));
                //break;
                //}
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            System.out.println("done!");
            System.out.println("avgContentErrors = " + runSFXLOLSOnInstance.avgContentErrors);
            System.out.println("avgWordErrors = " + runSFXLOLSOnInstance.avgWordErrors);

            System.out.print("Create new classifiers...");
            HashMap<String, ArrayList<Instance>> totalNewAttrTrainingInstances = new HashMap<String, ArrayList<Instance>>();
            HashMap<String, HashMap<String, ArrayList<Instance>>> totalNewWordTrainingInstances = new HashMap<>();
            for (DatasetInstance di : subTrainingData) {
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
                    if (availableAttributeActions.containsKey(predicate)) {
                        for (String attr : availableAttributeActions.get(predicate)) {
                            if (!totalNewWordTrainingInstances.get(predicate).containsKey(attr)) {
                                totalNewWordTrainingInstances.get(predicate).put(attr, new ArrayList<Instance>());
                            }
                            totalNewWordTrainingInstances.get(predicate).get(attr).addAll(newWordTrainingInstances.get(di).get(predicate).get(attr));
                        }
                    }
                }
            }
            System.out.println("done!");

            //UPDATE CLASSIFIERS
            System.out.print("Update classifiers...");
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
            ExecutorService executorTrainAdditional = Executors.newFixedThreadPool(THREADS_COUNT);
            for (String predicate : trainingAttrInstances.keySet()) {
                totalTrainingAttrInstances.get(predicate).addAll(totalNewAttrTrainingInstances.get(predicate));

                if (trainedWordClassifiers_i.containsKey(predicate)) {
                    for (String attr : trainedWordClassifiers_i.get(predicate).keySet()) {
                        if (!totalNewWordTrainingInstances.get(predicate).get(attr).isEmpty()) {
                            totalTrainingWordInstances.get(predicate).get(attr).addAll(totalNewWordTrainingInstances.get(predicate).get(attr));
                        }
                    }
                }
            }
            for (String predicate : totalNewAttrTrainingInstances.keySet()) {
                executorTrainAdditional.execute(new TrainAdditionalThread(trainedAttrClassifier_ii.get(predicate), totalNewAttrTrainingInstances.get(predicate), datasetParser.averaging, datasetParser.shuffling, datasetParser.rounds, datasetParser.additionalTrainingParam, datasetParser.adapt));
                //trainedAttrClassifier_ii.get(predicate).trainAdditional(new ArrayList<Instance>(totalNewAttrTrainingInstances.get(predicate)), true, false, 10, adapt, additionalParam);

                if (totalNewWordTrainingInstances.containsKey(predicate)) {
                    for (String attr : totalNewWordTrainingInstances.get(predicate).keySet()) {
                        if (!totalNewWordTrainingInstances.get(predicate).get(attr).isEmpty()) {
                            /*for (Instance in : totalNewWordTrainingInstances.get(predicate).get(attr)) {
                                for (String action : in.getCorrectLabels()) {
                                    if (!trainedWordClassifiers_ii.get(predicate).get(attr).getCurrentVarianceVectors().containsKey(action)) {
                                        System.out.println("WTRF");
                                        System.out.println(predicate);
                                        System.out.println(attr);
                                        System.out.println(availableWordActions.get(predicate).get(attr));
                                        System.out.println(trainedWordClassifiers_ii.get(predicate).get(attr).getCurrentVarianceVectors().keySet());
                                        System.out.println(action);
                                        //System.exit(0);
                                    }
                                }
                            }*/
                            executorTrainAdditional.execute(new TrainAdditionalThread(trainedWordClassifiers_ii.get(predicate).get(attr), totalNewWordTrainingInstances.get(predicate).get(attr), datasetParser.averaging, datasetParser.shuffling, datasetParser.rounds, datasetParser.additionalTrainingParam, datasetParser.adapt));
                            //trainedWordClassifiers_ii.get(predicate).get(attr).trainAdditional(totalNewWordTrainingInstances.get(predicate).get(attr), true, false, 10, adapt, additionalParam);
                        }
                    }
                }
            }
            executorTrainAdditional.shutdown();
            while (!executorTrainAdditional.isTerminated()) {
            }
            System.out.println("done!");

            trainedAttrClassifiers.add(trainedAttrClassifier_ii);
            trainedWordClassifiers.add(trainedWordClassifiers_ii);

            //FIRST NEED TO AVERAGE OVER ALL CLASSIFIERS
            System.out.print("Averaging classifiers...");
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
                if (availableAttributeActions.containsKey(predicate)) {
                    for (String attribute : availableAttributeActions.get(predicate)) {
                        if (!attribute.equals(Action.TOKEN_END)) {
                            reorganizedClassifiersWords.get(predicate).put(attribute, new ArrayList<JAROW>());
                        }
                    }
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
                if (availableAttributeActions.containsKey(predicate)) {
                    for (String attribute : availableAttributeActions.get(predicate)) {
                        if (!attribute.equals(Action.TOKEN_END)) {
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
                }
            }
            System.out.println("done!");

            System.out.println("AVERAGE CLASSIFIER at epoch = " + e);
            datasetParser.evaluateGeneration(avgClassifiersAttrs, avgClassifiersWords, testingData, availableAttributeActions, availableWordActions, true, e + 1, detailedResults);
        }

        /*HashMap<String, ArrayList<JAROW>> reorganizedClassifiersAttrs = new HashMap<>();
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
            if (availableAttributeActions.containsKey(predicate)) {
                for (String attribute : availableAttributeActions.get(predicate)) {
                    reorganizedClassifiersWords.get(predicate).put(attribute, new ArrayList<JAROW>());
                }
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
            if (availableAttributeActions.containsKey(predicate)) {
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
        }
        System.out.println("AVERAGE CLASSIFIER");
        SFX.evaluateGeneration(avgClassifiersAttrs, avgClassifiersWords, testingData, availableAttributeActions, availableWordActions, nGrams, true, 150);

        HashMap<String, JAROW> trainedAttrClassifiers_retrain2 = new HashMap<>();
        HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_retrain2 = new HashMap<>();
        for (String predicate : trainingAttrInstances.keySet()) {
            trainedAttrClassifiers_retrain2.put(predicate, trainClassifier(totalTrainingAttrInstances.get(predicate), avgClassifiersAttrs.get(predicate).getParam(), adapt));
            trainedWordClassifiers_retrain2.put(predicate, new HashMap<String, JAROW>());
            if (availableAttributeActions.containsKey(predicate)) {
                for (String attribute : availableAttributeActions.get(predicate)) {
                    if (trainingWordInstances.get(predicate).containsKey(attribute) && !trainingWordInstances.get(predicate).get(attribute).isEmpty()) {
                        trainedWordClassifiers_retrain2.get(predicate).put(attribute, trainClassifier(totalTrainingWordInstances.get(predicate).get(attribute), avgClassifiersWords.get(predicate).get(attribute).getParam(), adapt));
                    }
                }
            }
        }
        System.out.println("TOTAL (NON OPT) CLASSIFIER");
        SFX.evaluateGeneration(trainedAttrClassifiers_retrain2, trainedWordClassifiers_retrain2, testingData, availableAttributeActions, availableWordActions, nGrams, true, 200);

        Object[] results = new Object[2];
        results[0] = avgClassifiersAttrs;
        results[1] = avgClassifiersWords;
        return results;*/
        return null;
    }
}

class runSFXLOLSOnInstance extends Thread {

    DatasetParser datasetParser;
    double beta;
    DatasetInstance di;
    HashMap<String, HashSet<String>> availableAttributeActions;
    HashMap<String, JAROW> trainedAttrClassifier_i;
    HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_i;
    HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments;
    HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions;
    ArrayList<DatasetInstance> trainingData;
    ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>> newAttrTrainingInstances;
    ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>>> newWordTrainingInstances;

    boolean printDebugInfo = false;

    public static int avgContentErrors;
    public static int avgWordErrors;

    private double costBalance1 = 0.9999;
    private double costBalance2 = 0.0001;

    public runSFXLOLSOnInstance(DatasetParser datasetParser, double beta, DatasetInstance di, HashMap<String, HashSet<String>> availableAttributeActions, HashMap<String, JAROW> trainedAttrClassifier_i, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_i, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions, ArrayList<DatasetInstance> trainingData, ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>> newAttrTrainingInstances, ConcurrentHashMap<DatasetInstance, ConcurrentHashMap<String, ConcurrentHashMap<String, CopyOnWriteArrayList<Instance>>>> newWordTrainingInstances) {
        this.datasetParser = datasetParser;
        this.beta = beta;

        this.di = di;

        this.availableAttributeActions = availableAttributeActions;
        this.trainedAttrClassifier_i = trainedAttrClassifier_i;
        this.trainedWordClassifiers_i = trainedWordClassifiers_i;
        this.valueAlignments = valueAlignments;

        this.availableWordActions = availableWordActions;
        this.trainingData = trainingData;

        this.newAttrTrainingInstances = newAttrTrainingInstances;
        this.newWordTrainingInstances = newWordTrainingInstances;
    }

    public void run() {
        String predicate = di.getMeaningRepresentation().getPredicate();
        //ROLL-IN
        //System.out.println(di.getMeaningRepresentation().getMRstr());
        ActionSequence rollInContentSequence = getLearnedPolicyRollIn_Content(predicate, di, trainedAttrClassifier_i.get(predicate));

        boolean earlyStop = false;
        int earlyStopSteps = 0;

        //FOR EACH ACTION IN ROLL-IN SEQUENCE
        for (int index = 0; index < rollInContentSequence.getSequence().size(); index++) {
            //FOR EACH POSSIBLE ALTERNATIVE ACTION

            //Make the same decisions for all action substitutions
            boolean useReferenceRollout = false;
            double v = datasetParser.randomGen.nextDouble();
            if (v < beta) {
                useReferenceRollout = true;
            }

            //ALTERNATIVE ATTRS
            TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
            for (String attr : availableAttributeActions.get(predicate)) {
                if (di.getMeaningRepresentation().getAttributes().keySet().contains(attr)) {
                    costs.put(attr.toLowerCase().trim(), 1.0);
                }
            }
            costs.put(Action.TOKEN_END.toLowerCase().trim(), 1.0);

            HashSet<String> wrongActions = new HashSet<>();
            for (String attr : costs.keySet()) {
                String value = "";
                boolean eligibleWord = true;
                if (!attr.equals(Action.TOKEN_END)) {
                    if (rollInContentSequence.getSequence().get(index).getAttrValuesAfterThisPointInContentSequence() == null) {
                        System.out.println(attr + " " + rollInContentSequence.getSequence().get(index));
                    }
                    value = datasetParser.chooseNextValue(attr, rollInContentSequence.getSequence().get(index).getAttrValuesAfterThisPointInContentSequence());
                    if (value.isEmpty()
                            && !attr.equals("empty")) {
                        eligibleWord = false;
                    } else if (!di.getMeaningRepresentation().getAttributes().containsKey(attr)
                            && !(value.equals("empty") && attr.equals("empty"))) {
                        eligibleWord = false;
                    } else {
                        ArrayList<String> cleanAttrValuesSeq = new ArrayList<String>();
                        String previousAttrValue = "";
                        for (Action a : rollInContentSequence.getSequence().subList(0, index)) {
                            if (a.getWord().equals(Action.TOKEN_START)) {
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
                }
                if (eligibleWord) {
                    ActionSequence modifiedSequence = new ActionSequence(rollInContentSequence);
                    if (!attr.equals(Action.TOKEN_END)) {
                        modifiedSequence.modifyAndShortenSequence(index, new Action(Action.TOKEN_START, attr + "=" + value));
                    } else {
                        modifiedSequence.modifyAndShortenSequence(index, new Action(Action.TOKEN_END, attr));
                    }
                    //ROLL-OUT
                    //System.out.println(">>> " + modifiedSequence.getSequence().get(modifiedSequence.getSequence().size() - 1).getAttrValuesAfterThisPointInContentSequence());                    
                    costs.put(attr.trim().toLowerCase(), getPolicyRollOutCost_Content(predicate, modifiedSequence, di, trainedAttrClassifier_i.get(predicate), trainedWordClassifiers_i.get(predicate), valueAlignments, useReferenceRollout, trainingData, availableWordActions.get(predicate)));
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
                costs.put(Action.TOKEN_END, 0.0);
                bestActionCost = 0.0;
            }
            for (String s : costs.keySet()) {
                if (costs.get(s) != 1.0) {
                    costs.put(s, costs.get(s) - bestActionCost);
                }
            }
            Instance in = datasetParser.createAttrInstanceWithCosts(predicate, costs, new ArrayList<>(rollInContentSequence.getAttributeSubSequence(index)), rollInContentSequence.getSequence().get(index).getAttrValuesBeforeThisPointInContentSequence(), rollInContentSequence.getSequence().get(index).getAttrValuesAfterThisPointInContentSequence(), availableAttributeActions, di.getMeaningRepresentation());
            if (in.getCorrectLabels().size() > 1) {
                /*
                System.out.println("AMBIGUITY: Multiple correct actions in content sequence");
                System.out.println(di.getMeaningRepresentation().getAbstractMR());
                System.out.println(rollInContentSequence);
                System.out.println(index + " >> " + rollInContentSequence.getSequence().get(index));
                System.out.println(in.getCorrectLabels());
                System.out.println(di.getTrainAttrRealization());
                 */
 /*printDebugInfo = true;
                for (String availableActionStr : in.getCorrectLabels()) {
                    String value = SFX.chooseNextValue(availableActionStr, rollInContentSequence.getSequence().get(index).getAttrValuesAfterThisPointInContentSequence());
                    ActionSequence modifiedSequence = new ActionSequence(rollInContentSequence);
                    if (!availableActionStr.equals(Action.TOKEN_END)) {
                        modifiedSequence.modifyAndShortenSequence(index, new Action(Action.TOKEN_START, availableActionStr + "=" + value, ""));
                    } else {
                        modifiedSequence.modifyAndShortenSequence(index, new Action(Action.TOKEN_END, availableActionStr, ""));
                    }

                    //ROLL-OUT
                    //System.out.println(availableActionStr.trim().toLowerCase() + " >> " + getPolicyRollOutCost_Content(predicate, modifiedSequence, di, trainedAttrClassifier_i.get(predicate), trainedWordClassifiers_i.get(predicate), valueAlignments, useReferenceRollout, trainingData, availableWordActions.get(predicate)));
                }
                printDebugInfo = false;*/
            }
            if (in.getCorrectLabels().isEmpty()) {
                System.out.println("NO COR");
                System.out.println(predicate);
                System.out.println(costs);
                System.exit(0);
            }
            if (trainedAttrClassifier_i.get(predicate).isInstanceLeadingToFix(in)) {
                earlyStop = true;
                //index = rollInContentSequence.getSequence().size() + 1;

                if (!in.getCorrectLabels().isEmpty()) {
                    String fixedAttr = new ArrayList<>(in.getCorrectLabels()).get(0);
                    String fixedValue = datasetParser.chooseNextValue(fixedAttr, rollInContentSequence.getSequence().get(index).getAttrValuesAfterThisPointInContentSequence());

                    if (fixedAttr.equals(Action.TOKEN_END)) {
                        Action fixedAction = new Action(Action.TOKEN_END, fixedAttr);
                        fixedAction.setAttrValueTracking(rollInContentSequence.getSequence().get(index).getAttrValuesBeforeThisPointInContentSequence(), rollInContentSequence.getSequence().get(index).getAttrValuesAfterThisPointInContentSequence());
                    } else {
                        Action fixedAction = new Action(Action.TOKEN_START, fixedAttr + "=" + fixedValue);
                        fixedAction.setAttrValueTracking(rollInContentSequence.getSequence().get(index).getAttrValuesBeforeThisPointInContentSequence(), rollInContentSequence.getSequence().get(index).getAttrValuesAfterThisPointInContentSequence());
                    }
                }
            }
            if (earlyStop) {
                if (earlyStopSteps >= JLOLS.earlyStopMaxFurtherSteps) {
                    index = rollInContentSequence.getSequence().size() + 1;
                } else {
                    earlyStopSteps++;
                }
            }
            newAttrTrainingInstances.get(di).get(predicate).add(in);
        }
        avgContentErrors += earlyStopSteps;

        //IF CONTENT SEQUENCE HAD ERRORS REPLACE CONTENT SEQUENCE ROLLIN
        boolean fixedRollInContent = false;
        if (earlyStop) {
            rollInContentSequence = new ActionSequence(di.getTrainAttrRealization());
            fixedRollInContent = true;

            earlyStop = false;
            earlyStopSteps = 0;
        }

        ActionSequence rollInWordSequence = getLearnedPolicyRollIn_Word(predicate, di, rollInContentSequence, trainedWordClassifiers_i.get(predicate), valueAlignments, availableWordActions.get(predicate));

        //if (earlyStopSteps == 0) {
        HashSet<String> encounteredXValues = new HashSet<>();
        for (int index = 0; index < rollInWordSequence.getSequence().size(); index++) {
            //Make the same decisions for all action substitutions
            boolean useReferenceRollout = false;
            double v = datasetParser.randomGen.nextDouble();
            if (v < beta) {
                useReferenceRollout = true;
            }

            //calculateSubReferences(di, new ArrayList<>(rollInWordSequence.getSequence().subList(0, index)));
            TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

            String attrValue = rollInWordSequence.getSequence().get(index).getAttribute();
            String attr = attrValue.substring(0, attrValue.indexOf('='));
            HashSet<String> wrongActions = new HashSet<>();
            if (trainedWordClassifiers_i.get(predicate).containsKey(attr)) {
                //FOR EACH POSSIBLE ALTERNATIVE ACTION
                for (Action action : availableWordActions.get(predicate).get(attr)) {
                    costs.put(action.getAction(), 1.0);
                }
                costs.put(Action.TOKEN_END.toLowerCase().trim(), 1.0);

                //If currentAttr's dictionary doesn't contain the best ref action
                /*if (useReferenceRollout && !availableWordActions.get(predicate).get(attr).contains(new Action(di.getRealizationCorrectAction(di.getTrainRealization()), attrValue, ""))) {
                        costs.put(Action.TOKEN_END.toLowerCase().trim(), 0.0);
                    } else {*/
                String value = datasetParser.chooseNextValue(attr, rollInWordSequence.getSequence().get(index).getAttrValuesAfterThisPointInContentSequence());
                for (Action availableAction : availableWordActions.get(predicate).get(attr)) {
                    availableAction.setAttribute(attrValue);

                    boolean eligibleWord = true;
                    if (availableAction.getWord().trim().toLowerCase().startsWith(Action.TOKEN_X)) {
                        if (value.isEmpty()) {
                            eligibleWord = false;
                        } else if (value.equals("no")
                                || value.equals("yes")
                                || value.equals("yes or no")
                                || value.equals("none")
                                || value.equals("empty") //|| value.equals("dont_care")
                                ) {
                            eligibleWord = false;
                        } else if (encounteredXValues.contains(availableAction.getWord().trim().toLowerCase())) {
                            eligibleWord = false;
                        } else {
                            int xIndex = Integer.parseInt(availableAction.getWord().trim().toLowerCase().substring(availableAction.getWord().trim().toLowerCase().lastIndexOf("_") + 1));
                            for (int x = 0; x < xIndex; x++) {
                                if (!encounteredXValues.contains(Action.TOKEN_X + attr + "_" + x)) {
                                    eligibleWord = false;
                                }
                            }
                        }
                    }
                    if (availableAction.getWord().equals(Action.TOKEN_END) && index == 0) {
                        eligibleWord = false;
                    } else if (index > 0
                            && availableAction.getWord().equals(Action.TOKEN_END)
                            && rollInWordSequence.getSequence().get(index - 1).getWord().equals(Action.TOKEN_END)) {
                        eligibleWord = false;
                    }
                    if (eligibleWord && !availableAction.getWord().equals(Action.TOKEN_END)) {
                        ArrayList<Action> cleanActSeq = new ArrayList<>();
                        for (Action a : rollInWordSequence.getSequence().subList(0, index)) {
                            if (!a.getWord().equals(Action.TOKEN_START)
                                    && !a.getWord().equals(Action.TOKEN_END)) {
                                cleanActSeq.add(a);
                            }
                        }
                        cleanActSeq.add(availableAction);
                        for (int j = 1; j <= Math.floor(cleanActSeq.size() / 2); j++) {
                            String followingStr = " " + (new ActionSequence(new ArrayList<>(cleanActSeq.subList(cleanActSeq.size() - j, cleanActSeq.size())))).getWordSequenceToNoPunctString().trim();
                            String previousStr = " " + (new ActionSequence(new ArrayList<>(cleanActSeq.subList(0, cleanActSeq.size() - j)))).getWordSequenceToNoPunctString().trim();

                            if (previousStr.endsWith(followingStr)) {
                                eligibleWord = false;
                            }
                        }
                    }
                    if (eligibleWord) {
                        ActionSequence modifiedSequence = new ActionSequence(rollInWordSequence);
                        modifiedSequence.modifyAndShortenSequence(index, new Action(availableAction));

                        //ROLL-OUT
                        if (modifiedSequence.getSequence().get(index).getWord().equals(",")) {
                            System.out.println("WTF: " + rollInWordSequence.getSequence());
                            System.exit(0);
                        }
                        costs.put(availableAction.getAction(), getPolicyRollOutCost_Words(predicate, modifiedSequence, di, rollInContentSequence, trainedWordClassifiers_i.get(predicate), valueAlignments, useReferenceRollout, trainingData, availableWordActions.get(predicate)));
                    } else {
                        wrongActions.add(availableAction.getAction());
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
                Instance in = datasetParser.createWordInstanceWithCosts(predicate, attrValue, costs, rollInWordSequence.getSequence().get(index).getAttrValuesBeforeThisPointInWordSequence(), new ArrayList<Action>(rollInWordSequence.getSequence().subList(0, index)), rollInWordSequence.getSequence().get(index).getAttrValuesAfterThisPointInWordSequence(), rollInWordSequence.getSequence().get(index).getAttrValuesBeforeThisPointInContentSequence(), rollInWordSequence.getSequence().get(index).getAttrValuesAfterThisPointInContentSequence(), rollInWordSequence.getSequence().get(index).isValueMentionedAtThisPoint(), availableWordActions.get(predicate));

                if (in.getCorrectLabels().size() > 1) {
                    /*
                    System.out.println("AMBIGUITY: Multiple correct actions in word sequence");
                    System.out.println(rollInWordSequence);
                    System.out.println(index + " >> " + rollInWordSequence.getSequence().get(index));
                    System.out.println(in.getCorrectLabels());
                    System.out.println(useReferenceRollout); 
                    System.out.println(di.getTrainRealization());
                    System.out.println(rollInContentSequence);
                    System.out.println(di.getTrainAttrRealization());*/
 /*
                    printDebugInfo = true;
                    for (String availableActionStr : in.getCorrectLabels()) {
                        Action availableAction = new Action(availableActionStr, attrValue, "");

                        boolean eligibleWord = true;
                        if (availableAction.getWord().trim().toLowerCase().startsWith(Action.TOKEN_X)) {
                            if (value.isEmpty()) {
                                eligibleWord = false;
                            } else if (value.equals("no")
                                    || value.equals("yes")
                                    || value.equals("yes or no")
                                    || value.equals("none")
                                    || value.equals("empty") //|| value.equals("dont_care")
                                    ) {
                                eligibleWord = false;
                            } else if (encounteredXValues.contains(availableAction.getWord().trim().toLowerCase())) {
                                eligibleWord = false;
                            } else {
                                int xIndex = Integer.parseInt(availableAction.getWord().trim().toLowerCase().substring(availableAction.getWord().trim().toLowerCase().lastIndexOf("_") + 1));
                                for (int x = 0; x < xIndex; x++) {
                                    if (!encounteredXValues.contains(Action.TOKEN_X + attr + "_" + x)) {
                                        eligibleWord = false;
                                    }
                                }
                            }
                        }
                        if (availableAction.getWord().equals(Action.TOKEN_END) && index == 0) {
                            eligibleWord = false;
                        } else if (index > 0) {
                            if (availableAction.getWord().equals(Action.TOKEN_END)
                                    && rollInWordSequence.getSequence().get(index - 1).getWord().equals(Action.TOKEN_END)) {
                                eligibleWord = false;
                            }
                        }
                        if (eligibleWord && !availableAction.getWord().equals(Action.TOKEN_END)) {
                            ArrayList<Action> cleanActSeq = new ArrayList<>();
                            for (Action a : rollInWordSequence.getSequence().subList(0, index)) {
                                if (!a.getWord().equals(Action.TOKEN_START)
                                        && !a.getWord().equals(Action.TOKEN_END)) {
                                    cleanActSeq.add(a);
                                }
                            }
                            cleanActSeq.add(availableAction);
                            for (int j = 1; j <= Math.floor(cleanActSeq.size() / 2); j++) {
                                String followingStr = " " + (new ActionSequence(new ArrayList<>(cleanActSeq.subList(cleanActSeq.size() - j, cleanActSeq.size())))).getWordSequenceToNoPunctString().trim();
                                String previousStr = " " + (new ActionSequence(new ArrayList<>(cleanActSeq.subList(0, cleanActSeq.size() - j)))).getWordSequenceToNoPunctString().trim();

                                if (previousStr.endsWith(followingStr)) {
                                    eligibleWord = false;
                                }
                            }
                        }
                        if (eligibleWord) {
                            ActionSequence modSeq = new ActionSequence(rollInWordSequence);
                            modSeq.modifyAndShortenSequence(index, new Action(availableAction));

                            //ROLL-OUT
                            System.out.println(availableAction.getAction() + " >> " + getPolicyRollOutCost_Words(predicate, modSeq, di, rollInContentSequence, trainedWordClassifiers_i.get(predicate), valueAlignments, useReferenceRollout, trainingData, availableWordActions.get(predicate)));
                        } else {
                            System.out.println(availableAction.getAction() + " >> wrong action");
                        }
                    }
                    printDebugInfo = false;
                    System.exit(0);*/
                }
                newWordTrainingInstances.get(di).get(predicate).get(rollInWordSequence.getSequence().get(index).getAttribute().substring(0, rollInWordSequence.getSequence().get(index).getAttribute().indexOf('='))).add(in);
                if (trainedWordClassifiers_i.get(predicate).get(rollInWordSequence.getSequence().get(index).getAttribute().substring(0, rollInWordSequence.getSequence().get(index).getAttribute().indexOf('='))).isInstanceLeadingToFix(in)) {
                    earlyStop = true;
                }
                if (earlyStop) {
                    if (earlyStopSteps >= JLOLS.earlyStopMaxFurtherSteps
                            && index + 1 < di.getTrainRealization().size()) {
                        if (!fixedRollInContent) {
                            rollInContentSequence = new ActionSequence(di.getTrainAttrRealization());
                            fixedRollInContent = true;
                        }
                        ArrayList<Action> refRollInWordSequence = new ArrayList<>(di.getTrainRealization().subList(0, index + 1));

                        if (refRollInWordSequence.get(refRollInWordSequence.size() - 1).getAttribute().equals(Action.TOKEN_END)) {
                            index = rollInWordSequence.getSequence().size() + 1;
                        } else {
                            rollInWordSequence = generateWordSequence(predicate, di, rollInContentSequence, new ActionSequence(refRollInWordSequence), trainedWordClassifiers_i.get(predicate), valueAlignments, availableWordActions.get(predicate));

                            earlyStopSteps = 0;
                            earlyStop = false;
                        }
                    } else {
                        earlyStopSteps++;
                    }
                }
                //}
            }
            if (index < rollInWordSequence.getSequence().size()
                    && rollInWordSequence.getSequence().get(index).getWord().startsWith(Action.TOKEN_X)) {
                encounteredXValues.add(rollInWordSequence.getSequence().get(index).getWord());
            }
        }
        avgWordErrors += earlyStopSteps;
        //}
    }

    public Double getPolicyRollOutCost_Content(String predicate, ActionSequence rollInContentSequence, DatasetInstance di, JAROW classifierAttrs, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, boolean useReferenceRollout, ArrayList<DatasetInstance> trainingData, HashMap<String, HashSet<Action>> availableWordActions) {
        if (useReferenceRollout) {
            return getReferencePolicyRollOutCost_Content(rollInContentSequence, di, availableWordActions);
        } else {
            return getLearnedPolicyRollOutCost_Content(rollInContentSequence, predicate, di, classifierAttrs, classifierWords, valueAlignments, availableWordActions);
        }
    }

    public Double getPolicyRollOutCost_Words(String predicate, ActionSequence actSeq, DatasetInstance di, ActionSequence contentSequence, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, boolean useReferenceRollout, ArrayList<DatasetInstance> trainingData, HashMap<String, HashSet<Action>> availableWordActions) {
        if (useReferenceRollout) {
            return getReferencePolicyRollOutCost_Words(actSeq, di, availableWordActions);
        } else {
            return getLearnedPolicyRollOutCost_Words(actSeq, predicate, di, contentSequence, classifierWords, valueAlignments, availableWordActions);
        }
    }

    public Double getReferencePolicyRollOutCost_Content(ActionSequence rollInSeq, DatasetInstance di, HashMap<String, HashSet<Action>> availableWordActions) {
        HashMap<ActionSequence, ArrayList<Action>> refs = new HashMap<>();
        refs.put(new ActionSequence(di.getTrainAttrRealization(), false), di.getTrainAttrRealization());

        if (rollInSeq.getNoPunctLength() > 1 && rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 1).getWord().equals(rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 2).getWord())) {
            //Do not repeat the same word twice in a row
            return 1.0;
        } else {
            double minCost = 1.0;
            String key = "REF_CONTCOST|" + rollInSeq.getSequence().toString() + "|" + refs.toString();
            Double cost = JLOLS.costCache.get(key);
            if (cost != null) {
                return cost;
            }
            for (ActionSequence refSeq : refs.keySet()) {
                String currentAttr = rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 1).getAttribute();

                ArrayList<Action> rollOutList = new ArrayList<>(rollInSeq.getSequence());
                ArrayList<Action> refList = new ArrayList<>(refSeq.getSequence());

                if (rollOutList.size() < refList.size()) {
                    if (currentAttr.equals(Action.TOKEN_END)) {
                        while (rollOutList.size() != refList.size()) {
                            rollOutList.add(new Action("££", "££"));
                        }
                    } else {
                        rollOutList.addAll(refList.subList(rollInSeq.getSequence().size(), refList.size()));
                    }
                } else {
                    while (rollOutList.size() != refList.size()) {
                        refList.add(new Action("££", "££"));
                    }
                }

                String rollOut = new ActionSequence(rollOutList).getAttrSequenceToString().toLowerCase().trim();
                ActionSequence newRefSeq = new ActionSequence(refList, false);
                ArrayList<String> refWindows = new ArrayList<>();
                refWindows.add(newRefSeq.getAttrSequenceToString().toLowerCase().trim());

                Integer totalAttrValuesInRef = 0;
                Integer attrValuesInRefAndNotInRollIn = 0;
                for (Action attrValueAct : refList) {
                    if (!attrValueAct.getAttribute().equals(Action.TOKEN_END)) {
                        totalAttrValuesInRef++;

                        boolean containsAttrValue = false;
                        for (Action a : rollOutList) {
                            if (a.getAttribute().equals(attrValueAct.getAttribute())) {
                                containsAttrValue = true;
                                break;
                            }
                        }
                        if (!containsAttrValue) {
                            attrValuesInRefAndNotInRollIn++;
                        }
                    }
                }
                double coverage = attrValuesInRefAndNotInRollIn.doubleValue() / totalAttrValuesInRef.doubleValue();
                //System.out.println("ROLLOUT " + rollOut);
                //System.out.println("REFS " + refWindows);
                double refCost = ActionSequence.getCostMetric(rollOut, refWindows, coverage);
                //refCost = (refCost + (((double)rollOutSeq.getSequence().size())/((double)SFX.maxAttrRealizationSize)))/2.0;
                if (refCost < minCost) {
                    minCost = refCost;
                }
            }
            JLOLS.costCache.put(key, minCost);
            return minCost;
        }
    }

    public Double getReferencePolicyRollOutCost_Words(ActionSequence rollInSeq, DatasetInstance di, HashMap<String, HashSet<Action>> availableWordActions) {
        HashMap<ActionSequence, ArrayList<Action>> refs = new HashMap<>();
        refs.put(new ActionSequence(di.getTrainRealization(), true), di.getTrainRealization());

        if (rollInSeq.getNoPunctLength() > 1 && rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 1).getWord().equals(rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 2).getWord())) {
            //Do not repeat the same word twice in a row
            if (printDebugInfo) {
                System.out.println("Word repeating!");
            }
            return 1.0;
        } else {
            double minCost = 1.0;
            for (ActionSequence refSeq : refs.keySet()) {
                //If the end of the ATTR
                boolean resolved = false;
                if (rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 1).getWord().equals(Action.TOKEN_END)) {
                    if (rollInSeq.getNoPunctLength() < refSeq.getSequence().size()) {
                        //If last action is the end of the attr and ref says we should continue with the same attr
                        //In other words, we should not have ended expressing that attr!
                        if (rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 1).getAttribute().equals(refSeq.getSequence().get(rollInSeq.getNoPunctLength()).getAttribute())
                                && 1.0 <= minCost) {
                            if (printDebugInfo) {
                                System.out.println("Went over attr allignment!");
                            }
                            minCost = 1.0;
                            resolved = true;

                        }
                    } else {
                        if (printDebugInfo) {
                            System.out.println("Ended before ref length!");
                        }
                        minCost = 0.0;
                        resolved = true;
                    }
                }
                if (!resolved) {
                    double minRefCost = 1.0;
                    ArrayList<Action> minRollOutWordSeq = null;
                    /*if (rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 1).getWord().equals(di.getRealizationCorrectAction(refs.get(refSeq)))) {
                        refCost = 0.0;
                    } else {
                        refCost = ((double) JDAggerForSFX.rollOutWindowSize) / (JDAggerForSFX.rollOutWindowSize + JDAggerForSFX.rollOutWindowSize + 1.0);
                    }*/
                    ArrayList<String> refWindows = new ArrayList<>();
                    refWindows.add(di.getTrainReference());

                    String key = "REF_WORDCOST|" + rollInSeq.getSequence().toString() + "|" + refWindows.toString();
                    Double cost = JLOLS.costCache.get(key);
                    if (cost != null) {
                        return cost;
                    }
                    if (!refWindows.isEmpty()) {
                        String minRollOut = "";
                        String minRef = "";
                        for (int i = 1; i < di.getTrainRealization().size(); i++) {
                            ArrayList<Action> rollInSeqCopy = new ArrayList<>();
                            for (Action act : rollInSeq.getSequence()) {
                                rollInSeqCopy.add(new Action(act));
                            }
                            rollInSeqCopy.addAll(di.getTrainRealization().subList(i, di.getTrainRealization().size()));

                            String rollOut = datasetParser.postProcessWordSequence(di, rollInSeqCopy);
                            //System.out.println("ROLLOUT " + rollOut);
                            //System.out.println("REFS " + refWindows);
                            double refCost = ActionSequence.getROUGE(rollOut, refWindows);

                            if (printDebugInfo) {
                                System.out.println("---");
                                System.out.println(refCost);
                                System.out.println(rollOut);
                                System.out.println(refWindows.get(0));
                            }
                            if (refCost < minRefCost) {
                                minRefCost = refCost;
                                minRollOut = rollOut;
                                minRef = refWindows.get(0);
                                minRollOutWordSeq = rollInSeqCopy;
                            }
                        }
                        if (printDebugInfo) {
                            System.out.println("MIN");
                            System.out.println(minRefCost);
                            System.out.println(minRollOut);
                            System.out.println(minRef);
                        }
                    }

                    if (printDebugInfo) {
                        System.out.println("Roll-in action " + rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 1).getWord() + " against ref action: " + di.getRealizationCorrectAction(refs.get(refSeq)));
                    }

                    if (minRollOutWordSeq != null) {
                        minRefCost = costBalance1 * minRefCost + costBalance2 * (((double) minRollOutWordSeq.size()) / ((double) datasetParser.maxWordRealizationSize));
                    } else {
                        minRefCost = 1.0;
                    }
                    if (minRefCost < minCost) {
                        minCost = minRefCost;
                    }
                    JLOLS.costCache.put(key, minCost);
                }
            }
            return minCost;
        }
    }

    public ActionSequence getLearnedPolicyRollIn_Content(String predicate, DatasetInstance di, JAROW classifierAttrs) {
        return generateContentSequence(predicate, di, new ActionSequence(), classifierAttrs);
    }

    public double getLearnedPolicyRollOutCost_Content(ActionSequence rollInSeq, String predicate, DatasetInstance di, JAROW classifierAttrs, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashSet<Action>> availableWordActions) {
        Action examinedAction = rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 1);

        if (rollInSeq.getNoPunctLength() > 1 && examinedAction.getWord().equals(rollInSeq.getSequence().get(rollInSeq.getSequence().size() - 2).getWord())) {
            //Do not repeat the same word twice in a row
            return 1.0;
        } else {
            ActionSequence rollOutContentSequence = generateContentSequence(predicate, di, rollInSeq, classifierAttrs);
            ActionSequence rollOutWordSeq = generateWordSequence(predicate, di, rollOutContentSequence, new ActionSequence(), classifierWords, valueAlignments, availableWordActions);

            String rollOut = datasetParser.postProcessWordSequence(di, rollOutWordSeq.getSequence());
            ArrayList<String> refWindows = new ArrayList<>();
            refWindows.add(di.getTrainReference());

            String key = "LEARNED_CONTCOST|" + rollOut + "|" + refWindows.toString();
            Double cost = JLOLS.costCache.get(key);
            if (cost != null) {
                return cost;
            }
            //System.out.println("ROLLOUT " + rollOut);
            //System.out.println("REFS " + refWindows);
            double refCost = 1.0;
            if (!refWindows.isEmpty()) {
                int attrValuesStillToBeMentionedSize = 0;
                for (String attrValue : examinedAction.getAttrValuesAfterThisPointInContentSequence()) {
                    if (!attrValue.endsWith("placetoeat")) {
                        attrValuesStillToBeMentionedSize++;
                    }
                }

                refCost = ActionSequence.getCostMetric(rollOut, refWindows, ((double) attrValuesStillToBeMentionedSize) / ((double) (attrValuesStillToBeMentionedSize + examinedAction.getAttrValuesBeforeThisPointInContentSequence().size())));
            }
            JLOLS.costCache.put(key, refCost);

            //refCost = (refCost + (((double)rollOutWordSeq.getSequence().size())/((double)SFX.maxWordRealizationSize)))/2.0;
            /*System.out.println(rollInSeq.getSequence());
            System.out.println(rollOutContentList);
            System.out.println(rollOutWordSeq.getWordSequenceToString());
            System.out.println(refWindows);
            System.out.println(refCost);
            System.out.println("====");*/
            return refCost;
        }
    }

    public ActionSequence getLearnedPolicyRollIn_Word(String predicate, DatasetInstance di, ActionSequence contentSequence, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashSet<Action>> availableWordActions) {
        return generateWordSequence(predicate, di, contentSequence, new ActionSequence(), classifierWords, valueAlignments, availableWordActions);
    }

    public double getLearnedPolicyRollOutCost_Words(ActionSequence rollInWordSeq, String predicate, DatasetInstance di, ActionSequence contentSequence, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashSet<Action>> availableWordActions) {
        Action examinedAction = rollInWordSeq.getSequence().get(rollInWordSeq.getSequence().size() - 1);

        if (rollInWordSeq.getNoPunctLength() > 1 && examinedAction.getWord().equals(rollInWordSeq.getSequence().get(rollInWordSeq.getSequence().size() - 2).getWord())) {
            //Do not repeat the same word twice in a row
            if (printDebugInfo) {
                System.out.println("Word repeating!");
            }
            return 1.0;
        } else {
            ActionSequence rollOutWordSeq = generateWordSequence(predicate, di, contentSequence, rollInWordSeq, classifierWords, valueAlignments, availableWordActions);
            String rollOut = datasetParser.postProcessWordSequence(di, rollOutWordSeq.getSequence());
            //If the end of an ATTR
            //If last action is the end of the attr and rollout says we should continue with the same attr
            //In other words, we should not have ended expressing that attr!
            if (examinedAction.getWord().equals(Action.TOKEN_END)
                    && rollInWordSeq.getSequence().size() < rollOutWordSeq.getSequence().size()
                    && !rollOutWordSeq.getSequence().isEmpty()
                    && examinedAction.getAttribute().equals(rollOutWordSeq.getSequence().get(rollInWordSeq.getSequence().size()).getAttribute())) {
                if (printDebugInfo) {
                    System.out.println("Going over attr alignment!");
                }
                return 100000000000000000.0;

            }

            ArrayList<String> refWindows = new ArrayList<>();
            refWindows.add(di.getTrainReference());

            String key = "LEARNED_WORDCOST|" + rollOut + "|" + refWindows.toString();
            Double cost = JLOLS.costCache.get(key);
            if (cost != null) {
                double refCost = 1.0;
                if (!refWindows.isEmpty()) {
                    refCost = ActionSequence.getCostMetric(rollOut, refWindows, -1.0);
                }
                if (printDebugInfo) {
                    System.out.println("Roll-out actions: " + rollOutWordSeq + " " + cost + " " + refCost);
                }
                return cost;
            }
            //System.out.println("ROLLOUT " + rollOut);
            //System.out.println("REFS " + refWindows);
            double refCost = 1.0;
            if (!refWindows.isEmpty()) {
                refCost = ActionSequence.getCostMetric(rollOut, refWindows, -1.0);
            }
            if (printDebugInfo) {
                System.out.println("Roll-out actions: " + rollOutWordSeq + " " + refCost);
            }

            refCost = costBalance1 * refCost + costBalance2 * (((double) rollOutWordSeq.getSequence().size()) / ((double) datasetParser.maxWordRealizationSize));
            JLOLS.costCache.put(key, refCost);
            /*System.out.println(rollInSeq.getSequence());
            System.out.println(predictedActionsList);
            System.out.println(rollOutWordSeq.getWordSequenceToString());
            System.out.println(refWindows);
            System.out.println(refCost);
            System.out.println("====");*/
            return refCost;
        }
    }

    private ActionSequence generateContentSequence(String predicate, DatasetInstance di, ActionSequence partialContentSequence, JAROW classifierAttrs) {
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
        ActionSequence cachedSequence = JLOLS.wordSequenceCache.get(key);
        if (cachedSequence != null) {
            //System.out.println("Retrieve from cache!!! " + key);
            return cachedSequence;
        }

        while (!predictedAttr.equals(Action.TOKEN_END) && predictedAttrValues.size() < datasetParser.maxAttrRealizationSize) {
            if (contentSequence.size() < partialContentSequence.getSequence().size()) {
                predictedAttr = partialContentSequence.getSequence().get(contentSequence.size()).getAttribute();
                predictedAttrValues.add(predictedAttr);

                if (!predictedAttr.equals(Action.TOKEN_END)) {
                    contentSequence.add(new Action(Action.TOKEN_START, predictedAttr));
                } else {
                    contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END));
                }
                contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned);

                if (!predictedAttr.isEmpty()) {
                    attrValuesAlreadyMentioned.add(predictedAttr);
                    attrValuesToBeMentioned.remove(predictedAttr);
                }
            } else if (!attrValuesToBeMentioned.isEmpty()) {
                Instance attrTrainingVector = datasetParser.createAttrInstance(predicate, "@TOK@", predictedAttrValues, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), availableAttributeActions);

                if (attrTrainingVector != null) {
                    Prediction predictAttr = classifierAttrs.predict(attrTrainingVector);
                    predictedAttr = predictAttr.getLabel().trim();
                    String predictedValue = "";
                    if (!predictedAttr.equals(Action.TOKEN_END)) {
                        predictedValue = datasetParser.chooseNextValue(predictedAttr, attrValuesToBeMentioned);

                        HashSet<String> rejectedAttrs = new HashSet<>();

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
                                predictedValue = datasetParser.chooseNextValue(predictedAttr, attrValuesToBeMentioned);
                            }
                        }
                        predictedAttr += "=" + predictedValue;
                    }
                    predictedAttrValues.add(predictedAttr);

                    if (!predictedAttr.equals(Action.TOKEN_END)) {
                        String attribute = predictedAttr.split("=")[0];

                        if (!attribute.equals(Action.TOKEN_END)) {
                            contentSequence.add(new Action(Action.TOKEN_START, predictedAttr));
                        } else {
                            contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END));
                        }
                        contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned);
                    } else {
                        contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END));
                        contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned);
                    }
                    if (!predictedAttr.isEmpty()) {
                        attrValuesAlreadyMentioned.add(predictedAttr);
                        attrValuesToBeMentioned.remove(predictedAttr);
                    }
                } else {
                    predictedAttr = Action.TOKEN_END;
                    contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END));
                    contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned);
                }
            } else {
                predictedAttr = Action.TOKEN_END;
                contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END));
                contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned);
            }
        }
        if (!contentSequence.get(contentSequence.size() - 1).getAttribute().equals(Action.TOKEN_END)) {
            //System.out.println("ATTR ROLL-IN IS UNENDING");
            //System.out.println(contentSequence);
            contentSequence.add(new Action(Action.TOKEN_END, Action.TOKEN_END));
            contentSequence.get(contentSequence.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned);
        }
        cachedSequence = new ActionSequence(contentSequence);
        JLOLS.wordSequenceCache.put(key, cachedSequence);
        return cachedSequence;
    }

    private ActionSequence generateWordSequence(String predicate, DatasetInstance di, ActionSequence contentSequence, ActionSequence partialWordSequence, HashMap<String, JAROW> classifierWords, HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments, HashMap<String, HashSet<Action>> availableWordActions) {
        ArrayList<Action> predictedActionsList = new ArrayList<>();
        ArrayList<Action> predictedWordList = new ArrayList<>();

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
        ActionSequence cachedSequence = JLOLS.wordSequenceCache.get(key);
        if (cachedSequence != null) {
            //System.out.println("Retrieve from cache!!! " + key);
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

                        boolean isValueMentioned = false;
                        String valueTBM = "";
                        if (attrValue.contains("=")) {
                            valueTBM = attrValue.substring(attrValue.indexOf('=') + 1);
                        }
                        if (valueTBM.isEmpty()) {
                            isValueMentioned = true;
                        }
                        ArrayList<String> subPhrase = new ArrayList<>();
                        while (!predictedWord.equals(Action.TOKEN_END) && predictedWordList.size() < datasetParser.maxWordRealizationSize) {
                            if (predictedActionsList.size() < partialWordSequence.getSequence().size()) {
                                predictedWord = partialWordSequence.getSequence().get(predictedActionsList.size()).getWord();
                                predictedActionsList.add(new Action(predictedWord, attrValue));
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
                                Instance wordTrainingVector = datasetParser.createWordInstance(predicate, new Action("@TOK@", attrValue), predictedAttributesForInstance, predictedActionsList, nextAttributesForInstance, attrValuesAlreadyMentioned, attrValuesToBeMentioned, isValueMentioned, availableWordActions);

                                if (wordTrainingVector != null) {
                                    if (classifierWords.get(attribute) != null) {
                                        Prediction predictWord = classifierWords.get(attribute).predict(wordTrainingVector);

                                        if (predictWord.getLabel() != null) {
                                            predictedWord = predictWord.getLabel().trim();
                                            while (predictedWord.equals(Action.TOKEN_END) && !predictedActionsList.isEmpty() && predictedActionsList.get(predictedActionsList.size() - 1).getWord().equals(Action.TOKEN_END)) {
                                                double maxScore = -Double.MAX_VALUE;
                                                for (String word : predictWord.getLabel2Score().keySet()) {
                                                    if (!word.equals(Action.TOKEN_END)
                                                            && (Double.compare(predictWord.getLabel2Score().get(word), maxScore) > 0)) {
                                                        maxScore = predictWord.getLabel2Score().get(word);
                                                        predictedWord = word;
                                                    }
                                                }
                                            }
                                        } else {
                                            predictedWord = Action.TOKEN_END;
                                        }
                                        predictedActionsList.add(new Action(predictedWord, attrValue));
                                        predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned, predictedAttributes, nextAttributesForInstance, isValueMentioned);
                                        if (!predictedWord.equals(Action.TOKEN_START)
                                                && !predictedWord.equals(Action.TOKEN_END)) {
                                            subPhrase.add(predictedWord);
                                            predictedWordList.add(new Action(predictedWord, attrValue));
                                        }
                                    } else {
                                        predictedWord = Action.TOKEN_END;
                                        predictedActionsList.add(new Action(predictedWord, attrValue));
                                        predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned, predictedAttributes, nextAttributesForInstance, isValueMentioned);
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
                                        String valueToCheck = valueTBM;
                                        if (valueToCheck.equals("no")
                                                || valueToCheck.equals("yes")
                                                || valueToCheck.equals("yes or no")
                                                || valueToCheck.equals("none")
                                                || valueToCheck.equals("empty") //|| valueToCheck.equals("dont_care")
                                                ) {
                                            valueToCheck = attribute + ":" + valueTBM;
                                        }
                                        if (!valueToCheck.equals("empty:empty")
                                                && valueAlignments.containsKey(valueToCheck)) {
                                            for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                                if (datasetParser.endsWith(subPhrase, alignedStr)) {
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
                            if (!predictedWord.startsWith(Action.TOKEN_X)) {
                                for (String attrValueTBM : attrValuesToBeMentioned) {
                                    if (attrValueTBM.contains("=")) {
                                        String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                        if (!(value.matches("\"[xX][0-9]+\"")
                                                || value.matches("[xX][0-9]+")
                                                || value.startsWith(Action.TOKEN_X))) {
                                            String valueToCheck = value;
                                            if (valueToCheck.equals("no")
                                                    || valueToCheck.equals("yes")
                                                    || valueToCheck.equals("yes or no")
                                                    || valueToCheck.equals("none")
                                                    || valueToCheck.equals("empty") //|| valueToCheck.equals("dont_care")
                                                    ) {
                                                valueToCheck = attrValueTBM.replace("=", ":");
                                            }
                                            if (!valueToCheck.equals("empty:empty")
                                                    && valueAlignments.containsKey(valueToCheck)) {
                                                for (ArrayList<String> alignedStr : valueAlignments.get(valueToCheck).keySet()) {
                                                    if (datasetParser.endsWith(subPhrase, alignedStr)) {
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
                        if (predictedWordList.size() >= datasetParser.maxWordRealizationSize
                                && !predictedActionsList.get(predictedActionsList.size() - 1).getWord().equals(Action.TOKEN_END)) {
                            predictedWord = Action.TOKEN_END;
                            predictedActionsList.add(new Action(predictedWord, attrValue));
                            predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned, predictedAttributes, nextAttributesForInstance, isValueMentioned);
                        }
                    } else {
                        predictedActionsList.add(new Action(Action.TOKEN_END, attrValue));
                        predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned, predictedAttributes, nextAttributesForInstance, true);
                    }
                } else {
                    predictedActionsList.add(new Action(Action.TOKEN_END, Action.TOKEN_END));
                    predictedActionsList.get(predictedActionsList.size() - 1).setAttrValueTracking(attrValuesAlreadyMentioned, attrValuesToBeMentioned, predictedAttributes, nextAttributesForInstance, true);
                }
            }
        }

        cachedSequence = new ActionSequence(predictedActionsList);
        JLOLS.wordSequenceCache.put(key, cachedSequence);
        return cachedSequence;
    }
}
