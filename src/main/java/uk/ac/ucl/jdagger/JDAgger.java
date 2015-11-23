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
import uk.ac.ucl.jarow.DatasetInstance;
import uk.ac.ucl.jarow.Instance;
import uk.ac.ucl.jarow.JAROW;
import uk.ac.ucl.jarow.MeaningRepresentation;
import uk.ac.ucl.jarow.Prediction;
import uk.ac.ucl.jarow.RoboCup;
import static uk.ac.ucl.jarow.RoboCup.createUnbiasedReferenceList;
import static uk.ac.ucl.jarow.RoboCup.createWordInstance;

public class JDAgger {

    final static int threadsCount = Runtime.getRuntime().availableProcessors() * 2;
    public static int ep = 0;
    public static boolean train = false;

    public JDAgger() {
    }
    public static Random r = new Random();
    public static HashMap<ActionSequence, ActionSequence> rollOutCache = new HashMap<>();

    public JAROW runVDAggerOnBagel(String predicate, ArrayList<DatasetInstance> trainingDatasetInstances, ArrayList<Action> availableActions, int epochs, double beta) {
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
            for (DatasetInstance datasetInstance : trainingDatasetInstances) {
                MeaningRepresentation meaningRepr = datasetInstance.getMeaningRepresentation();

                double v = r.nextDouble();
                boolean useReferenceRollIn = false;
                if (v <= p) {
                    useReferenceRollIn = true;
                }
                ArrayList<ArrayList<Action>> refs = new ArrayList<>(datasetInstance.getRealizations());
                int randomRef = new Random().nextInt(refs.size());

                ArrayList<Action> as = new ArrayList<>();
                for (Action s : refs.get(randomRef)) {
                    as.add(s);
                }
                as.add(new Action(Bagel.TOKEN_END, ""));
                ActionSequence ref = new ActionSequence(as, 0.0);

                //ROLL-IN
                ActionSequence actSeq = getPolicyRollIn(predicate, meaningRepr, classifierWords, useReferenceRollIn, ref);

                //FOR EVERY ACTION IN THE SEQUENCE
                for (int a = 0; a < actSeq.getSequence().size(); a++) {
                    if (a < ref.getSequence().size() + 2) {
                        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                        for (Action action : availableActions) {
                            costs.put(action.getWord(), 1.0);
                        }

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
                //System.out.println("|-> " + modActSeq.getSequenceToString());
                //System.out.println("C " + modActSeq.getCost());
            }
            Collections.shuffle(newTrainingInstances);
            //if (classifierWords == null) {
            classifierWords = trainClassifier(newTrainingInstances);
            /*} else {
            classifierWords.trainAdditional(newTrainingWordInstances, true, true, 10, 0.1, true);
            }*/
            long endTime = System.currentTimeMillis();
            long totalTime = endTime - startTime;
            SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss");
            Date resultdate = new Date(endTime);

            System.out.println("Epoch after: " + totalTime / 1000 / 60 + " mins, " + sdf.format(resultdate));
        }
        //multiTrainClassifier(newTrainingInstances);
        return classifierWords;
    }

    public JAROW runDAgger(String predicate, ArrayList<Instance> trainingWordInstances, ArrayList<MeaningRepresentation> meaningReprs, ArrayList<Action> availableActions, HashMap<ActionSequence, Integer> referencePolicy, HashMap<MeaningRepresentation, ArrayList<String>> oneRefPatterns, int epochs, double beta) {
        ArrayList<ActionSequence> referencePolicyKeyList = new ArrayList(referencePolicy.keySet());
        Collections.sort(referencePolicyKeyList);
        JAROW classifierWords = null;//trainClassifier(trainingWordInstances);
        ArrayList<Instance> newTrainingWordInstances = new ArrayList();
        for (int i = 1; i <= epochs; i++) {
            ep = i;
            rollOutCache = new HashMap<>();
            System.out.println("Starting epoch " + i);
            long startTime = System.currentTimeMillis();

            double p = Math.pow(1.0 - beta, (double) i - 1);
            System.out.println("p = " + p);

            //ArrayList<Instance> newTrainingWordInstances = new ArrayList();
            //CHANGE
            for (MeaningRepresentation meaningRepr : meaningReprs) {
                double v = r.nextDouble();
                boolean useReferenceRollIn = false;
                if (v <= p) {
                    useReferenceRollIn = true;
                }

                ArrayList<Action> as = new ArrayList<>();
                for (String s : oneRefPatterns.get(meaningRepr)) {
                    as.add(new Action(s, ""));
                }
                ActionSequence ref = new ActionSequence(as, 0.0);
                ref.getSequence().add(new Action(RoboCup.TOKEN_END, ""));

                //ROLL-IN
                ActionSequence actSeq = getPolicyRollIn(predicate, meaningRepr, classifierWords, useReferenceRollIn, ref);

                //FOR EVERY ACTION IN THE SEQUENCE
                for (int a = 0; a < actSeq.getSequence().size(); a++) {
                    if (a < ref.getSequence().size() + 2) {
                        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                        for (Action action : availableActions) {
                            costs.put(action.getWord(), 1.0);
                        }

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
                        rollOutSeq.modifyAndShortenSequence(a, RoboCup.TOKEN_END);
                        rollOutSeq.getSequence().remove(rollOutSeq.getSequence().size() - 1);

                        train = true;
                        newTrainingWordInstances.add(generateTrainingInstance(predicate, meaningRepr, availableActions, rollOutSeq, a, costs));
                        train = false;
                        //System.exit(0);
                    }
                }

                //System.out.println("|-> " + modActSeq.getSequenceToString());
                //System.out.println("C " + modActSeq.getCost());
            }
            Collections.shuffle(newTrainingWordInstances);
            //if (classifierWords == null) {
            classifierWords = trainClassifier(newTrainingWordInstances);
            /*} else {
            classifierWords.trainAdditional(newTrainingWordInstances, true, true, 10, 0.1, true);
            }*/
            long endTime = System.currentTimeMillis();
            long totalTime = endTime - startTime;
            SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss");
            Date resultdate = new Date(endTime);
            System.out.println("Epoch after: " + totalTime / 1000 / 60 + " mins, " + sdf.format(resultdate));
        }
        multiTrainClassifier(newTrainingWordInstances);
        return classifierWords;
    }

    public static JAROW runStochasticDAgger(String predicate, ArrayList<Instance> trainingWordInstances, ArrayList<MeaningRepresentation> meaningReprs, ArrayList<Action> availableActions, HashMap<ActionSequence, Integer> referencePolicy, HashMap<MeaningRepresentation, ArrayList<String>> oneRefPatterns, int epochs, double beta) {
        ArrayList<String> unbiasedRefList = createUnbiasedReferenceList(predicate);

        ArrayList<ActionSequence> referencePolicyKeyList = new ArrayList(referencePolicy.keySet());
        Collections.sort(referencePolicyKeyList);

        JAROW classifierWords = trainClassifier(trainingWordInstances);
        for (int i = 1; i <= epochs; i++) {
            System.out.println("Starting epoch " + i);
            double p = Math.pow(1.0 - beta, (double) i - 1);
            System.out.println("p = " + p);

            ArrayList<Instance> newTrainingWordInstances = new ArrayList();
            for (MeaningRepresentation meaningRepr : meaningReprs) {
                double v = r.nextDouble();
                boolean useReferenceRollIn = false;
                if (v <= p) {
                    useReferenceRollIn = true;
                }

                ArrayList<Action> as = new ArrayList<>();
                for (String s : oneRefPatterns.get(meaningRepr)) {
                    as.add(new Action(s, ""));
                }
                ActionSequence ref = new ActionSequence(as, 0.0);

                //ROLL-IN
                ActionSequence actSeq = getPolicyRollIn(predicate, meaningRepr, classifierWords, useReferenceRollIn, ref);

                //CHOSE A RANDOM ACTION IN THE SEQUENCE AND MODIFY IT TO ANOTHER RANDOM ACTION
                //It could be made to be less random, like selecting those actions that are more liable to lead to mistakes
                //System.out.println(actSeq.getSequenceToString());
                int index = r.nextInt(actSeq.getSequence().size());
                int wordIndex = r.nextInt(availableActions.size());
                while (availableActions.get(wordIndex).getWord().equals(actSeq.getSequence().get(index).getWord())) {
                    wordIndex = r.nextInt(availableActions.size());
                }
                //System.out.println("->| " + actSeq.getSequenceToString());
                actSeq.modifyAndShortenSequence(index, availableActions.get(wordIndex).getWord());
                //System.out.println("M " + actSeq.getSequenceToString());

                //ROLL-OUT
                ActionSequence rollOutSeq = getReferencePolicyRollOut(actSeq, referencePolicyKeyList, referencePolicy, unbiasedRefList, ref);

                //GENERATE NEW TRAINING EXAMPLE
                newTrainingWordInstances.add(generateTrainingInstance(predicate, meaningRepr, availableActions, rollOutSeq, index));

                //System.out.println("|-> " + modActSeq.getSequenceToString());
                //System.out.println("C " + modActSeq.getCost());
                //System.out.println("====================================");
            }
            Collections.shuffle(newTrainingWordInstances);
            classifierWords.trainAdditional(newTrainingWordInstances, true, true, 10, 0.1, true);
        }
        return classifierWords;
    }

    public static JAROW runLOLS(String predicate, ArrayList<Instance> trainingWordInstances, ArrayList<MeaningRepresentation> meaningReprs, ArrayList<Action> availableActions, HashMap<ActionSequence, Integer> referencePolicy, int epochs, double beta, ActionSequence ref) {
        ArrayList<String> unbiasedRefList = createUnbiasedReferenceList(predicate);

        ArrayList<ActionSequence> referencePolicyKeyList = new ArrayList(referencePolicy.keySet());
        Collections.sort(referencePolicyKeyList);

        ArrayList<JAROW> trainedClassifiers = new ArrayList();
        //INITIALIZE A POLICY P_0 (initializing on ref)
        JAROW classifierWords = trainClassifier(trainingWordInstances);
        for (int i = 1; i <= epochs; i++) {
            for (MeaningRepresentation meaningRepr : meaningReprs) {
                trainedClassifiers.add(classifierWords);

                //Initialize new training set
                ArrayList<Instance> newTrainingInstances = new ArrayList();
                //ROLL-IN
                ActionSequence actSeq = getLearnedPolicyRollIn(predicate, meaningRepr, classifierWords, ref);

                //FOR EACH ACTION IN ROLL-IN SEQUENCE
                //The number of actions is not definite...might cause issues
                for (int index = 0; index < actSeq.getSequence().size(); index++) {
                    //FOR EACH POSSIBLE ALTERNATIVE ACTION
                    ActionSequence rollOutSeq = null;
                    TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                    for (Action action : availableActions) {
                        costs.put(action.getWord(), 1.0);
                    }

                    //Make the same decisions for all action substitutions
                    boolean useReferenceRollout = false;
                    double v = r.nextDouble();
                    if (v < beta) {
                        useReferenceRollout = true;
                    }

                    for (Action availableAction : availableActions) {
                        if (!availableAction.getWord().equals(actSeq.getSequence().get(index).getWord())) {
                            //System.out.println("->| " + actSeq.getSequenceToString());
                            actSeq.modifyAndShortenSequence(index, availableAction.getWord());
                            //System.out.println("M " + actSeq.getSequenceToString());
                            //ROLL-OUT
                            rollOutSeq = getPolicyRollOut(predicate, actSeq, referencePolicyKeyList, referencePolicy, unbiasedRefList, meaningRepr, classifierWords, useReferenceRollout, ref);
                            costs.put(availableAction.getWord(), rollOutSeq.getCost());
                        }
                    }

                    //GENERATE NEW TRAINING EXAMPLE
                    newTrainingInstances.add(generateTrainingInstance(predicate, meaningRepr, availableActions, rollOutSeq, index, costs));
                }

                //UPDATE CLASSIFIER
                Collections.shuffle(newTrainingInstances);
                classifierWords.trainAdditional(newTrainingInstances, true, true, 10, 0.1, true);
            }
        }

        //FIRST NEED TO AVERAGE OVER ALL CLASSIFIERS
        JAROW avgClassifierWords = new JAROW();
        avgClassifierWords.averageOverClassifiers(trainedClassifiers);

        return avgClassifierWords;
    }

    public static ActionSequence getPolicyRollIn(String predicate, MeaningRepresentation mr, JAROW classifierWords, double p, ActionSequence ref) {
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
    }

    public static ActionSequence getPolicyRollOut(String predicate, ActionSequence actSeq, ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, MeaningRepresentation mr, JAROW classifierWords, double p, ActionSequence ref) {
        double v = r.nextDouble();

        if (v <= p) {
            return getReferencePolicyRollOut(actSeq, referencePolicyKeyList, referencePolicy, unbiasedRefList, ref);
        } else {
            return getLearnedPolicyRollOut(predicate, actSeq, mr, classifierWords, referencePolicy, unbiasedRefList, ref);
        }
    }

    public static ActionSequence getPolicyRollOut(String predicate, ActionSequence actSeq, ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, MeaningRepresentation mr, JAROW classifierWords, boolean useReferenceRollout, ActionSequence ref) {
        if (useReferenceRollout) {
            return getReferencePolicyRollOut(actSeq, referencePolicyKeyList, referencePolicy, unbiasedRefList, ref);
        } else {
            return getLearnedPolicyRollOut(predicate, actSeq, mr, classifierWords, referencePolicy, unbiasedRefList, ref);
        }
    }

    public static ActionSequence getReferencePolicyRollIn(ActionSequence ref) {
        return new ActionSequence(ref);
    }

    public static ActionSequence getReferencePolicyRollOut(ActionSequence pAS, ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, ActionSequence ref) {
        ActionSequence mAS = null;

        double minDistance = Integer.MAX_VALUE;
        //HashSet<ActionSequence> allSeqs = new HashSet<>();

        //if (rollOutCache.containsKey(pAS)) {
//            return rollOutCache.get(pAS);
        //       } else {
        for (ActionSequence as : referencePolicyKeyList) {
            for (int i = as.getSequence().size() - 2; i >= 0; i--) {
                ActionSequence newAS = new ActionSequence(pAS);
                System.out.println(pAS.getWordSequenceToString() + "||| " + newAS.getWordSequenceToString());
                for (int j = 0; j < as.getSequence().size(); j++) {
                    if (j >= i) {
                        newAS.getSequence().add(new Action(as.getSequence().get(j).getWord(), ""));
                    }
                }
                System.out.println(pAS.getWordSequenceToString() + "||| " + newAS.getWordSequenceToString());
                HashSet<Action> toBeRemoved = new HashSet<>();
                for (Action a : newAS.getSequence()) {
                    if (a.getWord().equals(RoboCup.TOKEN_END)) {
                        toBeRemoved.add(a);
                    }
                }
                newAS.getSequence().removeAll(toBeRemoved);
                System.out.println(pAS.getWordSequenceToString() + "||| " + newAS.getWordSequenceToString());
                newAS.getSequence().add(new Action(RoboCup.TOKEN_END, ""));

                newAS.recalculateCost(ref);
                //System.out.println(newAS.getCost() + " -> " + newAS.getSequenceToString());  
                //allSeqs.add(newAS);
                if (newAS.getCost() < minDistance) {
                    minDistance = newAS.getCost();
                    mAS = newAS;
                    if (minDistance == 0.0) {
                        //                rollOutCache.put(pAS, mAS);
                        return mAS;
                    }
                } else if (newAS.getCost() > minDistance) {
                    i = -1;
                }
            }
        }

        // Doesn't really matter to choose the smallest (in length) roll-out sequence, since we do nothing with it other than use the score
            /*HashSet<ActionSequence> bestSeqs = new HashSet<>();
        for (ActionSequence newAS : allSeqs) {
        if (newAS.getCost() == minDistance) {
        bestSeqs.add(newAS);
        }
        }
        
        int minLength = Integer.MAX_VALUE;
        for (ActionSequence bestAS : bestSeqs) {
        if (bestAS.getSequence().size() <= minLength) {
        minLength = bestAS.getSequence().size();
        mAS = bestAS;
        }
        }*/
        //  rollOutCache.put(pAS, mAS);
        System.out.println(pAS.getWordSequenceToString() + " -> " + mAS.getWordSequenceToString());
        return mAS;
        //}
    }

    public static ActionSequence getLearnedPolicyRollIn(String predicate, MeaningRepresentation mr, JAROW classifierWords, ActionSequence ref) {
        String predictedWord = "";
        int w = 0;
        ArrayList<String> predictedWordsList = new ArrayList<>();

        HashMap<String, Boolean> argumentsToBeMentioned = new HashMap<>();
        for (String argument : mr.getAttributes().keySet()) {
            argumentsToBeMentioned.put(argument, true);
        }
        //System.out.println(">>>>>>>>>>>>>>>>>>>>>>>");
        while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10000) {
            ArrayList<String> tempList = new ArrayList(predictedWordsList);
            tempList.add("@TOK@");
            Instance trainingVector = createWordInstance(predicate, tempList, w, argumentsToBeMentioned);
            if (trainingVector != null) {
                Prediction predict = classifierWords.predict(trainingVector);
                //System.out.println(predict.getLabel() + " - " + predict.getScore());
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
        //System.out.println("<<<<<<<<<<<<<<<<<<<<<<<");

        ArrayList<Action> actionList = new ArrayList<>();
        for (String word : predictedWordsList) {
            actionList.add(new Action(word, ""));
        }
        return new ActionSequence(actionList, ref);
    }

    public static ActionSequence getLearnedPolicyRollOut(String predicate, ActionSequence pAS, MeaningRepresentation mr, JAROW classifierWords, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, ActionSequence ref) {
        String predictedWord = "";
        int w = 0;
        ArrayList<String> predictedWordsList = new ArrayList<>();
        HashMap<String, Boolean> argumentsToBeMentioned = new HashMap<>();
        for (String argument : mr.getAttributes().keySet()) {
            argumentsToBeMentioned.put(argument, true);
        }
        for (Action a : pAS.getSequence()) {
            predictedWord = a.getWord();

            predictedWordsList.add(predictedWord);
            for (String arg : argumentsToBeMentioned.keySet()) {
                if (predictedWord.equals(arg)) {
                    argumentsToBeMentioned.put(arg, false);
                }
            }
        }

        while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10000) {
            ArrayList<String> tempList = new ArrayList(predictedWordsList);
            tempList.add("@TOK@");
            Instance trainingVector = createWordInstance(predicate, tempList, w, argumentsToBeMentioned);
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

        ArrayList<Action> actionList = new ArrayList<>();
        for (String word : predictedWordsList) {
            actionList.add(new Action(word, ""));
        }
        return new ActionSequence(actionList, ref);
    }

    public static Instance generateTrainingInstance(String predicate, MeaningRepresentation meaningRepr, ArrayList<Action> availableActions, ActionSequence modActSeq, int index) {
        HashMap<String, Boolean> argumentsToBeMentioned = new HashMap<>();
        for (String argument : meaningRepr.getAttributes().keySet()) {
            argumentsToBeMentioned.put(argument, true);
        }
        ArrayList<String> predictedWordsList = new ArrayList<>();
        for (int i = 0; i <= index; i++) {
            String predictedWord = modActSeq.getSequence().get(i).getWord();

            predictedWordsList.add(predictedWord);
            for (String arg : argumentsToBeMentioned.keySet()) {
                if (predictedWord.equals(arg)) {
                    argumentsToBeMentioned.put(arg, false);
                }
            }
        }
        return createWordInstance(predicate, predictedWordsList, index, modActSeq.getCost(), argumentsToBeMentioned);
    }

    public static Instance generateTrainingInstance(String predicate, MeaningRepresentation meaningRepr, ArrayList<Action> availableActions, ActionSequence modActSeq, int index, TObjectDoubleHashMap<String> costs) {
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
        return createWordInstance(predicate, predictedWordsList, index, costs, argumentsToBeMentioned);
    }

    public static JAROW trainClassifier(ArrayList<Instance> trainingWordInstances) {
        //JAROW classifierWords = new JAROW();
        //classifierWords.train(trainingWordInstances, true, true, 10, 0.1, true);

        Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
        JAROW classifierWords = JAROW.trainOpt(trainingWordInstances, RoboCup.rounds, params, 0.1, true, false, 0);

        return classifierWords;
    }

    public static void multiTrainClassifier(ArrayList<Instance> trainingWordInstances) {
        System.out.println("Start Multi");

        for (int i = 0; i < 10; i++) {
            JAROW classifierWords = new JAROW();
            Collections.shuffle(trainingWordInstances);
            System.out.println(classifierWords.train(trainingWordInstances, true, true, RoboCup.rounds, 0.1, true));
        }

        System.out.println("End Multi");
    }

    class ReferenceRollOutThread extends Thread {

        ActionSequence actSeq;
        TObjectDoubleHashMap<String> costs;
        ActionSequence ref;
        int index;
        Action aAction;

        public ReferenceRollOutThread(ActionSequence actSeq, TObjectDoubleHashMap<String> costs, int index, Action aAction, ActionSequence ref) {
            this.actSeq = new ActionSequence(actSeq);
            this.costs = costs;
            this.ref = ref;

            this.index = index;
            this.aAction = aAction;
        }

        public void run() {
            //System.out.print("=-");
            actSeq.modifyAndShortenSequence(index, aAction.getWord());

            //ROLL-OUT
            ActionSequence rollOut = getSubOptimalReferencePolicyRollOut(actSeq, ref);
            costs.put(aAction.getWord(), rollOut.getCost());
            //System.out.println(aAction.getDecision() + " -> " + rollOut.getCost());
            //System.out.println("Rollout -> " + rollOut.getSequenceToString());
            //if (actSeq.getSequence().toString().equals("[A{@arg1@}, A{kicks}, A{passes}]")) {
            //if (actSeq.getSequence().size() == 6) {
            //if (ep > 2) {
            //System.out.println("SUB " + actSeq.getSequence() + " | " + " -> " + rollOut.getCost());
            //System.out.println("ROLL " + rollOut.getSequence() + " | "  + " -> " + rollOut.getCost());
            //    System.out.println("MAS " + ref.getSequence() + " |");
            //}
            //}
        }

        public ActionSequence getReferencePolicyRollOut(ActionSequence pAS, ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList) {
            ActionSequence mAS = null;

            //if (rollOutCache.containsKey(pAS)) {
            //    return rollOutCache.get(pAS);
            //} else {
            double minDistance = Integer.MAX_VALUE;
            for (ActionSequence as : referencePolicyKeyList) {
                //System.out.println("R > " + as.getSequenceToString());
                double previousCost = Integer.MAX_VALUE;
                if (as.getWordSequenceToString().startsWith(pAS.getWordSequenceToString())) {
                    mAS = new ActionSequence(as);
                    mAS.recalculateCost(ref);
                    return mAS;
                }
                for (int i = 0; i < as.getSequence().size() - 1; i++) {
                    ActionSequence newAS = new ActionSequence(pAS);
                    for (int j = i; j < as.getSequence().size(); j++) {
                        //if (j <= i) {
                        newAS.getSequence().add(new Action(as.getSequence().get(j).getWord(), ""));
                        //}
                    }
                    HashSet<Action> toBeRemoved = new HashSet<>();
                    for (Action a : newAS.getSequence()) {
                        if (a.getWord().equals(RoboCup.TOKEN_END)) {
                            toBeRemoved.add(a);
                        }
                    }
                    newAS.getSequence().removeAll(toBeRemoved);
                    newAS.getSequence().add(new Action(RoboCup.TOKEN_END, ""));

                    newAS.recalculateCost(ref);
                    //allSeqs.add(newAS);
                    if (newAS.getCost() < minDistance) {
                        minDistance = newAS.getCost();
                        mAS = newAS;
                        if (minDistance == 0.0) {
                            //rollOutCache.put(pAS, mAS);
                            return mAS;
                        }
                    } else if (newAS.getCost() > previousCost) {
                        i = as.getSequence().size();
                    }
                    previousCost = newAS.getCost();
                }
            }

            // Doesn't really matter to choose the smallest (in length) roll-out sequence, since we do nothing with it other than use the score
            /*HashSet<ActionSequence> bestSeqs = new HashSet<>();
            for (ActionSequence newAS : allSeqs) {
            if (newAS.getCost() == minDistance) {
            bestSeqs.add(newAS);
            }
            }
            
            int minLength = Integer.MAX_VALUE;
            for (ActionSequence bestAS : bestSeqs) {
            if (bestAS.getSequence().size() <= minLength) {
            minLength = bestAS.getSequence().size();
            mAS = bestAS;
            }
            }*/
            pAS.recalculateCost(ref);
            if (pAS.getCost() < minDistance) {
                //rollOutCache.put(pAS, mAS);
                return pAS;
            }
            //rollOutCache.put(pAS, mAS);
            return mAS;
            //}
            //}
        }

        public ActionSequence getSubOptimalReferencePolicyRollOut(ActionSequence pAS, ActionSequence ref) {
            ActionSequence mAS = null;

            //System.out.println(">>> " + pAS.getSequence());
            if (rollOutCache.containsKey(pAS)) {
                return rollOutCache.get(pAS);
            } else {
                pAS.recalculateCost(ref);
                /*double minDistance = Integer.MAX_VALUE;
                ActionSequence minRAS = null;
                for (ActionSequence as : referencePolicyKeyList) {
                double score = Levenshtein.getDistance(pAS.getSequenceToString(), as.getSequenceToString());
                if (score < minDistance) {
                minDistance = score;
                minRAS = as;
                if (minDistance == 0.0) {
                break;
                }
                }
                }*/
                //System.out.println("MIN " + minRAS.getSequenceToString());

                if (pAS.getSequence().size() > 1 && pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(pAS.getSequence().get(pAS.getSequence().size() - 2).getWord())) {
                    //Do not repeat the same word twice in a row
                    ActionSequence newAS = new ActionSequence(pAS);
                    newAS.setCost(1.0);
                    if (rollOutCache.keySet().size() <= 1000) {
                        rollOutCache.put(pAS, newAS);
                    }
                    //System.out.println("END " + pAS.getSequence() + " | " + pAS.getCost());
                    return newAS;
                } else if (pAS.getSequence().size() > ref.getSequence().size()) {
                    //Do not exceed (or plan to exceed) the length of the reference
                    //*** Perhaps this needs to be more elegant, too tired now
                    ActionSequence newAS = new ActionSequence(pAS);
                    if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(RoboCup.TOKEN_END)) {
                        newAS.recalculateCost(ref);
                    } else {
                        newAS.setCost(1.0);
                    }
                    if (rollOutCache.keySet().size() <= 1000) {
                        rollOutCache.put(pAS, newAS);
                    }
                    //System.out.println("END " + newAS.getSequence() + " | " + newAS.getCost());
                    return newAS;
                } else if (pAS.getSequence().get(pAS.getSequence().size() - 1).getWord().equals(RoboCup.TOKEN_END)) {
                    ActionSequence newAS = new ActionSequence(pAS);
                    newAS.recalculateCost(ref);
                    if (rollOutCache.keySet().size() <= 1000) {
                        rollOutCache.put(pAS, newAS);
                    }
                    //System.out.println("END " + pAS.getSequence() + " | " + pAS.getCost());
                    return newAS;
                } else {
                    ActionSequence minRAS = ref;

                    //Let;s assume for now that the only correct response is the particular sentence that corresponds to the meaning representation in the data
                    //Lets also assume that a good cost estimation is comparing the rollin (plus mutation) to the correct responce cut to the same length
                    //That is because we assume that all next words will be best and not make any impression to the score (reference rollout)
                    ActionSequence newAS = new ActionSequence();
                    for (int i = 0; i < pAS.getSequence().size(); i++) {
                        if (i < minRAS.getSequence().size()) {
                            newAS.getSequence().add(new Action(minRAS.getSequence().get(i).getWord(), ""));
                        }
                    }
                    newAS.setCost(Levenshtein.getNormDistance(pAS.getWordSequenceToString(), newAS.getWordSequenceToString(), minRAS.getSequence().size()));
                    if (rollOutCache.keySet().size() <= 1000) {
                        rollOutCache.put(pAS, newAS);
                    }
                    return newAS;

                    //MAYBE A NEW COST: HOW MANY POSITIONS IN THE ARRAY HAVE WRONG ACTIONS?
                    /*double minDistance = Integer.MAX_VALUE;
                    for (int i = 0; i < minRAS.getSequence().size(); i++) {
                    ActionSequence newAS = new ActionSequence(pAS);
                    for (int j = i; j < minRAS.getSequence().size(); j++) {
                    //if (j <= i) {
                    newAS.getSequence().add(new Action(minRAS.getSequence().get(j).getDecision()));
                    //}
                    }
                    HashSet<Action> toBeRemoved = new HashSet<>();
                    for (Action a : newAS.getSequence()) {
                    if (a.getDecision().equals(RoboCup.TOKEN_END)) {
                    toBeRemoved.add(a);
                    }
                    }
                    newAS.getSequence().removeAll(toBeRemoved);
                    newAS.getSequence().add(new Action(RoboCup.TOKEN_END));
                    
                    newAS.recalculateCost(ref);
                    if (actSeq.getSequence().toString().equals("[A{@arg1@}, A{kicks}, A{passes}]")) {
                    System.out.println("NEW " + newAS.getSequenceToString() + " SCORE " + newAS.getCost());
                    }
                    
                    if (newAS.getCost() < minDistance) {
                    minDistance = newAS.getCost();
                    mAS = newAS;
                    if (minDistance == 0.0) {
                    if (rollOutCache.keySet().size() <= 30000) {
                    rollOutCache.put(pAS, mAS);
                    }
                    return mAS;
                    }
                    }
                    }
                    
                    if (pAS.getCost() < minDistance) {
                    if (rollOutCache.keySet().size() <= 30000) {
                    rollOutCache.put(pAS, mAS);
                    }
                    return pAS;
                    }
                    
                    if (rollOutCache.keySet().size() <= 30000) {
                    rollOutCache.put(pAS, mAS);
                    }
                    return mAS;*/
                }
                //}
            }
        }
    }
}
