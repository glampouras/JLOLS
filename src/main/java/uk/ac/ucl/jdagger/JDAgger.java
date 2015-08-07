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
import uk.ac.ucl.jarow.Instance;
import uk.ac.ucl.jarow.JAROW;
import uk.ac.ucl.jarow.MeaningRepresentation;
import uk.ac.ucl.jarow.Prediction;
import uk.ac.ucl.jarow.RoboCup;
import static uk.ac.ucl.jarow.RoboCup.createUnbiasedReferenceList;
import static uk.ac.ucl.jarow.RoboCup.createWordInstance;

public class JDAgger {

    final static int threadsCount = Runtime.getRuntime().availableProcessors() * 2;
    
    static int ep = 0;

    public JDAgger() {
    }

    public static Random r = new Random();
    public static HashMap<ActionSequence, ActionSequence> rollOutCache = new HashMap<>();

    public JAROW runDAgger(String predicate, ArrayList<Instance> trainingWordInstances, ArrayList<MeaningRepresentation> meaningReprs, ArrayList<Action> availableActions, HashMap<ActionSequence, Integer> referencePolicy, HashMap <MeaningRepresentation, ArrayList<String>> oneRefPatterns, int epochs, double beta) {
        ArrayList<String> unbiasedRefList = createUnbiasedReferenceList(predicate);

        ArrayList<ActionSequence> referencePolicyKeyList = new ArrayList(referencePolicy.keySet());
        Collections.sort(referencePolicyKeyList);
        JAROW classifierWords = null;//trainClassifier(trainingWordInstances);
        for (int i = 1; i <= epochs; i++) {
            ep = i;
            rollOutCache = new HashMap<>();
            System.out.println("Starting epoch " + i);
            long startTime = System.currentTimeMillis();
            
            double p = Math.pow(1.0 - beta, (double) i - 1);
            System.out.println("p = " + p);

            double v = r.nextDouble();
            boolean useReferenceRollIn = false;
            if (v <= p) {
                useReferenceRollIn = true;
            }

            ArrayList<Instance> newTrainingWordInstances = new ArrayList();
            //CHANGE
            for (MeaningRepresentation meaningRepr : meaningReprs) {
                ArrayList<Action> as = new ArrayList<>();
                for (String s : oneRefPatterns.get(meaningRepr)) {
                    as.add(new Action(s));
                }
                ActionSequence ref = new ActionSequence(as, 0.0);
                ref.getSequence().add(new Action(RoboCup.TOKEN_END));
                
                //ROLL-IN
                ActionSequence actSeq = getPolicyRollIn(referencePolicyKeyList, referencePolicy, unbiasedRefList, meaningRepr, classifierWords, useReferenceRollIn, ref);
                
                //FOR EVERY ACTION IN THE SEQUENCE
                for (int a = 0; a < actSeq.getSequence().size(); a++) {
                    TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                    availableActions.stream().forEach((action) -> {
                        costs.put(action.getDecision(), 1.0);
                    });

                    //MODIFY IT TO EACH POSSIBLE AVAILABLE ACTION           
                    ExecutorService executor = Executors.newFixedThreadPool(threadsCount);
                    for (Action availableAction : availableActions) {
                        executor.execute(new ReferenceRollOutThread(actSeq, costs, a, availableAction, referencePolicyKeyList, referencePolicy, unbiasedRefList, ref));
                    }
                    executor.shutdown();
                    while (!executor.isTerminated()) {
                    }
                    //GENERATE NEW TRAINING EXAMPLE
                    ActionSequence rollOutSeq = new ActionSequence(actSeq);
                    rollOutSeq.modifyAndShortenSequence(a, RoboCup.TOKEN_END);
                    rollOutSeq.getSequence().remove(rollOutSeq.getSequence().size() - 1);

                    newTrainingWordInstances.add(generateTrainingInstance(meaningRepr, availableActions, rollOutSeq, a, costs));
                    //System.exit(0);
                }

                //System.out.println("|-> " + modActSeq.getSequenceToString());
                //System.out.println("C " + modActSeq.getCost());
            }
            Collections.shuffle(newTrainingWordInstances);
            if (classifierWords == null) {
                classifierWords = trainClassifier(newTrainingWordInstances);
            } else {
                classifierWords.trainAdditional(newTrainingWordInstances, true, true, 10, 0.1, true);
            }
            long endTime = System.currentTimeMillis();
            long totalTime = endTime - startTime;
            SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss");
            Date resultdate = new Date(endTime);
            System.out.println("Epoch after: " + totalTime / 1000 / 60+ " mins, " + sdf.format(resultdate));
        }
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

            double v = r.nextDouble();
            boolean useReferenceRollIn = false;
            if (v <= p) {
                useReferenceRollIn = true;
            }

            ArrayList<Instance> newTrainingWordInstances = new ArrayList();
            for (MeaningRepresentation meaningRepr : meaningReprs) {
                ArrayList<Action> as = new ArrayList<Action>();
                for (String s : oneRefPatterns.get(meaningRepr)) {
                    as.add(new Action(s));
                }
                ActionSequence ref = new ActionSequence(as, 0.0);
                
                //ROLL-IN
                ActionSequence actSeq = getPolicyRollIn(referencePolicyKeyList, referencePolicy, unbiasedRefList, meaningRepr, classifierWords, useReferenceRollIn, ref);

                //CHOSE A RANDOM ACTION IN THE SEQUENCE AND MODIFY IT TO ANOTHER RANDOM ACTION
                //It could be made to be less random, like selecting those actions that are more liable to lead to mistakes
                //System.out.println(actSeq.getSequenceToString());
                int index = r.nextInt(actSeq.getSequence().size());
                int wordIndex = r.nextInt(availableActions.size());
                while (availableActions.get(wordIndex).getDecision().equals(actSeq.getSequence().get(index).getDecision())) {
                    wordIndex = r.nextInt(availableActions.size());
                }
                //System.out.println("->| " + actSeq.getSequenceToString());
                actSeq.modifyAndShortenSequence(index, availableActions.get(wordIndex).getDecision());
                //System.out.println("M " + actSeq.getSequenceToString());

                //ROLL-OUT
                ActionSequence rollOutSeq = getReferencePolicyRollOut(actSeq, referencePolicyKeyList, referencePolicy, unbiasedRefList, ref);

                //GENERATE NEW TRAINING EXAMPLE
                newTrainingWordInstances.add(generateTrainingInstance(meaningRepr, availableActions, rollOutSeq, index));

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
                ActionSequence actSeq = getLearnedPolicyRollIn(meaningRepr, classifierWords, unbiasedRefList, ref);

                //FOR EACH ACTION IN ROLL-IN SEQUENCE
                //The number of actions is not definite...might cause issues
                for (int index = 0; index < actSeq.getSequence().size(); index++) {
                    //FOR EACH POSSIBLE ALTERNATIVE ACTION
                    ActionSequence rollOutSeq = null;
                    TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                    availableActions.stream().forEach((action) -> {
                        costs.put(action.getDecision(), 1.0);
                    });

                    //Make the same decisions for all action substitutions
                    boolean useReferenceRollout = false;
                    double v = r.nextDouble();
                    if (v < beta) {
                        useReferenceRollout = true;
                    }

                    for (Action availableAction : availableActions) {
                        if (!availableAction.getDecision().equals(actSeq.getSequence().get(index).getDecision())) {
                            //System.out.println("->| " + actSeq.getSequenceToString());
                            actSeq.modifyAndShortenSequence(index, availableAction.getDecision());
                            //System.out.println("M " + actSeq.getSequenceToString());
                            //ROLL-OUT
                            rollOutSeq = getPolicyRollOut(actSeq, referencePolicyKeyList, referencePolicy, unbiasedRefList, meaningRepr, classifierWords, useReferenceRollout, ref);
                            costs.put(availableAction.getDecision(), rollOutSeq.getCost());
                        }
                    }

                    //GENERATE NEW TRAINING EXAMPLE
                    newTrainingInstances.add(generateTrainingInstance(meaningRepr, availableActions, rollOutSeq, index, costs));
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

    public static ActionSequence getPolicyRollIn(ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, MeaningRepresentation mr, JAROW classifierWords, double p, ActionSequence ref) {
        double v = r.nextDouble();

        if (v <= p) {
            return getReferencePolicyRollIn(referencePolicyKeyList, referencePolicy, ref);
        } else {
            return getLearnedPolicyRollIn(mr, classifierWords, unbiasedRefList, ref);
        }
    }

    public static ActionSequence getPolicyRollIn(ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, MeaningRepresentation mr, JAROW classifierWords, boolean useReferenceRollin, ActionSequence ref) {
        if (useReferenceRollin) {
            return getReferencePolicyRollIn(referencePolicyKeyList, referencePolicy, ref);
        } else {
            return getLearnedPolicyRollIn(mr, classifierWords, unbiasedRefList, ref);
        }
    }

    public static ActionSequence getPolicyRollOut(ActionSequence actSeq, ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, MeaningRepresentation mr, JAROW classifierWords, double p, ActionSequence ref) {
        double v = r.nextDouble();

        if (v <= p) {
            return getReferencePolicyRollOut(actSeq, referencePolicyKeyList, referencePolicy, unbiasedRefList, ref);
        } else {
            return getLearnedPolicyRollOut(actSeq, mr, classifierWords, referencePolicy, unbiasedRefList, ref);
        }
    }

    public static ActionSequence getPolicyRollOut(ActionSequence actSeq, ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, MeaningRepresentation mr, JAROW classifierWords, boolean useReferenceRollout, ActionSequence ref) {
        if (useReferenceRollout) {
            return getReferencePolicyRollOut(actSeq, referencePolicyKeyList, referencePolicy, unbiasedRefList, ref);
        } else {
            return getLearnedPolicyRollOut(actSeq, mr, classifierWords, referencePolicy, unbiasedRefList, ref);
        }
    }

    public static ActionSequence getReferencePolicyRollIn(ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ActionSequence ref) {
        /*ActionSequence mAS = null;

        int max = -1;
        for (ActionSequence as : referencePolicyKeyList) {
            if (referencePolicy.get(as) > max) {
                max = referencePolicy.get(as);
                mAS = as;
            }
        }*/

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
                System.out.println(pAS.getSequenceToString() + "||| " + newAS.getSequenceToString());
                for (int j = 0; j < as.getSequence().size(); j++) {
                    if (j >= i) {
                        newAS.getSequence().add(new Action(as.getSequence().get(j).getDecision()));
                    }
                }
                System.out.println(pAS.getSequenceToString() + "||| " + newAS.getSequenceToString());
                HashSet<Action> toBeRemoved = new HashSet<>();
                for (Action a : newAS.getSequence()) {
                    if (a.getDecision().equals(RoboCup.TOKEN_END)) {
                        toBeRemoved.add(a);
                    }
                }
                newAS.getSequence().removeAll(toBeRemoved);
                System.out.println(pAS.getSequenceToString() + "||| " + newAS.getSequenceToString());
                newAS.getSequence().add(new Action(RoboCup.TOKEN_END));

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
        System.out.println(pAS.getSequenceToString() + " -> " + mAS.getSequenceToString());
        return mAS;
        //}
    }

    public static ActionSequence getLearnedPolicyRollIn(MeaningRepresentation mr, JAROW classifierWords, ArrayList<String> unbiasedRefList, ActionSequence ref) {
        String predictedWord = "";
        int w = 0;
        ArrayList<String> predictedWordsList = new ArrayList<>();
        boolean arg1toBeMentioned = false;
        boolean arg2toBeMentioned = false;
        for (String argument : mr.getArguments()) {
            if (mr.getArguments().indexOf(argument) == 0) {
                arg1toBeMentioned = true;
            } else if (mr.getArguments().indexOf(argument) == 1) {
                arg2toBeMentioned = true;
            }
        }
        while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10000) {
            ArrayList<String> tempList = new ArrayList(predictedWordsList);
            tempList.add("@TOK@");
            Instance trainingVector = createWordInstance(tempList, w, arg1toBeMentioned, arg2toBeMentioned);
            if (trainingVector != null) {
                Prediction predict = classifierWords.predict(trainingVector);
                predictedWord = predict.getLabel().trim();
                predictedWordsList.add(predictedWord);

                if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                    arg1toBeMentioned = false;
                } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                    arg2toBeMentioned = false;
                }
            }
            w++;
        }

        ArrayList<Action> actionList = new ArrayList<>();
        for (String word : predictedWordsList) {
            actionList.add(new Action(word));
        }
        return new ActionSequence(actionList, ref);
    }

    public static ActionSequence getLearnedPolicyRollOut(ActionSequence pAS, MeaningRepresentation mr, JAROW classifierWords, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, ActionSequence ref) {
        String predictedWord = "";
        int w = 0;
        ArrayList<String> predictedWordsList = new ArrayList<>();
        boolean arg1toBeMentioned = false;
        boolean arg2toBeMentioned = false;
        for (String argument : mr.getArguments()) {
            if (mr.getArguments().indexOf(argument) == 0) {
                arg1toBeMentioned = true;
            } else if (mr.getArguments().indexOf(argument) == 1) {
                arg2toBeMentioned = true;
            }
        }
        for (Action a : pAS.getSequence()) {
            predictedWord = a.getDecision();

            predictedWordsList.add(predictedWord);
            if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                arg1toBeMentioned = false;
            } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                arg2toBeMentioned = false;
            }
        }

        while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10000) {
            ArrayList<String> tempList = new ArrayList(predictedWordsList);
            tempList.add("@TOK@");
            Instance trainingVector = createWordInstance(tempList, w, arg1toBeMentioned, arg2toBeMentioned);
            if (trainingVector != null) {
                Prediction predict = classifierWords.predict(trainingVector);
                predictedWord = predict.getLabel().trim();
                predictedWordsList.add(predictedWord);

                if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                    arg1toBeMentioned = false;
                } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                    arg2toBeMentioned = false;
                }
            }
            w++;
        }

        ArrayList<Action> actionList = new ArrayList<>();
        for (String word : predictedWordsList) {
            actionList.add(new Action(word));
        }
        return new ActionSequence(actionList, ref);
    }

    public static Instance generateTrainingInstance(MeaningRepresentation meaningRepr, ArrayList<Action> availableActions, ActionSequence modActSeq, int index) {
        boolean arg1toBeMentioned = false;
        boolean arg2toBeMentioned = false;
        for (String argument : meaningRepr.getArguments()) {
            if (meaningRepr.getArguments().indexOf(argument) == 0) {
                arg1toBeMentioned = true;
            } else if (meaningRepr.getArguments().indexOf(argument) == 1) {
                arg2toBeMentioned = true;
            }
        }
        ArrayList<String> predictedWordsList = new ArrayList<>();
        for (int i = 0; i <= index; i++) {
            String predictedWord = modActSeq.getSequence().get(i).getDecision();

            predictedWordsList.add(predictedWord);
            if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                arg1toBeMentioned = false;
            } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                arg2toBeMentioned = false;
            }
        }
        return createWordInstance(predictedWordsList, index, modActSeq.getCost(), arg1toBeMentioned, arg2toBeMentioned);
    }

    public static Instance generateTrainingInstance(MeaningRepresentation meaningRepr, ArrayList<Action> availableActions, ActionSequence modActSeq, int index, TObjectDoubleHashMap<String> costs) {
        boolean arg1toBeMentioned = false;
        boolean arg2toBeMentioned = false;
        for (String argument : meaningRepr.getArguments()) {
            if (meaningRepr.getArguments().indexOf(argument) == 0) {
                arg1toBeMentioned = true;
            } else if (meaningRepr.getArguments().indexOf(argument) == 1) {
                arg2toBeMentioned = true;
            }
        }
        ArrayList<String> predictedWordsList = new ArrayList<>();
        for (int i = 0; i <= index - 1; i++) {
            String predictedWord = modActSeq.getSequence().get(i).getDecision();

            predictedWordsList.add(predictedWord);
            if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                arg1toBeMentioned = false;
            } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                arg2toBeMentioned = false;
            }
        }
        return createWordInstance(predictedWordsList, index, costs, arg1toBeMentioned, arg2toBeMentioned);
    }

    public static JAROW trainClassifier(ArrayList<Instance> trainingWordInstances) {
        JAROW classifierWords = new JAROW();
        classifierWords.train(trainingWordInstances, true, true, 10, 0.1, true);
        return classifierWords;
    }

    class ReferenceRollOutThread extends Thread {

        ActionSequence actSeq;
        TObjectDoubleHashMap<String> costs;
        ArrayList<ActionSequence> referencePolicyKeyList;
        HashMap<ActionSequence, Integer> referencePolicy;
        ArrayList<String> unbiasedRefList;
        ActionSequence ref;
        int a;
        Action aAction;

        public ReferenceRollOutThread(ActionSequence actSeq, TObjectDoubleHashMap<String> costs, int a, Action aAction, ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, ActionSequence ref) {
            this.actSeq = new ActionSequence(actSeq);
            this.costs = costs;
            this.referencePolicyKeyList = referencePolicyKeyList;
            this.referencePolicy = referencePolicy;
            this.unbiasedRefList = unbiasedRefList;
            this.ref = ref;           

            this.a = a;
            this.aAction = aAction;
        }

        public void run() {
            //System.out.print("=-");
            actSeq.modifyAndShortenSequence(a, aAction.getDecision());

            //ROLL-OUT
            ActionSequence rollOut = getSubOptimalReferencePolicyRollOut(actSeq, referencePolicyKeyList, referencePolicy, unbiasedRefList, ref);
            costs.put(aAction.getDecision(), rollOut.getCost());
            //System.out.println(aAction.getDecision() + " -> " + rollOut.getCost());
            //System.out.println("Rollout -> " + rollOut.getSequenceToString());
            //if (actSeq.getSequence().toString().equals("[A{@arg1@}, A{kicks}, A{passes}]")) {
            //if (actSeq.getSequence().size() == 6) {
            if (ep == 5) {
                System.out.println("SUB " + actSeq.getSequence() + " | "  + " -> " + rollOut.getCost());
                //System.out.println("ROLL " + rollOut.getSequence() + " | "  + " -> " + rollOut.getCost());
            //    System.out.println("MAS " + ref.getSequence() + " |");
            }
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
                    if (as.getSequenceToString().startsWith(pAS.getSequenceToString())) {
                        mAS = new ActionSequence(as);
                        mAS.recalculateCost(ref);
                        return mAS;
                    }
                    for (int i = 0; i < as.getSequence().size() - 1; i++) {
                        ActionSequence newAS = new ActionSequence(pAS);
                        for (int j = i; j < as.getSequence().size(); j++) {
                            //if (j <= i) {
                            newAS.getSequence().add(new Action(as.getSequence().get(j).getDecision()));
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

        public ActionSequence getSubOptimalReferencePolicyRollOut(ActionSequence pAS, ArrayList<ActionSequence> referencePolicyKeyList, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, ActionSequence ref) {
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
                
                if (pAS.getSequence().size() > 1 && pAS.getSequence().get(pAS.getSequence().size() - 1).getDecision().equals(pAS.getSequence().get(pAS.getSequence().size() - 2).getDecision())) {
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
                    if (pAS.getSequence().get(pAS.getSequence().size() - 1).getDecision().equals(RoboCup.TOKEN_END)) {
                        newAS.recalculateCost(ref);
                    } else {
                        newAS.setCost(1.0);
                    }
                    if (rollOutCache.keySet().size() <= 1000) {
                        rollOutCache.put(pAS, newAS);
                    }
                    //System.out.println("END " + newAS.getSequence() + " | " + newAS.getCost());
                    return newAS;
                } else if (pAS.getSequence().get(pAS.getSequence().size() - 1).getDecision().equals(RoboCup.TOKEN_END)) {            
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
                            newAS.getSequence().add(new Action(minRAS.getSequence().get(i).getDecision()));
                        }
                    }
                    newAS.setCost(Levenshtein.getNormDistance(pAS.getSequenceToString(), newAS.getSequenceToString(), minRAS.getSequence().size()));
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
