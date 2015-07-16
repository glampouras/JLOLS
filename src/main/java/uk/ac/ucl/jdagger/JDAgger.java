/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uk.ac.ucl.jdagger.jarow;

import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import uk.ac.ucl.jdagger.jarow.Action;
import uk.ac.ucl.jdagger.jarow.ActionSequence;
import uk.ac.ucl.jdagger.jarow.Instance;
import uk.ac.ucl.jdagger.jarow.JAROW;
import uk.ac.ucl.jdagger.jarow.MeaningRepresentation;
import uk.ac.ucl.jdagger.jarow.Prediction;
import uk.ac.ucl.jdagger.jarow.RoboCup;
import static uk.ac.ucl.jdagger.jarow.RoboCup.createUnbiasedReferenceList;
import static uk.ac.ucl.jdagger.jarow.RoboCup.createWordTrainingVectorChoice;
import static uk.ac.ucl.jdagger.jarow.RoboCup.getReferencePolicy;

/**
 *
 * @author localadmin
 */
public class JDAgger {           
    public static Random r = new Random();
    
    public static JAROW runStochasticDAgger(ArrayList<Instance> trainingWordInstances, ArrayList<MeaningRepresentation> meaningReprs, ArrayList<Action> availableActions, HashMap<ActionSequence, Integer> referencePolicy, int epochs, double beta) {
        ArrayList<String> unbiasedRefList = createUnbiasedReferenceList();

        JAROW classifierWords = trainClassifier(trainingWordInstances);
        for (int i = 1; i <= epochs; i++) {
            double p = Math.pow(1.0 - beta, (double) i - 1);
            System.out.println("p = " + p);

            for (MeaningRepresentation meaningRepr : meaningReprs) {
                //ROLL-IN
                ActionSequence actSeq = getPolicyRollIn(referencePolicy, unbiasedRefList, meaningRepr, classifierWords, p);

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
                ActionSequence modActSeq = getReferencePolicyRollOut(actSeq, referencePolicy, unbiasedRefList);

                //GENERATE NEW TRAINING EXAMPLE
                trainingWordInstances.add(generateTrainingInstance(meaningRepr, availableActions, modActSeq, index));

                //System.out.println("|-> " + modActSeq.getSequenceToString());
                //System.out.println("C " + modActSeq.getCost());
                //System.out.println("====================================");
            }
            classifierWords = trainClassifier(trainingWordInstances);
        }
        return classifierWords;
    }

    public static ActionSequence getPolicyRollIn(HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList, MeaningRepresentation mr, JAROW classifierWords, double p) {
        double v = r.nextDouble();

        if (v <= p) {
            return getReferencePolicyRollIn(referencePolicy);
        } else {
            return getLearnedPolicyRollIn(mr, classifierWords, unbiasedRefList);
        }
    }

    public static ActionSequence getReferencePolicyRollIn(HashMap<ActionSequence, Integer> referencePolicy) {
        ActionSequence mAS = null;

        int max = -1;
        for (ActionSequence as : referencePolicy.keySet()) {
            if (referencePolicy.get(as) > max) {
                max = referencePolicy.get(as);
                mAS = as;
            }
        }

        return new ActionSequence(mAS);
    }

    public static ActionSequence getReferencePolicyRollOut(ActionSequence pAS, HashMap<ActionSequence, Integer> referencePolicy, ArrayList<String> unbiasedRefList) {
        ActionSequence mAS = null;

        double minDistance = Integer.MAX_VALUE;
        HashSet<ActionSequence> allSeqs = new HashSet<>();
        for (ActionSequence as : referencePolicy.keySet()) {
            for (int i = as.getSequence().size() - 2; i >= 0; i--) {
                ActionSequence newAS = new ActionSequence(pAS);
                for (int j = 0; j < as.getSequence().size(); j++) {
                    if (j >= i) {
                        newAS.getSequence().add(new Action(as.getSequence().get(j).getDecision()));
                    }
                }
                HashSet<Action> toBeRemoved = new HashSet<>();
                for (Action a : newAS.getSequence()) {
                    if (a.getDecision().equals(RoboCup.TOKEN_END)) {
                        toBeRemoved.add(a);
                    }
                }
                newAS.getSequence().removeAll(toBeRemoved);
                newAS.getSequence().add(new Action(RoboCup.TOKEN_END));

                newAS.recalculateCost(unbiasedRefList);
                //System.out.println(newAS.getCost() + " -> " + newAS.getSequenceToString());  
                allSeqs.add(newAS);
                if (newAS.getCost() < minDistance) {
                    minDistance = newAS.getCost();
                } else if (newAS.getCost() > minDistance) {
                    i = -1;
                }
            }
        }

        HashSet<ActionSequence> bestSeqs = new HashSet<>();
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
        }

        return mAS;
    }

    public static ActionSequence getLearnedPolicyRollIn(MeaningRepresentation mr, JAROW classifierWords, ArrayList<String> unbiasedRefList) {
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
            String trainingVector = createWordTrainingVectorChoice(tempList, w, arg1toBeMentioned, arg2toBeMentioned);

            //System.out.println("TV " + trainingVector);
            if (!trainingVector.isEmpty()) {
                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                String[] details;
                details = trainingVector.split(" ");

                for (int j = 1; j < details.length; j++) {
                    String[] feature;
                    feature = details[j].split(":");

                    featureVector.put(feature[0], Double.parseDouble(feature[1]));
                }

                Prediction predict = classifierWords.predict(new Instance(featureVector, costs));
                predictedWord = predict.getLabel().trim();
                predictedWordsList.add(predictedWord);

                if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                    arg1toBeMentioned = false;
                } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                    arg2toBeMentioned = false;
                }

                //System.out.println(trainingVector);
                //System.out.println("T: " + nlWords[w] + " P: " + predict.getLabel());
            }
            w++;
        }

        ArrayList<Action> actionList = new ArrayList<>();
        for (String word : predictedWordsList) {
            actionList.add(new Action(word));
        }
        return new ActionSequence(actionList, unbiasedRefList);
    }

    /*public static ActionSequence getLearnedPolicyRollOut(ActionSequence pAS, MeaningRepresentation mr, JAROW classifierWords, HashMap<ActionSequence, Integer> referencePolicy, double p) {
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
     } else if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
     arg2toBeMentioned = false;
     }
     }

     while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10000) {
     ArrayList<String> tempList = new ArrayList(predictedWordsList);
     tempList.add("@TOK@");
     String trainingVector = createWordTrainingVectorChoice(tempList, w, arg1toBeMentioned, arg2toBeMentioned);

     //System.out.println("TV " + trainingVector);
     if (!trainingVector.isEmpty()) {
     TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
     TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

     String[] details;
     details = trainingVector.split(" ");

     for (int j = 1; j < details.length; j++) {
     String[] feature;
     feature = details[j].split(":");

     featureVector.put(feature[0], Double.parseDouble(feature[1]));
     }

     Prediction predict = classifierWords.predict(new Instance(featureVector, costs));
     predictedWord = predict.getLabel().trim();
     predictedWordsList.add(predictedWord);

     if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
     arg1toBeMentioned = false;
     } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
     arg2toBeMentioned = false;
     }

     //System.out.println(trainingVector);
     //System.out.println("T: " + nlWords[w] + " P: " + predict.getLabel());
     }
     w++;
     }

     ArrayList<Action> actionList = new ArrayList<>();        
     for (String word : predictedWordsList) {
     actionList.add(new Action(word));
     }
     return new ActionSequence(actionList, unbiasedRefList);
     }*/
    
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
        for (Action a : modActSeq.getSequence()) {
            String predictedWord = a.getDecision();

            predictedWordsList.add(predictedWord);
            if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                arg1toBeMentioned = false;
            } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                arg2toBeMentioned = false;
            }
        }
        String wordTrainingVector = createWordTrainingVectorChoice(predictedWordsList, index, arg1toBeMentioned, arg2toBeMentioned);

        String[] details;
        details = wordTrainingVector.split(" ");
        TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

        availableActions.stream().forEach((action) -> {
            costs.put(action.getDecision(), 1.0);
        });
        costs.put(availableActions.get(Integer.parseInt(details[0])).getDecision(), modActSeq.getCost());

        for (int j = 1; j < details.length; j++) {
            String[] feature;
            feature = details[j].split(":");

            featureVector.put(feature[0], 1 - Double.parseDouble(feature[1]));
        }
        return new Instance(featureVector, costs);
    }
    
    public static JAROW trainClassifier(ArrayList<Instance> trainingWordInstances) {
        JAROW classifierWords = new JAROW();
        Collections.shuffle(trainingWordInstances);
        classifierWords.train(trainingWordInstances, true, true, 10, 0.1, true);
        return classifierWords;
    }
}
