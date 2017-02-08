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

import gnu.trove.map.hash.TObjectDoubleHashMap;
import jarow.Instance;
import jarow.JAROW;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

/**
 *
 * @author Gerasimos Lampouras
 */
abstract public class DatasetParser {
    
    ArrayList<DatasetInstance> trainingData;
    ArrayList<DatasetInstance> validationData;
    ArrayList<DatasetInstance> testingData;

    /**
     *
     */
    public static boolean useLMs;

    /**
     *
     */
    public static boolean useSubsetData;

    /**
     *
     */
    public boolean detailedResults;

    /**
     *
     */
    public boolean resetLists;

    /**
     *
     */
    public static String dataset;

    /**
     *
     */
    public int maxAttrRealizationSize;

    /**
     *
     */
    public int maxWordRealizationSize;

    /**
     *
     */
    public static Random randomGen;

    /**
     *
     */
    public double wordRefRolloutChance;
    
    //Training params

    /**
     *
     */
    public boolean averaging;

    /**
     *
     */
    public boolean shuffling;

    /**
     *
     */
    public int rounds;

    /**
     *
     */
    public Double initialTrainingParam;

    /**
     *
     */
    public Double additionalTrainingParam;

    /**
     *
     */
    public boolean adapt;
    
    /**
     *
     * @param useDAggerArg
     * @param useDAggerWord
     */
    abstract public void runImitationLearning(boolean useDAggerArg, boolean useDAggerWord);
    
    /**
     *
     * @param classifierAttrs
     * @param classifierWords
     * @param trainingData
     * @param testingData
     * @param availableAttributeActions
     * @param availableWordActions
     * @param detailedResults
     * @return
     */
    
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
    abstract public Double evaluateGeneration(HashMap<String, JAROW> classifierAttrs, HashMap<String, HashMap<String, JAROW>> classifierWords, ArrayList<DatasetInstance> testingData, HashMap<String, HashSet<String>> availableAttributeActions, HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions, boolean printResults, int epoch, boolean detailedResults);

    /**
     *
     * @param trainingData
     */
    abstract public void createNaiveAlignments(ArrayList<DatasetInstance> trainingData);

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
    abstract public Instance createAttrInstance(String predicate, String bestAction, ArrayList<String> previousGeneratedAttrs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, MeaningRepresentation MR, HashMap<String, HashSet<String>> availableAttributeActions);

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
    abstract public Instance createAttrInstanceWithCosts(String predicate, TObjectDoubleHashMap<String> costs, ArrayList<String> previousGeneratedAttrs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, HashMap<String, HashSet<String>> availableAttributeActions, MeaningRepresentation MR);

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
    abstract public Instance createWordInstance(String predicate, Action bestAction, ArrayList<String> previousGeneratedAttributes, ArrayList<Action> previousGeneratedWords, ArrayList<String> nextGeneratedAttributes, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesThatFollow, boolean wasValueMentioned, HashMap<String, HashSet<Action>> availableWordActions);

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
    abstract public Instance createWordInstanceWithCosts(String predicate, String currentAttrValue, TObjectDoubleHashMap<String> costs, ArrayList<String> generatedAttributes, ArrayList<Action> previousGeneratedWords, ArrayList<String> nextGeneratedAttributes, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesThatFollow, boolean wasValueMentioned, HashMap<String, HashSet<Action>> availableWordActions);
   
    /**
     *
     * @param phrase
     * @param subPhrase
     * @return
     */
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

    /**
     *
     * @param attribute
     * @param attrValuesToBeMentioned
     * @return
     */
    public String chooseNextValue(String attribute, HashSet<String> attrValuesToBeMentioned) {
        HashMap<String, Integer> relevantValues = new HashMap<>();
        attrValuesToBeMentioned.stream().forEach((attrValue) -> {
            String attr = attrValue.substring(0, attrValue.indexOf('='));
            String value = attrValue.substring(attrValue.indexOf('=') + 1);
            if (attr.equals(attribute)) {
                relevantValues.put(value, 0);
            }
        });
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

    /**
     *
     * @param di
     * @param wordSequence
     * @return
     */
    abstract public String postProcessWordSequence(DatasetInstance di, ArrayList<Action> wordSequence);

    /**
     *
     * @param di
     * @param refSeq
     * @return
     */
    abstract public String postProcessRef(DatasetInstance di, ArrayList<Action> refSeq);
    
    /**
     *
     * @param dataSize
     * @param trainedAttrClassifiers_0
     * @param trainedWordClassifiers_0
     * @return
     */
    abstract public boolean loadInitClassifiers(int dataSize, HashMap<String, JAROW> trainedAttrClassifiers_0, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0);

    /**
     *
     * @param dataSize
     * @param trainedAttrClassifiers_0
     * @param trainedWordClassifiers_0
     */
    abstract public void writeInitClassifiers(int dataSize, HashMap<String, JAROW> trainedAttrClassifiers_0, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0);
}
