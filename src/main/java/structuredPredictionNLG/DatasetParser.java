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
package structuredPredictionNLG;

import gnu.trove.map.hash.TObjectDoubleHashMap;
import imitationLearning.JLOLS;
import imitationLearning.LossFunction;
import jarow.Instance;
import jarow.JAROW;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import simpleLM.SimpleLM;

/**
 * This is an abstract specification of a DatasetParser.
 *
 * @author Gerasimos Lampouras
 * @organization University of Sheffield
 */
public abstract class DatasetParser {

    static final int SEED = 13;
    static final int THREAD_COUNT = Runtime.getRuntime().availableProcessors();

    // A random number generator
    private static Random randomGen;

    public static void resetRandomGen() {
        randomGen = new Random(SEED);
    }

    public static Random getRandomGen() {
        return randomGen;
    }

    // The identifier of the dataset
    private String dataset;
    // Training, validation, and testing subsets of the dataset    
    private ArrayList<DatasetInstance> trainingData;
    private ArrayList<DatasetInstance> validationData;
    private ArrayList<DatasetInstance> testingData;
    // Content and word actions, as observed from the dataset   
    private HashMap<String, HashSet<String>> availableContentActions;
    private HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions;
    //A flag determining whether stored caches should be reset or not
    private boolean resetStoredCaches;
    // A flag determining whether detailedResults per predicate should be calculated
    private boolean calculateResultsPerPredicate;
    // A flag determining whether to perform validation on the training, validation, or test data
    private String performEvaluationOn;
    // A flag determining whether to use random or naive alignments
    private String useAlignments;
    // Maximum lengths for content and word sequences, as observed in the training data
    private int maxContentSequenceLength;
    private int maxWordSequenceLength;
    // AROW parameters
    private boolean averaging;
    private boolean shuffling;
    private int rounds;
    private Double initialTrainingParam;
    private Double additionalTrainingParam;
    private boolean adapt;
    /*
     * The sequence in which the attribute/value pairs of the meaning representation have been mentioned in the reference.
     * Primarily used to determine the order in which values will be picked during generation.
     */
    private ArrayList<ArrayList<String>> observedAttrValueSequences;

    // Lists of all observed predicates, attributes, attribute/value pairs, and overall instances in the dataset
    private ArrayList<String> predicates = new ArrayList<>();
    private HashMap<String, HashSet<String>> attributes = new HashMap<>();
    private HashMap<String, HashSet<String>> attributeValuePairs = new HashMap<>();
    private HashMap<String, ArrayList<DatasetInstance>> datasetInstances = new HashMap<>();

    // Map of alignments between attribute values and subphrases of the references; naively calculated (e.g. through edit distance)
    private HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments = new HashMap<>();
    // Patterns describing the observed context of punctuation in the dataset references
    private HashMap<String, HashMap<ArrayList<Action>, Action>> punctuationPatterns = new HashMap<>();

    // The generated training data for content and word actions
    private HashMap<String, ArrayList<Instance>> predicateContentTrainingData;
    private HashMap<String, HashMap<String, ArrayList<Instance>>> predicateWordTrainingData;

    // Language models estimated on the word and content actions; one for each predicate
    private HashMap<String, SimpleLM> contentLMsPerPredicate = new HashMap<>();
    private HashMap<String, SimpleLM> wordLMsPerPredicate = new HashMap<>();

    // Map between 
    private HashMap<String, String> compositeWordsInData = new HashMap<>();

    /**
     * Main constructor
     * @param args Console arguments
     */
    public DatasetParser(String[] args) {
        resetRandomGen();

        Option helpOption = new Option("h", "help", false, "shows this message");
        Option datasetPathOption = new Option("d", "dataset", true, "dataset identifier");
        Option pOption = new Option("p", "p", true, "decaying factor for each epoch of LOLS: {0.0 ... 1.0}");
        Option sentenceCorrectionStepsOption = new Option("s", "scsteps", true, "number of steps to take after sentence correction");
        Option lossFunctionOption = new Option("l", "loss", true, "loss function to use: {B, R, BR, BC, RC, BRC}");
        Option resetStoredCachesOption = new Option("rst", "reset", false, "whether to reset any stored caches");
        Option testingOption = new Option("e", "evaluate", true, "whether to evaluate on training, validation or testing data: {train, valid, test}");
        Option alignmentOption = new Option("a", "alignment", true, "whether to employ random or naive alignments: {random, naive}");
        Option detailsOption = new Option("perPred", "resultsPerPredicate", false, "whether to also perform evaluation on each predicate seperately");
        Option averagingOption = new Option("avg", "averaging", false, "AROW parameter: whether to average over all intermediate weight vectors, instead of using the final weight vectors");
        Option shufflingOption = new Option("sh", "shuffling", false, "AROW parameter: whether to shuffle the training instances");
        Option roundsOption = new Option("r", "rounds", true, "AROW parameter: for how many rounds to repeat the training");
        Option initialTrainingParamOption = new Option("init", "initialTrainingParam", true, "AROW parameter: training parameter for the initial policies");
        Option additionalTrainingParamOption = new Option("add", "additionalTrainingParam", true, "AROW parameter: training parameter for the additional policies");
        Option adaptOption = new Option("ad", "adapt", false, "AROW parameter: whether to use AROW algorithm or passive aggressive-II with prediction-based updates");

        Options options = new Options();
        options.addOption(helpOption);
        options.addOption(datasetPathOption);
        options.addOption(pOption);
        options.addOption(sentenceCorrectionStepsOption);
        options.addOption(lossFunctionOption);
        options.addOption(resetStoredCachesOption);
        options.addOption(testingOption);
        options.addOption(alignmentOption);
        options.addOption(detailsOption);
        options.addOption(averagingOption);
        options.addOption(shufflingOption);
        options.addOption(roundsOption);
        options.addOption(initialTrainingParamOption);
        options.addOption(additionalTrainingParamOption);
        options.addOption(adaptOption);

        CommandLineParser parser = new BasicParser();
        try {
            CommandLine cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("sb server", options);
                System.exit(1);
            } else {
                if (cmd.hasOption("dataset")) {
                    setDataset(cmd.getOptionValue("dataset"));
                } else {
                    setDataset("hotel");
                }
                if (cmd.hasOption("p")) {
                    JLOLS.p = Double.parseDouble(cmd.getOptionValue("p"));
                }
                if (cmd.hasOption("scsteps")) {
                    JLOLS.sentenceCorrectionFurtherSteps = Integer.parseInt(cmd.getOptionValue("scsteps"));
                }
                LossFunction.metric = "BRC";
                if (cmd.hasOption("loss")) {
                    String opt = cmd.getOptionValue("loss");
                    if (!opt.isEmpty()
                            && (opt.equals("B")
                            || opt.equals("R")
                            || opt.equals("BC")
                            || opt.equals("RC")
                            || opt.equals("BRC")
                            || opt.equals("BR"))) {
                        LossFunction.metric = opt;
                    }
                }
                if (cmd.hasOption("reset")) {
                    setResetStoredCaches(true);
                } else {
                    setResetStoredCaches(false);
                }
                setPerformEvaluationOn("test");
                if (cmd.hasOption("test")) {
                    String opt = cmd.getOptionValue("test");
                    if (!opt.isEmpty()
                            && (opt.equals("test")
                            || opt.equals("valid")
                            || opt.equals("train"))) {
                        setPerformEvaluationOn(opt);
                    }
                }
                setUseAlignments("naive");
                if (cmd.hasOption("alignment")) {
                    String opt = cmd.getOptionValue("alignment");
                    if (!opt.isEmpty()
                            && (opt.equals("naive")
                            || opt.equals("random"))) {
                        setPerformEvaluationOn(opt);
                    }
                }
                if (cmd.hasOption("resultsPerPredicate")) {
                    setCalculateResultsPerPredicate(true);
                } else {
                    setCalculateResultsPerPredicate(false);
                }
                if (cmd.hasOption("averaging")) {
                    setAveraging(true);
                } else {
                    setAveraging(false);
                }
                if (cmd.hasOption("shuffling")) {
                    setShuffling(true);
                } else {
                    setShuffling(false);
                }
                if (cmd.hasOption("rounds")) {
                    setRounds(Integer.parseInt(cmd.getOptionValue("rounds")));
                } else {
                    setRounds(10);
                }
                if (cmd.hasOption("initialTrainingParam")) {
                    setInitialTrainingParam(Double.parseDouble(cmd.getOptionValue("initialTrainingParam")));
                } else {
                    setInitialTrainingParam(100.0);
                }
                if (cmd.hasOption("additionalTrainingParam")) {
                    setAdditionalTrainingParam(Double.parseDouble(cmd.getOptionValue("additionalTrainingParam")));
                } else {
                    setAdditionalTrainingParam(100.0);
                }
                if (cmd.hasOption("adapt")) {
                    setAdapt(true);
                } else {
                    setAdapt(false);
                }
            }
        } catch (ParseException ex) {
            System.out.println(ex.getMessage());
            System.out.println("Try \"--help\" option for details.");
        }
        trainingData = new ArrayList<>();
        validationData = new ArrayList<>();
        testingData = new ArrayList<>();
        
        observedAttrValueSequences = new ArrayList<>();
    }

    /**
     * In this method, the dataset should be parsed and the predicate, attribute, attribute/value, and value alignment collections should be populated.
     * Here, the data should also be split in training, validation, and testing subsets.
     */
    abstract public void parseDataset() ;
    
    /**
     * During this method, we need to calculate the alignments (naive or random), the language models, the available content and word actions, and finally the feature vectors.
     */
    abstract public void createTrainingData();
        
    /**
     * Creates and call the imitation learning engine
     */
    public void performImitationLearning() {
        JLOLS ILEngine = new JLOLS(this);
        if (getPerformEvaluationOn().equals("train")) {
            ILEngine.runLOLS(getTrainingData());
        } else if (getPerformEvaluationOn().equals("valid")) {
            ILEngine.runLOLS(getValidationData());
        } else if (getPerformEvaluationOn().equals("test")) {
            ILEngine.runLOLS(getTestingData());
        }
    }
    
    /**
     *
     * @param classifierAttrs
     * @param classifierWords
     * @param testingData
     * @param epoch
     * @return
     */
    abstract public Double evaluateGeneration(HashMap<String, JAROW> classifierAttrs, HashMap<String, HashMap<String, JAROW>> classifierWords, ArrayList<DatasetInstance> testingData, int epoch);

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
    abstract public Instance createContentInstance(String predicate, String bestAction, ArrayList<String> previousGeneratedAttrs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, MeaningRepresentation MR, HashMap<String, HashSet<String>> availableAttributeActions);

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
    abstract public Instance createContentInstanceWithCosts(String predicate, TObjectDoubleHashMap<String> costs, ArrayList<String> previousGeneratedAttrs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, HashMap<String, HashSet<String>> availableAttributeActions, MeaningRepresentation MR);

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
                for (ArrayList<String> mentionedValueSeq : observedAttrValueSequences) {
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
     * @param mr
     * @param refSeq
     * @return
     */
    abstract public String postProcessRef(MeaningRepresentation mr, ArrayList<Action> refSeq);

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

    public ArrayList<DatasetInstance> getTrainingData() {
        return trainingData;
    }

    public ArrayList<DatasetInstance> getValidationData() {
        return validationData;
    }

    public ArrayList<DatasetInstance> getTestingData() {
        return testingData;
    }

    public boolean isResetStoredCaches() {
        return resetStoredCaches;
    }

    public int getMaxWordSequenceLength() {
        return maxWordSequenceLength;
    }

    public HashMap<String, HashMap<ArrayList<String>, Double>> getValueAlignments() {
        return valueAlignments;
    }

    public HashMap<String, HashMap<ArrayList<Action>, Action>> getPunctuationPatterns() {
        return punctuationPatterns;
    }

    public HashMap<String, SimpleLM> getWordLMsPerPredicate() {
        return wordLMsPerPredicate;
    }

    public String getDataset() {
        return dataset;
    }

    public final void setDataset(String dataset) {
        this.dataset = dataset;
    }

    public int getMaxContentSequenceLength() {
        return maxContentSequenceLength;
    }

    public void setMaxContentSequenceLength(int maxContentSequenceLength) {
        this.maxContentSequenceLength = maxContentSequenceLength;
    }

    public void setMaxWordSequenceLength(int maxWordSequenceLength) {
        this.maxWordSequenceLength = maxWordSequenceLength;
    }

    public boolean getAveraging() {
        return averaging;
    }

    public final void setAveraging(boolean averaging) {
        this.averaging = averaging;
    }

    public boolean getShuffling() {
        return shuffling;
    }

    public final void setShuffling(boolean shuffling) {
        this.shuffling = shuffling;
    }

    public int getRounds() {
        return rounds;
    }

    public final void setRounds(int rounds) {
        this.rounds = rounds;
    }

    public Double getInitialTrainingParam() {
        return initialTrainingParam;
    }

    public final void setInitialTrainingParam(Double initialTrainingParam) {
        this.initialTrainingParam = initialTrainingParam;
    }

    public Double getAdditionalTrainingParam() {
        return additionalTrainingParam;
    }

    public final void setAdditionalTrainingParam(Double additionalTrainingParam) {
        this.additionalTrainingParam = additionalTrainingParam;
    }

    public boolean isAdapt() {
        return adapt;
    }

    public final void setAdapt(boolean adapt) {
        this.adapt = adapt;
    }

    public final void setResetStoredCaches(boolean resetStoredCaches) {
        this.resetStoredCaches = resetStoredCaches;
    }

    public String getPerformEvaluationOn() {
        return performEvaluationOn;
    }

    public final void setPerformEvaluationOn(String performEvaluationOn) {
        this.performEvaluationOn = performEvaluationOn;
    }

    public String getUseAlignments() {
        return useAlignments;
    }

    public final void setUseAlignments(String useAlignments) {
        this.useAlignments = useAlignments;
    }

    public boolean isCalculateResultsPerPredicate() {
        return calculateResultsPerPredicate;
    }

    public final void setCalculateResultsPerPredicate(boolean calculateResultsPerPredicate) {
        this.calculateResultsPerPredicate = calculateResultsPerPredicate;
    }

    public ArrayList<ArrayList<String>> getObservedAttrValueSequences() {
        return observedAttrValueSequences;
    }

    public ArrayList<String> getPredicates() {
        return predicates;
    }

    public HashMap<String, HashSet<String>> getAttributes() {
        return attributes;
    }

    public HashMap<String, HashSet<String>> getAttributeValuePairs() {
        return attributeValuePairs;
    }

    public HashMap<String, ArrayList<DatasetInstance>> getDatasetInstances() {
        return datasetInstances;
    }

    public HashMap<String, ArrayList<Instance>> getPredicateContentTrainingData() {
        return predicateContentTrainingData;
    }

    public HashMap<String, HashMap<String, ArrayList<Instance>>> getPredicateWordTrainingData() {
        return predicateWordTrainingData;
    }

    public HashMap<String, SimpleLM> getContentLMsPerPredicate() {
        return contentLMsPerPredicate;
    }

    public HashMap<String, String> getCompositeWordsInData() {
        return compositeWordsInData;
    }

    public void setTrainingData(ArrayList<DatasetInstance> trainingData) {
        this.trainingData = trainingData;
    }

    public void setValidationData(ArrayList<DatasetInstance> validationData) {
        this.validationData = validationData;
    }

    public void setTestingData(ArrayList<DatasetInstance> testingData) {
        this.testingData = testingData;
    }

    public void setContentLMsPerPredicate(HashMap<String, SimpleLM> contentLMsPerPredicate) {
        this.contentLMsPerPredicate = contentLMsPerPredicate;
    }

    public void setWordLMsPerPredicate(HashMap<String, SimpleLM> wordLMsPerPredicate) {
        this.wordLMsPerPredicate = wordLMsPerPredicate;
    }

    public void setValueAlignments(HashMap<String, HashMap<ArrayList<String>, Double>> valueAlignments) {
        this.valueAlignments = valueAlignments;
    }

    public void setObservedAttrValueSequences(ArrayList<ArrayList<String>> observedAttrValueSequences) {
        this.observedAttrValueSequences = observedAttrValueSequences;
    }

    public void setPredicates(ArrayList<String> predicates) {
        this.predicates = predicates;
    }

    public void setAttributes(HashMap<String, HashSet<String>> attributes) {
        this.attributes = attributes;
    }

    public void setAttributeValuePairs(HashMap<String, HashSet<String>> attributeValuePairs) {
        this.attributeValuePairs = attributeValuePairs;
    }

    public void setDatasetInstances(HashMap<String, ArrayList<DatasetInstance>> datasetInstances) {
        this.datasetInstances = datasetInstances;
    }

    public void setPunctuationPatterns(HashMap<String, HashMap<ArrayList<Action>, Action>> punctuationPatterns) {
        this.punctuationPatterns = punctuationPatterns;
    }

    public void setPredicateContentTrainingData(HashMap<String, ArrayList<Instance>> predicateContentTrainingData) {
        this.predicateContentTrainingData = predicateContentTrainingData;
    }

    public void setPredicateWordTrainingData(HashMap<String, HashMap<String, ArrayList<Instance>>> predicateWordTrainingData) {
        this.predicateWordTrainingData = predicateWordTrainingData;
    }

    public void setCompositeWordsInData(HashMap<String, String> compositeWordsInData) {
        this.compositeWordsInData = compositeWordsInData;
    }

    public HashMap<String, HashSet<String>> getAvailableContentActions() {
        return availableContentActions;
    }

    public void setAvailableContentActions(HashMap<String, HashSet<String>> availableContentActions) {
        this.availableContentActions = availableContentActions;
    }

    public HashMap<String, HashMap<String, HashSet<Action>>> getAvailableWordActions() {
        return availableWordActions;
    }

    public void setAvailableWordActions(HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions) {
        this.availableWordActions = availableWordActions;
    }
}