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

import edu.stanford.nlp.mt.metrics.BLEUMetric;
import edu.stanford.nlp.mt.metrics.NISTMetric;
import edu.stanford.nlp.mt.tools.NISTTokenizer;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.util.IStrings;
import edu.stanford.nlp.mt.util.ScoredFeaturizedTranslation;
import edu.stanford.nlp.mt.util.Sequence;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import imitationLearning.JLOLS;
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
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.json.JSONArray;
import org.json.JSONException;
import similarity_measures.Levenshtein;
import similarity_measures.Rouge;
import simpleLM.SimpleLM;

/**
 * Implementation of a DatasetParse for the SF datasets introduced by Wen et al. (2016)
 * @author Gerasimos Lampouras
 * @organization University of Sheffield
 */
public class SFX extends DatasetParser {

    private static final Logger LOG = Logger.getLogger(SFX.class.getName());
    
    /**
     * Main method; initialized a new DatasetParser for the SF datasets, and performs imitation learning on the it.
     * @param args
     */
    public static void main(String[] args) {
        SFX sfx = new SFX(args);
        sfx.parseDataset();
        sfx.createTrainingData();
        sfx.performImitationLearning();
    }
    // Collections for parsing the results of Wen et al. (2016), specifically the dataset splits (into train/valid/test), as well as the evaluation references used by Wen et al. in his experiments.
    // These are mainly used to ensure our results are comparable with Wen et al. (2016)
    private HashMap<String, String> wenDaToGen = new HashMap<>();
    private HashMap<String, ArrayList<Sequence<IString>>> wenEvaluationReferenceSequences = new HashMap<>();
    private HashMap<String, ArrayList<String>> wenEvaluationReferences = new HashMap<>();

    /**
     * Main constructor
     * @param args Console arguments
     */
    public SFX(String[] args) {
        super(args);
    }        
    
    /**
     * Method where the dataset is parsed and the predicate, attribute, attribute/value, and value alignment collections are populated.
     * Here, the data is also split in training, validation, and testing subsets.
     */
    @Override
    public void parseDataset() {        
        File dataFile = new File("sfx_data/sfx" + getDataset() + "/train+valid+test.json");
        String wenFile = "results/wenResults/sfxhotel.log";
        if (getDataset().equals("restaurant")) {
            wenFile = "results/wenResults/sfxrest.log";
        }
        
        if (!loadLists()) {
            // Populate the predicate, attribute, attribute/value, and value alignment collections
            createLists(dataFile);
            writeLists();
        }
        
        // Read Wen et al.'s (2016) result files to ascertain which DatasetInstances are used for testing
        ArrayList<String> testWenMRs = new ArrayList<>();
        wenDaToGen = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(wenFile))) {
            String s;
            boolean inGen = false;
            boolean inRef = false;
            String da = "";
            while ((s = br.readLine()) != null) {
                if (s.startsWith("DA")) {
                    inGen = false;
                    inRef = false;
                    da = s.substring(s.indexOf(':') + 1).replaceAll(",", ";").replaceAll("no or yes", "yes or no").replaceAll("ave ; presidio", "ave and presidio").replaceAll("point ; ste", "point and ste").trim();
                    testWenMRs.add(da);
                } else if (s.startsWith("Gen")) {
                    inGen = true;
                } else if (s.startsWith("Ref")) {
                    inRef = true;
                } else if (inGen) {
                    inGen = false;

                    if (!wenDaToGen.containsKey(da)) {
                        wenDaToGen.put(da.toLowerCase(), s.trim());
                    }
                    da = "";
                } else if (inRef && s.trim().isEmpty()) {
                    inRef = false;
                    da = "";
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }

        // Create a test subset of the DatasetInstances while making sure that it used the exact same DatasetInstances that Wen et al. (2016) also used.
        ArrayList<DatasetInstance> restData = new ArrayList<>();
        getPredicates().forEach((predicate) -> {
            getDatasetInstances().get(predicate).forEach((in) -> {
                if (testWenMRs.contains(in.getMeaningRepresentation().getMRstr())) {
                    getTestingData().add(in);
                    testWenMRs.remove(in.getMeaningRepresentation().getMRstr());
                } else {
                    restData.add(in);
                }
            });
        });
        // Split the rest of the DatasetInstances (minus the one's used for testing) into validation and training subsets.
        Collections.shuffle(restData, getRandomGen());
        for (int i = 0; i < restData.size(); i++) {
            if (i < getTestingData().size()) {
                getValidationData().add(restData.get(i));
            } else {
                getTrainingData().add(restData.get(i));
            }
        }
        resetRandomGen();
        
        // We need to make another "deep" parse of Wen et al.'s (2016) results to gather the evaluation references he used.
        // Idially we would want to merge this parse pass with the one made previously; keeping it separate for now, since we only need the second one when we want to explicitly use Wen's references as well.
        parseWenFiles(isCalculateResultsPerPredicate(), wenFile);
        System.out.println("Training data size: " + getTrainingData().size());
        System.out.println("Validation data size: " + getValidationData().size());
        System.out.println("Test data size: " + getTestingData().size());
    }
    
    /**
     * During this method, we calculate the alignments (naive or random), the language models, the available content and word actions, and finally the feature vectors.
     */
    @Override
    public void createTrainingData() {
        //setTrainingData(new ArrayList<>(getTrainingData().subList(0, 50)));
        //setTestingData(new ArrayList<>(getTrainingData()));

        // Calculate alignments between the word of the sentence and the atribute/values
        if (getUseAlignments().equals("naive")) {
            createNaiveAlignments(getTrainingData());
        } else {
            createRandomAlignments(getTrainingData());
        }

        // Create (or load from cache) the content and word language models per predicate
        if (isResetStoredCaches() || !loadLMs()) {
            HashMap<String, ArrayList<ArrayList<String>>> LMWordTrainingPerPred = new HashMap<>();
            HashMap<String, ArrayList<ArrayList<String>>> LMAttrTrainingPerPred = new HashMap<>();
            getTrainingData().stream().map((di) -> {
                if (!LMWordTrainingPerPred.containsKey(di.getMeaningRepresentation().getPredicate())) {
                    LMWordTrainingPerPred.put(di.getMeaningRepresentation().getPredicate(), new ArrayList<ArrayList<String>>());
                    LMAttrTrainingPerPred.put(di.getMeaningRepresentation().getPredicate(), new ArrayList<ArrayList<String>>());
                }
                return di;
            }).forEachOrdered((di) -> {
                HashSet<ArrayList<Action>> seqs = new HashSet<>();
                seqs.add(di.getDirectReferenceSequence());
                seqs.forEach((seq) -> {
                    ArrayList<String> wordSeq = new ArrayList<>();
                    ArrayList<String> attrSeq = new ArrayList<>();

                    // We add some empty tokens at the start of each sequence
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
                            attrSeq.add(seq.get(i).getAttribute());
                        } else if (!attrSeq.get(attrSeq.size() - 1).equals(seq.get(i).getAttribute())) {
                            attrSeq.add(seq.get(i).getAttribute());
                        }
                    }
                    wordSeq.add(Action.TOKEN_END);
                    LMWordTrainingPerPred.get(di.getMeaningRepresentation().getPredicate()).add(wordSeq);
                    LMAttrTrainingPerPred.get(di.getMeaningRepresentation().getPredicate()).add(attrSeq);
                });
            });

            setWordLMsPerPredicate(new HashMap<>());
            setContentLMsPerPredicate(new HashMap<>());
            LMWordTrainingPerPred.keySet().stream().map((pred) -> {
                SimpleLM simpleWordLM = new SimpleLM(3);
                simpleWordLM.trainOnStrings(LMWordTrainingPerPred.get(pred));
                getWordLMsPerPredicate().put(pred, simpleWordLM);
                return pred;
            }).forEachOrdered((pred) -> {
                SimpleLM simpleAttrLM = new SimpleLM(3);
                simpleAttrLM.trainOnStrings(LMAttrTrainingPerPred.get(pred));
                getContentLMsPerPredicate().put(pred, simpleAttrLM);
            });
            writeLMs();
        }

        // Go through the sequences in the data and populate the available content and word action dictionaries
        // We populate a distinct word dictionary for each attribute, and populate it with the words of word sequences whose corresponding content sequences contain that attribute
        HashMap<String, HashSet<String>> availableContentActions = new HashMap<>();
        HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions = new HashMap<>();
        getTrainingData().forEach((DI) -> {
            String predicate = DI.getMeaningRepresentation().getPredicate();
            if (!availableContentActions.containsKey(predicate)) {
                availableContentActions.put(predicate, new HashSet<String>());
                availableContentActions.get(predicate).add(Action.TOKEN_END);
            }
            if (!availableWordActions.containsKey(predicate)) {
                availableWordActions.put(predicate, new HashMap<String, HashSet<Action>>());
            }
            ArrayList<Action> realization = DI.getDirectReferenceSequence();
            realization.stream().filter((a) -> (!a.getAttribute().equals(Action.TOKEN_END))).forEachOrdered((Action a) -> {
                String attr;
                if (a.getAttribute().contains("=")) {
                    attr = a.getAttribute().substring(0, a.getAttribute().indexOf('='));
                } else {
                    attr = a.getAttribute();
                }
                availableContentActions.get(predicate).add(attr);
                if (!availableWordActions.get(predicate).containsKey(attr)) {
                    availableWordActions.get(predicate).put(attr, new HashSet<Action>());
                    availableWordActions.get(predicate).get(attr).add(new Action(Action.TOKEN_END, attr));
                }
                if (!a.getWord().equals(Action.TOKEN_START)
                        && !a.getWord().equals(Action.TOKEN_END)
                        && !a.getWord().matches("([,.?!;:'])")) {
                    if (a.getWord().startsWith(Action.TOKEN_X)) {
                        if (a.getWord().substring(3, a.getWord().lastIndexOf('_')).toLowerCase().trim().equals(attr)) {
                            availableWordActions.get(predicate).get(attr).add(new Action(a.getWord(), attr));
                        }
                    } else {
                        availableWordActions.get(predicate).get(attr).add(new Action(a.getWord(), attr));
                    }
                }
            });
        });
        setAvailableContentActions(availableContentActions);
        setAvailableWordActions(availableWordActions);

        //When using random alignments we do not consider the value alignments either
        if (getUseAlignments().equals("random")) {
            setValueAlignments(new HashMap<>());
        }

        // Infer the feature vectors of the training data
        if (isResetStoredCaches() || !loadTrainingData(getTrainingData().size())) {
            System.out.print("Create training data...");
            Object[] results = inferFeatureAndCostVectors();
            System.out.print("almost...");
            
            @SuppressWarnings("unchecked")
            ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>> getPredicateContentTrainingDataBefore = (ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>) results[0];
            @SuppressWarnings("unchecked")
            ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>> getPredicateWordTrainingDataBefore = (ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>>) results[1];

            // Reorganize the feature/cost vector collections 
            // Initially they are mapped according to DatasetInstance (since it helps with parallel processing) but we prefer them mapped by predicate for training
            setPredicateContentTrainingData(new HashMap<>());
            getTrainingData().forEach((di) -> {
                getPredicateContentTrainingDataBefore.get(di).keySet().stream().map((predicate) -> {
                    if (!getPredicateContentTrainingData().containsKey(predicate)) {
                        getPredicateContentTrainingData().put(predicate, new ArrayList<Instance>());
                    }
                    return predicate;
                }).forEachOrdered((predicate) -> {
                    getPredicateContentTrainingData().get(predicate).addAll(getPredicateContentTrainingDataBefore.get(di).get(predicate));
                });
            });
            setPredicateWordTrainingData(new HashMap<>());
            getTrainingData().forEach((di) -> {
                getPredicateWordTrainingDataBefore.get(di).keySet().stream().map((predicate) -> {
                    if (!getPredicateWordTrainingData().containsKey(predicate)) {
                        getPredicateWordTrainingData().put(predicate, new HashMap<String, ArrayList<Instance>>());
                    }
                    return predicate;
                }).forEachOrdered((predicate) -> {
                    getPredicateWordTrainingDataBefore.get(di).get(predicate).keySet().stream().map((attribute) -> {
                        if (!getPredicateWordTrainingData().get(predicate).containsKey(attribute)) {
                            getPredicateWordTrainingData().get(predicate).put(attribute, new ArrayList<Instance>());
                        }
                        return attribute;
                    }).forEachOrdered((attribute) -> {
                        getPredicateWordTrainingData().get(predicate).get(attribute).addAll(getPredicateWordTrainingDataBefore.get(di).get(predicate).get(attribute));
                    });
                });
            });
            writeTrainingData(getTrainingData().size());
        }
    }

    /**
     *
     * @param printResults
     * @param resultsFile
     */
    public void parseWenFiles(boolean printResults, String resultsFile) {
        wenEvaluationReferenceSequences = new HashMap<>();
        wenEvaluationReferences = new HashMap<>();
        HashMap<String, String> daToGen = new HashMap<>();
        HashMap<String, String> predictedWordSequences_overAllPredicates = new HashMap<>();
        HashMap<String, HashMap<String, String>> predictedWordSequences_perPredicates = new HashMap<>();
        HashMap<String, ArrayList<String>> finalReferencesWordSequences = new HashMap<>();
        HashMap<String, Double> attrCoverage = new HashMap<>();

        HashMap<String, String> MRtoAbstractMR = new HashMap<>();
        HashMap<String, String> abstractMRtoMR = new HashMap<>();
        HashMap<String, String> abstractMRtoText = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(resultsFile))) {
            String s;
            boolean inGen = false;
            boolean inRef = false;
            boolean firstRef = true;
            ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
            ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
            ArrayList<Sequence<IString>> references = new ArrayList<>();
            ArrayList<String> referencesStrings = new ArrayList<>();
            String da = "";
            String predicate = "";
            String abstractMR = "";
            String predictedWordSequence = "";
            String MRstr = "";
            int gens = 0;
            int c = 0;
            HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
            while ((s = br.readLine()) != null) {
                if (s.startsWith("DA")) {
                    inGen = false;
                    inRef = false;
                    firstRef = true;
                    da = s.substring(s.indexOf(':') + 1).trim();
                    c++;
                    MRstr = s.substring(s.indexOf(':') + 1).replaceAll(",", ";").replaceAll("no or yes", "yes or no").replaceAll("ave ; presidio", "ave and presidio").replaceAll("point ; ste", "point and ste").trim();
                    predicate = MRstr.substring(0, MRstr.indexOf('('));
                    abstractMR = predicate + ":";
                    String attributesStr = MRstr.substring(MRstr.indexOf('(') + 1, MRstr.length() - 1);
                    attributeValues = new HashMap<>();
                    if (!attributesStr.isEmpty()) {
                        HashMap<String, Integer> attrXIndeces = new HashMap<>();

                        String[] args = attributesStr.split(";");
                        if (attributesStr.contains("|")) {
                            System.exit(0);
                        }
                        for (String arg : args) {
                            String attr;
                            String value = "";
                            if (arg.contains("=")) {
                                String[] subAttr = arg.split("=");
                                value = subAttr[1].toLowerCase();
                                attr = subAttr[0].toLowerCase().replaceAll("_", "");

                                if (value.startsWith("\'")) {
                                    value = value.substring(1, value.length() - 1);
                                }
                                if (value.equals("true")) {
                                    value = "yes";
                                }
                                if (value.equals("false")) {
                                    value = "no";
                                }
                                if (value.equals("dontcare")) {
                                    value = "dont_care";
                                }
                                if (value.equals("no")
                                        || value.equals("yes")
                                        || value.equals("yes or no")
                                        || value.equals("none")
                                        || value.equals("empty")) {
                                    attr += "_" + value.replaceAll(" ", "_");
                                    value = attr;
                                }
                                if (value.equals("dont_care")) {
                                    String v = value;
                                    value = attr;
                                    attr = v;
                                }
                            } else {
                                attr = arg.replaceAll("_", "");
                            }
                            if (!attributeValues.containsKey(attr)) {
                                attributeValues.put(attr, new HashSet<String>());
                            }
                            if (value.isEmpty()) {
                                value = attr;
                            }

                            if (value.startsWith("\'")) {
                                value = value.substring(1, value.length() - 1);
                            }
                            if (value.toLowerCase().startsWith("x")) {
                                int index = 0;
                                if (!attrXIndeces.containsKey(attr)) {
                                    attrXIndeces.put(attr, 1);
                                } else {
                                    index = attrXIndeces.get(attr);
                                    attrXIndeces.put(attr, index + 1);
                                }
                                value = "x" + index;
                            }
                            if (value.isEmpty()) {
                                System.exit(0);
                            }

                            attributeValues.get(attr).add(value.trim().toLowerCase());
                        }
                    }
                } else if (s.startsWith("Gen")) {
                    gens = 0;
                    inGen = true;
                } else if (s.startsWith("Ref")) {
                    inRef = true;
                    references = new ArrayList<>();
                } else if (inGen) {
                    if (s.trim().isEmpty()) {
                        inGen = false;
                    } else {
                        predictedWordSequence = s.trim().toLowerCase();

                        Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedWordSequence));
                        ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
                        generations.add(tran);

                        inGen = false;
                        gens++;
                        da = "";
                    }
                } else if (inRef) {
                    if (s.trim().isEmpty()) {
                        for (int i = 0; i < gens; i++) {
                            finalReferences.add(references);
                            finalReferencesWordSequences.put(MRstr, referencesStrings);
                        }
                        wenEvaluationReferenceSequences.put(MRstr, references);
                        wenEvaluationReferences.put(MRstr, referencesStrings);
                        references = new ArrayList<>();
                        referencesStrings = new ArrayList<>();
                        inRef = false;
                        da = "";
                    } else {
                        String cleanedWords = s.trim().replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                        referencesStrings.add(cleanedWords);
                        references.add(IStrings.tokenize(NISTTokenizer.tokenize(cleanedWords)));

                        if (firstRef) {
                            ArrayList<String> attrs = new ArrayList<>(attributeValues.keySet());
                            Collections.sort(attrs);
                            HashMap<String, Integer> xCounts = new HashMap<>();
                            attrs.forEach((attr) -> {
                                xCounts.put(attr, 0);
                            });
                            for (String attr : attrs) {
                                abstractMR += attr + "={";

                                ArrayList<String> values = new ArrayList<>(attributeValues.get(attr));
                                Collections.sort(values);
                                for (String value : values) {

                                    if (attr.equals("name")
                                            || attr.equals("type")
                                            || attr.equals("pricerange")
                                            || attr.equals("price")
                                            || attr.equals("phone")
                                            || attr.equals("address")
                                            || attr.equals("postcode")
                                            || attr.equals("area")
                                            || attr.equals("near")
                                            || attr.equals("food")
                                            || attr.equals("goodformeal")
                                            || attr.equals("count")) {
                                        abstractMR += Action.TOKEN_X + attr + "_" + xCounts.get(attr) + ",";
                                        xCounts.put(attr, xCounts.get(attr) + 1);
                                    } else {
                                        abstractMR += value + ",";
                                    }
                                }
                                abstractMR += "}";
                            }

                            MRtoAbstractMR.put(da + c, abstractMR);
                            abstractMRtoMR.put(abstractMR, da);
                            abstractMRtoText.put(abstractMR, predictedWordSequence);

                            predictedWordSequences_overAllPredicates.put(MRstr, predictedWordSequence);

                            if (!predictedWordSequences_perPredicates.containsKey(predicate)) {
                                predictedWordSequences_perPredicates.put(predicate, new HashMap<String, String>());
                            }
                            predictedWordSequences_perPredicates.get(predicate).put(MRstr, predictedWordSequence);

                            if (!daToGen.containsKey(da)) {
                                daToGen.put(da.toLowerCase(), s.trim());
                            }
                            int mentioned = 0;
                            int total = 0;
                            ArrayList<String> errors = new ArrayList<String>();
                            String gen = " " + s.trim().toLowerCase() + " ";
                            for (String attr : attributeValues.keySet()) {
                                for (String value : attributeValues.get(attr)) {
                                    String searchValue = value;
                                    boolean ment = false;
                                    if (attr.startsWith("hasinternet")
                                            && gen.contains("internet")) {
                                        ment = true;
                                    } else if (attr.startsWith("dogsallowed")
                                            && gen.contains("dog")) {
                                        ment = true;
                                    } else if (attr.startsWith("kidsallowed")
                                            && (gen.contains("kid") || gen.contains("child"))) {
                                        ment = true;
                                    } else if (attr.startsWith("goodformeal")
                                            && (gen.contains(searchValue) || gen.contains("meal") || gen.contains("breakfast"))) {
                                        ment = true;
                                    } else if (attr.startsWith("acceptscreditcards")
                                            && gen.contains("credit")) {
                                        ment = true;
                                    }
                                    if (searchValue.startsWith("dont_care") || searchValue.startsWith("none") || searchValue.startsWith("yes or no") || searchValue.startsWith("no or yes")) {
                                        searchValue = attr;
                                    } else if (searchValue.contains("or")) {
                                        String[] values = searchValue.split(" or ");
                                        for (String v : values) {
                                            if (gen.contains(v)) {
                                                ment = true;
                                            }
                                        }
                                    }
                                    if (searchValue.startsWith("pricerange")
                                            && (gen.contains(searchValue) || gen.contains("price") || gen.contains("range"))) {
                                        ment = true;
                                    } else if (searchValue.startsWith("area")
                                            && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                                        ment = true;
                                    } else if (searchValue.startsWith("near")
                                            && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                                        ment = true;
                                    } else if (gen.contains(searchValue)) {
                                        ment = true;
                                    }

                                    if (ment) {
                                        mentioned++;
                                    } else {
                                        errors.add("not mentioned -> " + attr + ":" + attributeValues.get(attr));
                                    }
                                    total++;
                                }
                            }
                            double err = 1.0 - (mentioned / (double) total);
                            if (Double.isNaN(err)) {
                                err = 0.0;
                            }
                            attrCoverage.put(predictedWordSequence, err);
                        }
                    }
                    firstRef = false;
                }
            }
            if (printResults) {
                double avgErr = 0.0;
                avgErr = attrCoverage.values().stream().map((err) -> err).reduce(avgErr, (accumulator, _item) -> accumulator + _item);
                avgErr /= attrCoverage.size();
                System.out.println(finalReferences.size() + "\t" + generations.size() + "\t" + attrCoverage.size());
                BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
                NISTMetric NIST = new NISTMetric(finalReferences);
                System.out.println("WEN BLEU: \t" + BLEU.score(generations));

                double avgRougeScore = 0.0;
                String detailedRes = "";

                avgRougeScore = getTestingData().stream().map((di) -> {
                    double maxRouge = 0.0;
                    String pWS = predictedWordSequences_overAllPredicates.get(di.getMeaningRepresentation().getMRstr()).replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                    for (String ref : finalReferencesWordSequences.get(di.getMeaningRepresentation().getMRstr())) {
                        double rouge = Rouge.ROUGE_N(pWS, ref, 4);
                        if (rouge > maxRouge) {
                            maxRouge = rouge;
                        }
                    }
                    return maxRouge;
                }).map((maxRouge) -> maxRouge).reduce(avgRougeScore, (accumulator, _item) -> accumulator + _item);
                System.out.println("WEN ROUGE: \t" + (avgRougeScore / getTestingData().size()));
                System.out.println("WEN NIST: \t" + NIST.score(generations));
                System.out.println("WEN ERR: \t" + avgErr);
                System.out.println("WEN TOTAL: \t" + getTestingData().size());
                System.out.println("///////////////////");

                double uniqueMRsInTestAndNotInTrainAllPredWordBLEU = 0.0;
                double uniqueMRsInTestAndNotInTrainAllPredWordROUGE = 0.0;
                double uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR = 0.0;
                double uniqueMRsInTestAndNotInTrainAllPredWordBRC = 0.0;

                detailedRes = "";
                ArrayList<DatasetInstance> abstractMRList = new ArrayList<>();
                HashSet<String> reportedAbstractMRs = new HashSet<>();
                for (DatasetInstance di : getTestingData()) {
                    if (!reportedAbstractMRs.contains(di.getMeaningRepresentation().getAbstractMR())) {
                        reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                        boolean isInTraining = false;
                        for (DatasetInstance di2 : getTrainingData()) {
                            if (di2.getMeaningRepresentation().getAbstractMR().equals(di.getMeaningRepresentation().getAbstractMR())) {
                                isInTraining = true;
                            }
                        }
                        if (!isInTraining) {
                            for (DatasetInstance di2 : getValidationData()) {
                                if (di2.getMeaningRepresentation().getAbstractMR().equals(di.getMeaningRepresentation().getAbstractMR())) {
                                    isInTraining = true;
                                }
                            }
                        }
                        if (!isInTraining) {
                            abstractMRList.add(di);
                        }
                    }
                }
                for (DatasetInstance di : abstractMRList) {
                    Double bestROUGE = -100.0;
                    Double bestBLEU = -100.0;
                    Double bestCover = -100.0;
                    Double bestHarmonicMean = -100.0;
                    String predictedString = predictedWordSequences_overAllPredicates.get(di.getMeaningRepresentation().getMRstr());
                    reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                    double maxRouge = 0.0;
                    String pWS = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                    for (String ref : finalReferencesWordSequences.get(di.getMeaningRepresentation().getMRstr())) {
                        double rouge = Rouge.ROUGE_N(pWS, ref, 4);
                        if (rouge > maxRouge) {
                            maxRouge = rouge;
                        }
                    }

                    double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(pWS, finalReferencesWordSequences.get(di.getMeaningRepresentation().getMRstr()), 4);
                    double cover = 1.0 - attrCoverage.get(predictedString);
                    double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                    if (harmonicMean > bestHarmonicMean) {
                        bestROUGE = maxRouge;
                        bestBLEU = BLEUSmooth;
                        bestCover = cover;
                        bestHarmonicMean = harmonicMean;
                    }

                    uniqueMRsInTestAndNotInTrainAllPredWordBLEU += bestBLEU;
                    uniqueMRsInTestAndNotInTrainAllPredWordROUGE += bestROUGE;
                    uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR += bestCover;
                    uniqueMRsInTestAndNotInTrainAllPredWordBRC += bestHarmonicMean;
                }
                uniqueMRsInTestAndNotInTrainAllPredWordBLEU /= abstractMRList.size();
                uniqueMRsInTestAndNotInTrainAllPredWordROUGE /= abstractMRList.size();
                uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR /= abstractMRList.size();
                uniqueMRsInTestAndNotInTrainAllPredWordBRC /= abstractMRList.size();
                System.out.println("WEN UNIQUE (NOT IN TRAIN) WORD ALL PRED BLEU: \t" + uniqueMRsInTestAndNotInTrainAllPredWordBLEU);
                System.out.println("WEN UNIQUE (NOT IN TRAIN) WORD ALL PRED ROUGE: \t" + uniqueMRsInTestAndNotInTrainAllPredWordROUGE);
                System.out.println("WEN UNIQUE (NOT IN TRAIN) WORD ALL PRED COVERAGE ERROR: \t" + (1.0 - uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR));
                System.out.println("WEN UNIQUE (NOT IN TRAIN) WORD ALL PRED BRC: \t" + uniqueMRsInTestAndNotInTrainAllPredWordBRC);

                abstractMRList.forEach((di) -> {
                    System.out.println(di.getMeaningRepresentation().getAbstractMR() + "\t" + di.getMeaningRepresentation().getMRstr());
                });
                System.out.println("TOTAL SET SIZE: \t" + abstractMRList.size());
                //System.out.println(abstractMRList);
                //System.out.println(detailedRes);
                //System.out.println(abstractMRList);  
                //System.out.println(detailedRes);
                System.out.println("///////////////////");

                ArrayList<String> bestPredictedStrings = new ArrayList<>();
                ArrayList<String> bestPredictedStringsMRs = new ArrayList<>();
                double uniqueAllPredWordBLEU = 0.0;
                double uniqueAllPredWordROUGE = 0.0;
                double uniqueAllPredWordCOVERAGEERR = 0.0;
                double uniqueAllPredWordBRC = 0.0;

                reportedAbstractMRs = new HashSet<>();
                for (DatasetInstance di : getTestingData()) {
                    //for (String predictedString : predictedWordSequences_overAllPredicates.get(di)) {
                    if (!reportedAbstractMRs.contains(di.getMeaningRepresentation().getAbstractMR())) {
                        String bestPredictedString = "";
                        Double bestROUGE = -100.0;
                        Double bestBLEU = -100.0;
                        Double bestCover = -100.0;
                        Double bestHarmonicMean = -100.0;
                        String predictedString = predictedWordSequences_overAllPredicates.get(di.getMeaningRepresentation().getMRstr());
                        reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                        double maxRouge = 0.0;
                        String pWS = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                        for (String ref : finalReferencesWordSequences.get(di.getMeaningRepresentation().getMRstr())) {
                            double rouge = Rouge.ROUGE_N(pWS, ref, 4);
                            if (rouge > maxRouge) {
                                maxRouge = rouge;
                            }
                        }

                        double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(pWS, finalReferencesWordSequences.get(di.getMeaningRepresentation().getMRstr()), 4);
                        double cover = 1.0 - attrCoverage.get(predictedString);
                        double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                        if (harmonicMean > bestHarmonicMean) {
                            bestPredictedString = predictedString;
                            bestROUGE = maxRouge;
                            bestBLEU = BLEUSmooth;
                            bestCover = cover;
                            bestHarmonicMean = harmonicMean;
                        }
                        bestPredictedStrings.add(bestPredictedString);
                        bestPredictedStringsMRs.add(di.getMeaningRepresentation().getMRstr());

                        uniqueAllPredWordBLEU += bestBLEU;
                        uniqueAllPredWordROUGE += bestROUGE;
                        uniqueAllPredWordCOVERAGEERR += bestCover;
                        uniqueAllPredWordBRC += bestHarmonicMean;
                    }
                    //}
                }
                uniqueAllPredWordBLEU /= reportedAbstractMRs.size();
                uniqueAllPredWordROUGE /= reportedAbstractMRs.size();
                uniqueAllPredWordCOVERAGEERR /= reportedAbstractMRs.size();
                uniqueAllPredWordBRC /= reportedAbstractMRs.size();
                System.out.println("WEN UNIQUE WORD ALL PRED BLEU: \t" + uniqueAllPredWordBLEU);
                System.out.println("WEN UNIQUE WORD ALL PRED ROUGE: \t" + uniqueAllPredWordROUGE);
                System.out.println("WEN UNIQUE WORD ALL PRED COVERAGE ERROR: \t" + (1.0 - uniqueAllPredWordCOVERAGEERR));
                System.out.println("WEN UNIQUE WORD ALL PRED BRC: \t" + uniqueAllPredWordBRC);
                System.out.println(detailedRes);
                System.out.println("WEN TOTAL: \t" + reportedAbstractMRs.size());

                ////////////////////////
                for (String pred : getPredicates()) {
                    detailedRes = "";
                    bestPredictedStrings = new ArrayList<>();
                    bestPredictedStringsMRs = new ArrayList<>();
                    double uniquePredWordBLEU = 0.0;
                    double uniquePredWordROUGE = 0.0;
                    double uniquePredWordCOVERAGEERR = 0.0;
                    double uniquePredWordBRC = 0.0;

                    reportedAbstractMRs = new HashSet<>();
                    for (DatasetInstance di : getTestingData()) {
                        if (di.getMeaningRepresentation().getPredicate().equals(pred)
                                && !reportedAbstractMRs.contains(di.getMeaningRepresentation().getAbstractMR())) {
                            String bestPredictedString = "";
                            Double bestROUGE = -100.0;
                            Double bestBLEU = -100.0;
                            Double bestCover = -100.0;
                            Double bestHarmonicMean = -100.0;

                            String predictedString = predictedWordSequences_overAllPredicates.get(di.getMeaningRepresentation().getMRstr());
                            reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                            double maxRouge = 0.0;
                            String pWS = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                            for (String ref : finalReferencesWordSequences.get(di.getMeaningRepresentation().getMRstr())) {
                                double rouge = Rouge.ROUGE_N(pWS, ref, 4);
                                if (rouge > maxRouge) {
                                    maxRouge = rouge;
                                }
                            }

                            double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(pWS, finalReferencesWordSequences.get(di.getMeaningRepresentation().getMRstr()), 4);
                            double cover = 1.0 - attrCoverage.get(predictedString);
                            double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                            if (harmonicMean > bestHarmonicMean) {
                                bestPredictedString = predictedString;
                                bestROUGE = maxRouge;
                                bestBLEU = BLEUSmooth;
                                bestCover = cover;
                                bestHarmonicMean = harmonicMean;
                            }
                            bestPredictedStrings.add(bestPredictedString);
                            bestPredictedStringsMRs.add(di.getMeaningRepresentation().getMRstr());

                            uniquePredWordBLEU += bestBLEU;
                            uniquePredWordROUGE += bestROUGE;
                            uniquePredWordCOVERAGEERR += bestCover;
                            uniquePredWordBRC += bestHarmonicMean;
                        }
                    }

                    uniquePredWordBLEU /= reportedAbstractMRs.size();
                    uniquePredWordROUGE /= reportedAbstractMRs.size();
                    uniquePredWordCOVERAGEERR /= reportedAbstractMRs.size();
                    uniquePredWordBRC /= reportedAbstractMRs.size();
                    System.out.println("WEN UNIQUE WORD " + pred + " BLEU: \t" + uniquePredWordBLEU);
                    System.out.println("WEN UNIQUE WORD " + pred + " ROUGE: \t" + uniquePredWordROUGE);
                    System.out.println("WEN UNIQUE WORD " + pred + " COVERAGE ERROR: \t" + (1.0 - uniquePredWordCOVERAGEERR));
                    System.out.println("WEN UNIQUE WORD " + pred + " BRC: \t" + uniquePredWordBRC);
                    System.out.println(detailedRes);
                    System.out.println("WEN TOTAL " + pred + ": \t" + reportedAbstractMRs.size());
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
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
    @Override
    public Double evaluateGeneration(HashMap<String, JAROW> classifierAttrs, HashMap<String, HashMap<String, JAROW>> classifierWords, ArrayList<DatasetInstance> testingData, int epoch) {
        System.out.println("Evaluate argument generation ");

        ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
        HashMap<DatasetInstance, ArrayList<Action>> generationActions = new HashMap<>();
        ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
        HashMap<DatasetInstance, ArrayList<String>> finalReferencesWordSequences = new HashMap<>();
        HashMap<DatasetInstance, String> predictedWordSequences_overAllPredicates = new HashMap<>();
        ArrayList<String> allPredictedWordSequences = new ArrayList<>();
        ArrayList<String> allPredictedMRStr = new ArrayList<>();
        ArrayList<ArrayList<String>> allPredictedReferences = new ArrayList<>();
        HashMap<String, Double> attrCoverage = new HashMap<>();

        HashMap<String, HashSet<String>> abstractMRsToMRs = new HashMap<>();

        for (DatasetInstance di : testingData) {
            String predicate = di.getMeaningRepresentation().getPredicate();
            ArrayList<Action> predictedActionList = new ArrayList<>();
            ArrayList<Action> predictedWordList = new ArrayList<>();

            //PHRASE GENERATION EVALUATION
            String predictedAttr = "";
            ArrayList<String> predictedAttrValues = new ArrayList<>();

            HashSet<String> attrValuesToBeMentioned = new HashSet<>();
            HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
            for (String attribute : di.getMeaningRepresentation().getAttributeValues().keySet()) {
                for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                    attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
                }
            }
            if (attrValuesToBeMentioned.isEmpty()) {
                attrValuesToBeMentioned.add("empty=empty");
            }
            while (!predictedAttr.equals(Action.TOKEN_END) && predictedAttrValues.size() < getMaxContentSequenceLength()) {
                if (!predictedAttr.isEmpty()) {
                    attrValuesToBeMentioned.remove(predictedAttr);
                }
                if (!attrValuesToBeMentioned.isEmpty()) {
                    Instance attrTrainingVector = createContentInstance(predicate, "@TOK@", predictedAttrValues, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), getAvailableContentActions());

                    if (attrTrainingVector != null) {
                        Prediction predictAttr = classifierAttrs.get(predicate).predict(attrTrainingVector);
                        if (predictAttr.getLabel() != null) {
                            predictedAttr = predictAttr.getLabel().trim();

                            if (!classifierAttrs.get(predicate).getCurrentWeightVectors().keySet().containsAll(di.getMeaningRepresentation().getAttributeValues().keySet())) {
                                //System.out.println("MR ATTR NOT IN CLASSIFIERS");
                                //System.out.println(classifierAttrs.get(predicate).getCurrentWeightVectors().keySet());
                            }
                            String predictedValue = "";
                            if (!predictedAttr.equals(Action.TOKEN_END)) {
                                predictedValue = chooseNextValue(predictedAttr, attrValuesToBeMentioned);

                                HashSet<String> rejectedAttrs = new HashSet<>();
                                while (predictedValue.isEmpty() && (!predictedAttr.equals(Action.TOKEN_END) || (predictedAttrValues.isEmpty() && classifierAttrs.get(predicate).getCurrentWeightVectors().keySet().containsAll(di.getMeaningRepresentation().getAttributeValues().keySet())))) {
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
            for (String attribute : di.getMeaningRepresentation().getAttributeValues().keySet()) {
                for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                    attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
                }
                valuesToBeMentioned.put(attribute, new ArrayList<>(di.getMeaningRepresentation().getAttributeValues().get(attribute)));
            }
            if (attrValuesToBeMentioned.isEmpty()) {
                attrValuesToBeMentioned.add("empty=empty");
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
                            ArrayList<String> nextAttributesForInstance = new ArrayList<>(predictedAttrValues.subList(a + 1, predictedAttrValues.size()));
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
                            while (!predictedWord.equals(Action.TOKEN_END) && predictedWordList.size() < getMaxWordSequenceLength()) {
                                ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                                for (int i = 0; i < predictedAttributes.size() - 1; i++) {
                                    predictedAttributesForInstance.add(predictedAttributes.get(i));
                                }
                                if (!predictedAttributes.get(predictedAttributes.size() - 1).equals(attrValue)) {
                                    predictedAttributesForInstance.add(predictedAttributes.get(predictedAttributes.size() - 1));
                                }
                                Instance wordTrainingVector = createWordInstance(predicate, new Action("@TOK@", attrValue), predictedAttributesForInstance, predictedActionList, nextAttributesForInstance, attrValuesAlreadyMentioned, attrValuesToBeMentioned, isValueMentioned, getAvailableWordActions().get(predicate));

                                if (wordTrainingVector != null
                                        && classifierWords.get(predicate) != null) {
                                    if (classifierWords.get(predicate).get(attribute) != null) {
                                        Prediction predictWord = classifierWords.get(predicate).get(attribute).predict(wordTrainingVector);
                                        if (predictWord.getLabel() != null) {
                                            predictedWord = predictWord.getLabel().trim();
                                            while (predictedWord.equals(Action.TOKEN_END) && !predictedActionList.isEmpty() && predictedActionList.get(predictedActionList.size() - 1).getWord().equals(Action.TOKEN_END)) {
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
                                            if (!predictedWord.equals(Action.TOKEN_START)
                                                    && !predictedWord.equals(Action.TOKEN_END)) {
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
                                            String valueToCheck = valueTBM;
                                            if (valueToCheck.equals("no")
                                                    || valueToCheck.equals("yes")
                                                    || valueToCheck.equals("yes or no")
                                                    || valueToCheck.equals("none")
                                                    //|| valueToCheck.equals("dont_care")
                                                    || valueToCheck.equals("empty")) {
                                                if (attribute.contains("=")) {
                                                    valueToCheck = attribute.replace("=", ":");
                                                } else {
                                                    valueToCheck = attribute + ":" + valueTBM;
                                                }
                                            }
                                            if (!valueToCheck.equals("empty:empty")
                                                    && getValueAlignments().containsKey(valueToCheck)) {
                                                for (ArrayList<String> alignedStr : getValueAlignments().get(valueToCheck).keySet()) {
                                                    if (endsWith(subPhrase, alignedStr)) {
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
                                                        //|| valueToCheck.equals("dont_care")
                                                        || valueToCheck.equals("empty")) {
                                                    valueToCheck = attrValueTBM.replace("=", ":");
                                                }
                                                if (!valueToCheck.equals("empty:empty")
                                                        && getValueAlignments().containsKey(valueToCheck)) {
                                                    for (ArrayList<String> alignedStr : getValueAlignments().get(valueToCheck).keySet()) {
                                                        if (endsWith(subPhrase, alignedStr)) {
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
                            if (predictedWordList.size() >= getMaxWordSequenceLength()
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
            predictedAttrValues.forEach((attributeValuePair) -> {
                predictedAttrs.add(attributeValuePair.split("=")[0]);
            });

            String predictedWordSequence = postProcessWordSequence(di, predictedActionList);

            ArrayList<String> predictedAttrList = getPredictedAttrList(predictedActionList);
            if (attrValuesToBeMentionedCopy.size() != 0.0) {
                double missingAttrs = 0.0;
                missingAttrs = attrValuesToBeMentionedCopy.stream().filter((attr) -> (!predictedAttrList.contains(attr))).map((_item) -> 1.0).reduce(missingAttrs, (accumulator, _item) -> accumulator + _item);
                double attrSize = attrValuesToBeMentionedCopy.size();
                attrCoverage.put(predictedWordSequence, missingAttrs / attrSize);
            }

            allPredictedWordSequences.add(predictedWordSequence);
            allPredictedMRStr.add(di.getMeaningRepresentation().getMRstr());
            predictedWordSequences_overAllPredicates.put(di, predictedWordSequence);

            if (!abstractMRsToMRs.containsKey(di.getMeaningRepresentation().getAbstractMR())) {
                abstractMRsToMRs.put(di.getMeaningRepresentation().getAbstractMR(), new HashSet<String>());
            }
            abstractMRsToMRs.get(di.getMeaningRepresentation().getAbstractMR()).add(di.getMeaningRepresentation().getMRstr());

            Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedWordSequence.toLowerCase()));
            ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
            generations.add(tran);
            generationActions.put(di, predictedActionList);

            ArrayList<Sequence<IString>> references = new ArrayList<>();
            ArrayList<String> referencesStrings = new ArrayList<>();

            if (getPerformEvaluationOn().equals("valid") || getPerformEvaluationOn().equals("train")) {
                for (String ref : di.getEvaluationReferences()) {
                    referencesStrings.add(ref);
                    references.add(IStrings.tokenize(NISTTokenizer.tokenize(ref)));
                }
            } else {
                references = wenEvaluationReferenceSequences.get(di.getMeaningRepresentation().getMRstr());
                referencesStrings = wenEvaluationReferences.get(di.getMeaningRepresentation().getMRstr());
                if (references == null) {
                    references = new ArrayList<>();
                    referencesStrings = new ArrayList<>();
                    for (String ref : di.getEvaluationReferences()) {
                        referencesStrings.add(ref);
                        references.add(IStrings.tokenize(NISTTokenizer.tokenize(ref)));
                    }
                }
            }
            allPredictedReferences.add(referencesStrings);
            finalReferencesWordSequences.put(di, referencesStrings);
            finalReferences.add(references);
        }

        BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
        Double bleuScore = BLEU.score(generations);

        double finalCoverageError = 0.0;
        finalCoverageError = attrCoverage.values().stream().map((c) -> c).reduce(finalCoverageError, (accumulator, _item) -> accumulator + _item);
        finalCoverageError /= attrCoverage.size();
        for (int i = 0; i < allPredictedWordSequences.size(); i++) {
            double maxRouge = 0.0;
            String predictedWordSequence = allPredictedWordSequences.get(i).replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
            for (String ref : allPredictedReferences.get(i)) {
                double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                if (rouge > maxRouge) {
                    maxRouge = rouge;
                }
            }
            //System.out.println(allPredictedMRStr.get(i) + "\t" + maxRouge + "\t" + allPredictedWordSequences.get(i) + "\t" + refs);
        }

        double avgRougeScore = 0.0;
        String detailedRes = "";

        avgRougeScore = testingData.stream().map((di) -> {
            double maxRouge = 0.0;
            if (!finalReferencesWordSequences.containsKey(di)) {
                System.out.println(di.getMeaningRepresentation().getAbstractMR());
            }
            String predictedWordSequence = predictedWordSequences_overAllPredicates.get(di).replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
            for (String ref : finalReferencesWordSequences.get(di)) {
                double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                if (rouge > maxRouge) {
                    maxRouge = rouge;
                }
            }
            return maxRouge;
        }).map((maxRouge) -> maxRouge).reduce(avgRougeScore, (accumulator, _item) -> accumulator + _item);
        System.out.println("BLEU: \t" + bleuScore);
        //System.out.println("g: " + generations);
        //System.out.println("attr: " + predictedAttrLists);
        //System.out.println("BLEU smooth: \t" + bleuSmoothScore);
        //System.out.println("g: " + generations);
        //System.out.println("attr: " + predictedAttrLists);
        //System.out.println("BLEU smooth: \t" + bleuSmoothScore);
        System.out.println("ROUGE: \t" + (avgRougeScore / allPredictedWordSequences.size()));
        System.out.println("COVERAGE ERROR: \t" + finalCoverageError);
        System.out.println("BRC: \t" + ((avgRougeScore / allPredictedWordSequences.size()) + bleuScore + (1.0 - finalCoverageError)) / 3.0);

        if (isCalculateResultsPerPredicate()) {
            ////////////////////////
            //ArrayList<String> bestPredictedStrings = new ArrayList<>();
            //ArrayList<String> bestPredictedStringsMRs = new ArrayList<>();
            double uniqueMRsInTestAndNotInTrainAllPredWordBLEU = 0.0;
            double uniqueMRsInTestAndNotInTrainAllPredWordROUGE = 0.0;
            double uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR = 0.0;
            double uniqueMRsInTestAndNotInTrainAllPredWordBRC = 0.0;

            detailedRes = "";
            ArrayList<DatasetInstance> abstractMRList = new ArrayList<>();
            HashSet<String> reportedAbstractMRs = new HashSet<>();
            testingData.stream().filter((di) -> (!reportedAbstractMRs.contains(di.getMeaningRepresentation().getAbstractMR()))).map((di) -> {
                reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                return di;
            }).forEachOrdered((di) -> {
                boolean isInTraining = false;
                for (DatasetInstance di2 : getTrainingData()) {
                    if (di2.getMeaningRepresentation().getAbstractMR().equals(di.getMeaningRepresentation().getAbstractMR())) {
                        isInTraining = true;
                    }
                }
                if (!isInTraining) {
                    for (DatasetInstance di2 : getValidationData()) {
                        if (di2.getMeaningRepresentation().getAbstractMR().equals(di.getMeaningRepresentation().getAbstractMR())) {
                            isInTraining = true;
                        }
                    }
                }
                if (!isInTraining) {
                    abstractMRList.add(di);
                }
            });
            for (DatasetInstance di : abstractMRList) {
                Double bestROUGE = -100.0;
                Double bestBLEU = -100.0;
                Double bestCover = -100.0;
                Double bestHarmonicMean = -100.0;
                String predictedString = predictedWordSequences_overAllPredicates.get(di);
                reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                double maxRouge = 0.0;
                String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                for (String ref : finalReferencesWordSequences.get(di)) {
                    double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                    if (rouge > maxRouge) {
                        maxRouge = rouge;
                    }
                }

                double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(di), 4);
                double cover = 1.0 - attrCoverage.get(predictedString);
                double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                if (harmonicMean > bestHarmonicMean) {
                    bestROUGE = maxRouge;
                    bestBLEU = BLEUSmooth;
                    bestCover = cover;
                    bestHarmonicMean = harmonicMean;
                }

                uniqueMRsInTestAndNotInTrainAllPredWordBLEU += bestBLEU;
                uniqueMRsInTestAndNotInTrainAllPredWordROUGE += bestROUGE;
                uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR += bestCover;
                uniqueMRsInTestAndNotInTrainAllPredWordBRC += bestHarmonicMean;
            }
            uniqueMRsInTestAndNotInTrainAllPredWordBLEU /= abstractMRList.size();
            uniqueMRsInTestAndNotInTrainAllPredWordROUGE /= abstractMRList.size();
            uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR /= abstractMRList.size();
            uniqueMRsInTestAndNotInTrainAllPredWordBRC /= abstractMRList.size();
            System.out.println("UNIQUE (NOT IN TRAIN) WORD ALL PRED BLEU: \t" + uniqueMRsInTestAndNotInTrainAllPredWordBLEU);
            System.out.println("UNIQUE (NOT IN TRAIN) WORD ALL PRED ROUGE: \t" + uniqueMRsInTestAndNotInTrainAllPredWordROUGE);
            System.out.println("UNIQUE (NOT IN TRAIN) WORD ALL PRED COVERAGE ERROR: \t" + (1.0 - uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR));
            System.out.println("UNIQUE (NOT IN TRAIN) WORD ALL PRED BRC: \t" + uniqueMRsInTestAndNotInTrainAllPredWordBRC);

            abstractMRList.forEach((di) -> {
                System.out.println(di.getMeaningRepresentation().getAbstractMR() + "\t" + predictedWordSequences_overAllPredicates.get(di));
            });
            System.out.println("TOTAL SET SIZE: \t" + abstractMRList.size());
            //System.out.println(abstractMRList);  
            //System.out.println(detailedRes);
        }
        ArrayList<String> bestPredictedStrings = new ArrayList<>();
        ArrayList<String> bestPredictedStringsMRs = new ArrayList<>();
        double uniqueAllPredWordBLEU = 0.0;
        double uniqueAllPredWordROUGE = 0.0;
        double uniqueAllPredWordCOVERAGEERR = 0.0;
        double uniqueAllPredWordBRC = 0.0;

        HashSet<String> reportedAbstractMRs = new HashSet<>();
        for (DatasetInstance di : testingData) {
            if (!reportedAbstractMRs.contains(di.getMeaningRepresentation().getAbstractMR())) {
                String bestPredictedString = "";
                Double bestROUGE = -100.0;
                Double bestBLEU = -100.0;
                Double bestCover = -100.0;
                Double bestHarmonicMean = -100.0;
                String predictedString = predictedWordSequences_overAllPredicates.get(di);
                reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                double maxRouge = 0.0;
                String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                for (String ref : finalReferencesWordSequences.get(di)) {
                    double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                    if (rouge > maxRouge) {
                        maxRouge = rouge;
                    }
                }

                double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(di), 4);
                double cover = 1.0 - attrCoverage.get(predictedString);
                double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                if (harmonicMean > bestHarmonicMean) {
                    bestPredictedString = predictedString;
                    bestROUGE = maxRouge;
                    bestBLEU = BLEUSmooth;
                    bestCover = cover;
                    bestHarmonicMean = harmonicMean;
                }
                bestPredictedStrings.add(bestPredictedString);
                bestPredictedStringsMRs.add(di.getMeaningRepresentation().getMRstr());

                uniqueAllPredWordBLEU += bestBLEU;
                uniqueAllPredWordROUGE += bestROUGE;
                uniqueAllPredWordCOVERAGEERR += bestCover;
                uniqueAllPredWordBRC += bestHarmonicMean;
            }
            //}
        }
        if (isCalculateResultsPerPredicate()) {
            uniqueAllPredWordBLEU /= reportedAbstractMRs.size();
            uniqueAllPredWordROUGE /= reportedAbstractMRs.size();
            uniqueAllPredWordCOVERAGEERR /= reportedAbstractMRs.size();
            uniqueAllPredWordBRC /= reportedAbstractMRs.size();
            System.out.println("UNIQUE WORD ALL PRED BLEU: \t" + uniqueAllPredWordBLEU);
            System.out.println("UNIQUE WORD ALL PRED ROUGE: \t" + uniqueAllPredWordROUGE);
            System.out.println("UNIQUE WORD ALL PRED COVERAGE ERROR: \t" + (1.0 - uniqueAllPredWordCOVERAGEERR));
            System.out.println("UNIQUE WORD ALL PRED BRC: \t" + uniqueAllPredWordBRC);
            System.out.println(detailedRes);
            System.out.println("TOTAL: \t" + reportedAbstractMRs.size());

            ////////////////////////
            for (String predicate : getPredicates()) {
                detailedRes = "";
                bestPredictedStrings = new ArrayList<>();
                bestPredictedStringsMRs = new ArrayList<>();
                double uniquePredWordBLEU = 0.0;
                double uniquePredWordROUGE = 0.0;
                double uniquePredWordCOVERAGEERR = 0.0;
                double uniquePredWordBRC = 0.0;

                reportedAbstractMRs = new HashSet<>();
                for (DatasetInstance di : testingData) {
                    if (di.getMeaningRepresentation().getPredicate().equals(predicate)
                            && !reportedAbstractMRs.contains(di.getMeaningRepresentation().getAbstractMR())) {
                        String bestPredictedString = "";
                        Double bestROUGE = -100.0;
                        Double bestBLEU = -100.0;
                        Double bestCover = -100.0;
                        Double bestHarmonicMean = -100.0;

                        String predictedString = predictedWordSequences_overAllPredicates.get(di);
                        reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                        double maxRouge = 0.0;
                        String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                        for (String ref : finalReferencesWordSequences.get(di)) {
                            double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                            if (rouge > maxRouge) {
                                maxRouge = rouge;
                            }
                        }

                        double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(di), 4);
                        double cover = 1.0 - attrCoverage.get(predictedString);
                        double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                        if (harmonicMean > bestHarmonicMean) {
                            bestPredictedString = predictedString;
                            bestROUGE = maxRouge;
                            bestBLEU = BLEUSmooth;
                            bestCover = cover;
                            bestHarmonicMean = harmonicMean;
                        }
                        bestPredictedStrings.add(bestPredictedString);
                        bestPredictedStringsMRs.add(di.getMeaningRepresentation().getMRstr());

                        uniquePredWordBLEU += bestBLEU;
                        uniquePredWordROUGE += bestROUGE;
                        uniquePredWordCOVERAGEERR += bestCover;
                        uniquePredWordBRC += bestHarmonicMean;
                    }
                }

                uniquePredWordBLEU /= reportedAbstractMRs.size();
                uniquePredWordROUGE /= reportedAbstractMRs.size();
                uniquePredWordCOVERAGEERR /= reportedAbstractMRs.size();
                uniquePredWordBRC /= reportedAbstractMRs.size();
                System.out.println("UNIQUE WORD " + predicate + " BLEU: \t" + uniquePredWordBLEU);
                System.out.println("UNIQUE WORD " + predicate + " ROUGE: \t" + uniquePredWordROUGE);
                System.out.println("UNIQUE WORD " + predicate + " COVERAGE ERROR: \t" + (1.0 - uniquePredWordCOVERAGEERR));
                System.out.println("UNIQUE WORD " + predicate + " BRC: \t" + uniquePredWordBRC);
                System.out.println(detailedRes);
                System.out.println("TOTAL " + predicate + ": \t" + reportedAbstractMRs.size());
            }
        }

        if (isCalculateResultsPerPredicate()) {
            BufferedWriter bw = null;
            File f = null;
            try {
                f = new File("results/random_SFX" + getDataset() + "TextsAfter" + (epoch) + "_" + JLOLS.sentenceCorrectionFurtherSteps + "_" + JLOLS.p + "epochsTESTINGDATA.txt");
            } catch (NullPointerException e) {
            }

            try {
                bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
            } catch (FileNotFoundException e) {
            }

            try {
                bw.write("BLEU:" + bleuScore);
                bw.write("\n");
            } catch (IOException e) {
            }
            for (int i = 0; i < bestPredictedStrings.size(); i++) {
                try {
                    String mr = bestPredictedStringsMRs.get(i);
                    bw.write("MR;" + mr.replaceAll(";", ",") + ";");
                    if (getDataset().equals("hotel")) {
                        bw.write("LOLS_SFHOT;");
                    } else {
                        bw.write("LOLS_SFRES;");
                    }

                    bw.write("\n");
                } catch (IOException e) {
                }
            }

            try {
                bw.close();
            } catch (IOException e) {
            }
        }
        return bleuScore;
    }

    /**
     * Populates the predicate, attribute, attribute/value pair, and value alignment collections
     * @param dataFile The dataset file.
     */
    public void createLists(File dataFile) {
        try {
            // Initialize the collections
            setPredicates(new ArrayList<>());
            setAttributes(new HashMap<>());
            setAttributeValuePairs(new HashMap<>());
            setValueAlignments(new HashMap<>());

            // Obtain the dataset portion of the file
            String dataPart = new String();
            boolean begin = false;
            try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {
                String s;
                while ((s = br.readLine()) != null) {
                    if (s.startsWith("[")) {
                        begin = true;
                    }
                    if (begin) {
                        dataPart += s;
                    }
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            }

            // Parse the dataset with JSON
            JSONArray overArray = new JSONArray(dataPart);
            for (int o = 0; o < overArray.length(); o++) {
                // "dial" notes each seperate dialog
                JSONArray arr = overArray.getJSONObject(o).getJSONArray("dial");
                for (int i = 0; i < arr.length(); i++) {
                    String MRstr;
                    String ref;
                    // "dact" notes every meaning representation
                    MRstr = arr.getJSONObject(i).getJSONObject("S").getString("dact");
                    // "ref" notes every corresponding reference
                    ref = arr.getJSONObject(i).getJSONObject("S").getString("ref").replaceAll("-s", "s");

                    //We split some composite words (based on Wen et al's (2016) code)
                    ref = (" " + ref + " ").replaceAll(" it's ", " it is ")
                            .replaceAll(" don't ", " do not ")
                            .replaceAll(" doesn't ", " does not ")
                            .replaceAll(" didn't ", " did not ")
                            .replaceAll(" you'd ", " you would ")
                            .replaceAll(" you're ", " you are ")
                            .replaceAll(" you'll ", " you will ")
                            .replaceAll(" i'm ", " i am ")
                            .replaceAll(" they're ", " they are ")
                            .replaceAll(" that's ", " that is ")
                            .replaceAll(" what's ", " what is ")
                            .replaceAll(" couldn't ", " could not ")
                            .replaceAll(" i've ", " i have ")
                            .replaceAll(" we've ", " we have ")
                            .replaceAll(" can't ", " cannot ")
                            .replaceAll(" i'd ", " i would ")
                            .replaceAll(" i'd ", " i would ")
                            .replaceAll(" aren't ", " are not ")
                            .replaceAll(" isn't ", " is not ")
                            .replaceAll(" wasn't ", " was not ")
                            .replaceAll(" weren't ", " were not ")
                            .replaceAll(" won't ", " will not ")
                            .replaceAll(" there's ", " there is ")
                            .replaceAll(" there're ", " there are ")
                            .replaceAll(" \\. \\. ", " \\. ")
                            .replaceAll(" restaurants ", " restaurant -s ")
                            .replaceAll(" hotels ", " hotel -s ")
                            .replaceAll(" laptops ", " laptop -s ")
                            .replaceAll(" cheaper ", " cheap -er ")
                            .replaceAll(" dinners ", " dinner -s ")
                            .replaceAll(" lunches ", " lunch -s ")
                            .replaceAll(" breakfasts ", " breakfast -s ")
                            .replaceAll(" expensively ", " expensive -ly ")
                            .replaceAll(" moderately ", " moderate -ly ")
                            .replaceAll(" cheaply ", " cheap -ly ")
                            .replaceAll(" prices ", " price -s ")
                            .replaceAll(" places ", " place -s ")
                            .replaceAll(" venues ", " venue -s ")
                            .replaceAll(" ranges ", " range -s ")
                            .replaceAll(" meals ", " meal -s ")
                            .replaceAll(" locations ", " location -s ")
                            .replaceAll(" areas ", " area -s ")
                            .replaceAll(" policies ", " policy -s ")
                            .replaceAll(" children ", " child -s ")
                            .replaceAll(" kids ", " kid -s ")
                            .replaceAll(" kidfriendly ", " kid friendly ")
                            .replaceAll(" cards ", " card -s ")
                            .replaceAll(" st ", " street ")
                            .replaceAll(" ave ", " avenue ")
                            .replaceAll(" upmarket ", " expensive ")
                            .replaceAll(" inpricey ", " cheap ")
                            .replaceAll(" inches ", " inch -s ")
                            .replaceAll(" uses ", " use -s ")
                            .replaceAll(" dimensions ", " dimension -s ")
                            .replaceAll(" driverange ", " drive range ")
                            .replaceAll(" includes ", " include -s ")
                            .replaceAll(" computers ", " computer -s ")
                            .replaceAll(" machines ", " machine -s ")
                            .replaceAll(" ecorating ", " eco rating ")
                            .replaceAll(" families ", " family -s ")
                            .replaceAll(" ratings ", " rating -s ")
                            .replaceAll(" constraints ", " constraint -s ")
                            .replaceAll(" pricerange ", " price range ")
                            .replaceAll(" batteryrating ", " battery rating ")
                            .replaceAll(" requirements ", " requirement -s ")
                            .replaceAll(" drives ", " drive -s ")
                            .replaceAll(" specifications ", " specification -s ")
                            .replaceAll(" weightrange ", " weight range ")
                            .replaceAll(" harddrive ", " hard drive ")
                            .replaceAll(" batterylife ", " battery life ")
                            .replaceAll(" businesses ", " business -s ")
                            .replaceAll(" hours ", " hour -s ")
                            .replaceAll(" accessories ", " accessory -s ")
                            .replaceAll(" ports ", " port -s ")
                            .replaceAll(" televisions ", " television -s ")
                            .replaceAll(" restrictions ", " restriction -s ")
                            .replaceAll(" extremely ", " extreme -ly ")
                            .replaceAll(" actually ", " actual -ly ")
                            .replaceAll(" typically ", " typical -ly ")
                            .replaceAll(" drivers ", " driver -s ")
                            .replaceAll(" teh ", " the ")
                            .replaceAll(" definitely ", " definite -ly ")
                            .replaceAll(" factors ", " factor -s ")
                            .replaceAll(" truly ", " true -ly ")
                            .replaceAll(" mostly ", " most -ly ")
                            .replaceAll(" nicely ", " nice -ly ")
                            .replaceAll(" surely ", " sure -ly ")
                            .replaceAll(" certainly ", " certain -ly ")
                            .replaceAll(" totally ", " total -ly ")
                            .replaceAll(" \\# ", " number ")
                            .replaceAll(" \\& ", " and ")
                            .replaceAll(" avenue ", " ave ")
                            .replaceAll(" -s ", " s ").trim();

                    // If the MR concerns one of the following predicates, and a ref is available
                    if ((MRstr.startsWith("inform(")
                            || MRstr.startsWith("inform_only")
                            || MRstr.startsWith("inform_no_match(")
                            || MRstr.startsWith("?confirm(")
                            || MRstr.startsWith("?select(")
                            || MRstr.startsWith("?request(")
                            || MRstr.startsWith("?reqmore(")
                            || MRstr.startsWith("goodbye("))
                            && !ref.isEmpty()) {
                        // Obtain the predicate
                        String predicate = MRstr.substring(0, MRstr.indexOf('('));
                        if (!getPredicates().contains(predicate) && predicate != null) {
                            getPredicates().add(predicate);

                            if (!getAttributes().containsKey(predicate)) {
                                getAttributes().put(predicate, new HashSet<String>());
                            }
                            if (!getDatasetInstances().containsKey(predicate)) {
                                getDatasetInstances().put(predicate, new ArrayList<DatasetInstance>());
                            }
                        }

                        // Obtain the attributes
                        String attributesStr = MRstr.substring(MRstr.indexOf('(') + 1, MRstr.length() - 1);
                        HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
                        // Track the indexes used for variables identifiers (seperately for each attribute)
                        HashMap<String, Integer> attrXIndeces = new HashMap<>();
                        if (!attributesStr.isEmpty()) {
                            // Parse the attributes and their values
                            String[] args = attributesStr.split(";");
                            for (String arg : args) {
                                String attr;
                                String value = "";
                                // If the attribute has corresponding values
                                if (arg.contains("=")) {
                                    String[] subAttr = arg.split("=");
                                    value = subAttr[1].toLowerCase();
                                    attr = subAttr[0].toLowerCase().replaceAll("_", "");

                                    if (value.startsWith("\'")) {
                                        value = value.substring(1, value.length() - 1);
                                    }
                                    // Normalize some closed set values
                                    if (value.equals("true")) {
                                        value = "yes";
                                    }
                                    if (value.equals("false")) {
                                        value = "no";
                                    }
                                    if (value.equals("dontcare")) {
                                        value = "dont_care";
                                    }
                                    if ((" " + value + " ").contains(" avenue ")) {
                                        value = (" " + value + " ").replace(" avenue ", " ave ").trim();
                                    }
                                    // Treat these values as seperate attributes since they are expressed quite differently
                                    if (value.equals("no")
                                            || value.equals("yes")
                                            || value.equals("yes or no")
                                            || value.equals("none")
                                            || value.equals("empty")) {
                                        attr += "_" + value.replaceAll(" ", "_");
                                        value = attr;
                                    }
                                    // Treat "dont_care" instances, as if "dont_care" is the attribute, and the original attribute is the value
                                    // We do this because the phrasing is very similar between different "dont_care" realizations
                                    if (value.equals("dont_care")) {
                                        String v = value;
                                        value = attr;
                                        attr = v;
                                    }
                                } else {
                                    attr = arg.replaceAll("_", "");
                                }
                                if (!getAttributes().get(predicate).contains(attr)) {
                                    getAttributes().get(predicate).add(attr);
                                }
                                if (!attributeValues.containsKey(attr)) {
                                    attributeValues.put(attr, new HashSet<String>());
                                }
                                // If the attribute has no corresponding value, we encode it by using the attibute identifier as the value
                                if (value.isEmpty()) {
                                    value = attr;
                                }

                                // If the value is a variable, we name it as {@X@ + attribute identifier + variable index (for this attribute)}
                                // This occurs when values are already set as variables in the MR, before any delexicalization happens
                                if (value.toLowerCase().startsWith("x")) {
                                    int index = 0;
                                    if (!attrXIndeces.containsKey(attr)) {
                                        attrXIndeces.put(attr, 1);
                                    } else {
                                        index = attrXIndeces.get(attr);
                                        attrXIndeces.put(attr, index + 1);
                                    }
                                    value = "x" + index;
                                }
                                attributeValues.get(attr).add(value.trim().toLowerCase());
                            }
                        }

                        // Delexicalizing the attribute/value pairs
                        HashMap<String, HashSet<String>> delexicalizedAttributeValues = new HashMap<>();
                        HashMap<String, HashMap<String, Integer>> attrValuePriorities = new HashMap<>();
                        int maximumPriority = 0;
                        /* Delixalization of values needs to happen incrementally with priority given to the values of greater lenth, to avoid overlap of values in the reference
                         * e.g. for the MR: inform{name="inn on castro", near="castro"}, with the reference "inn on castro is a nice restaurant",
                         *      we need to first align and delexicalize the "inn on castro" value, before the "castro" value 
                         *      (in this case because "castro" doesn't appear in the reference, but even if it appeared later the priorities would help align it with the correct one)
                        */
                        // We begin by determining which values may require delexicalization, and which not
                        for (String attr : attributeValues.keySet()) {
                            if (!attr.isEmpty()) {
                                delexicalizedAttributeValues.put(attr, new HashSet<String>());
                                attrValuePriorities.put(attr, new HashMap<String, Integer>());
                                for (String value : attributeValues.get(attr)) {
                                    if (!value.equals("none")
                                            && !value.equals("empty")
                                            && !value.equals("yes")
                                            && !value.equals("yes or no")
                                            && !value.equals("no")
                                            && !value.equals(attr)) {
                                        // Initially priorities are given according to value order
                                        attrValuePriorities.get(attr).put(value, maximumPriority);
                                        maximumPriority++;
                                    } else {
                                        // No delexicalization is needed here
                                        delexicalizedAttributeValues.get(attr).add(value);
                                    }
                                }
                            }
                        }
                        // We shift the priorities of different values, according to their perspective lengths (i.e. longer values have higher priority)
                        boolean change = true;
                        while (change) {
                            change = false;
                            for (String attr1 : attrValuePriorities.keySet()) {
                                for (String value1 : attrValuePriorities.get(attr1).keySet()) {
                                    for (String attr2 : attrValuePriorities.keySet()) {
                                        for (String value2 : attrValuePriorities.get(attr2).keySet()) {
                                            if (!value1.equals(value2)
                                                    && value1.contains(value2)
                                                    && attrValuePriorities.get(attr1).get(value1) > attrValuePriorities.get(attr2).get(value2)) {
                                                int prio1 = attrValuePriorities.get(attr1).get(value1);
                                                int prio2 = attrValuePriorities.get(attr2).get(value2);
                                                attrValuePriorities.get(attr1).put(value1, prio2);
                                                attrValuePriorities.get(attr2).put(value2, prio1);
                                                change = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Map between variables and their lexicalized values, required for relexicalization during postprocessing after the sentence generation
                        HashMap<String, String> delexicalizationMap = new HashMap<>();
                        ref = " " + ref + " ";
                        // Delexicalization occurs, in order of priority
                        for (int priority = 0; priority < maximumPriority; priority++) {
                            for (String attr : attrValuePriorities.keySet()) {
                                if (!attrXIndeces.containsKey(attr)) {
                                    attrXIndeces.put(attr, 0);
                                }
                                for (String value : attrValuePriorities.get(attr).keySet()) {
                                    if (attrValuePriorities.get(attr).get(value) == priority) {
                                        // If the value doesn't appear verbatim in the reference, and the value is not composed of multiple subvalues (i.e. doesn't contain connectives)
                                        if (!ref.contains(" " + value + " ")
                                                && !value.contains(" and ")
                                                && !value.contains(" or ")) {
                                            if (value.equals("restaurant")
                                                    && ref.contains(" place ")) {
                                                ref = ref.replace(" place ", " " + Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr) + " ");
                                                ref = ref.replaceAll("  ", " ");
                                                delexicalizedAttributeValues.get(attr).add(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr));
                                                delexicalizationMap.put(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr), "place");
                                                attrXIndeces.put(attr, attrXIndeces.get(attr) + 1);
                                            } else {
                                                delexicalizedAttributeValues.get(attr).add(value);
                                            }
                                        // If the value doesn't appear verbatim in the reference, but the value is composed of multiple sub-values
                                        } else if (!ref.contains(" " + value + " ")
                                                && (value.contains(" and ")
                                                || value.contains(" or "))) {
                                            // We first check if the value appears verbatim when we switch "and" with "or" and vice versa
                                            // We do this due to some inconsistencies in the dataset on how conjuctions are treated
                                            String tempValue = value;
                                            if (value.contains(" and ")) {
                                                tempValue = value.replace(" and ", " or ");
                                            } else if (value.contains(" or ")) {
                                                tempValue = value.replace(" or ", " and ");
                                            }

                                            if (ref.contains(" " + tempValue + " ")) {
                                                ref = ref.replace(" " + tempValue + " ", " " + Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr) + " ");
                                                ref = ref.replaceAll("  ", " ");
                                                delexicalizedAttributeValues.get(attr).add(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr));
                                                delexicalizationMap.put(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr), value);
                                                attrXIndeces.put(attr, attrXIndeces.get(attr) + 1);
                                            } else {
                                                // We split the conjunction into the seperate values; so far the code supports only 2 sub-values
                                                String[] values = new String[2];
                                                if (value.contains(" and ")) {
                                                    values = value.split(" and ");
                                                } else if (value.contains(" or ")) {
                                                    values = value.split(" or ");
                                                }
                                                // And check if the conjunction appears verbatim when we switch the position of the sub-values
                                                String newValue1 = values[1] + " and " + values[0];
                                                String newValue2 = values[1] + " or " + values[0];
                                                if (ref.contains(" " + newValue1 + " ")) {
                                                    ref = ref.replace(" " + newValue1 + " ", " " + Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr) + " ");
                                                    ref = ref.replaceAll("  ", " ");
                                                    delexicalizedAttributeValues.get(attr).add(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr));
                                                    delexicalizationMap.put(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr), value);
                                                    attrXIndeces.put(attr, attrXIndeces.get(attr) + 1);
                                                } else if (ref.contains(" " + newValue2 + " ")) {
                                                    ref = ref.replace(" " + newValue2 + " ", " " + Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr) + " ");
                                                    ref = ref.replaceAll("  ", " ");
                                                    delexicalizedAttributeValues.get(attr).add(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr));
                                                    delexicalizationMap.put(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr), value);
                                                    attrXIndeces.put(attr, attrXIndeces.get(attr) + 1);
                                                }
                                            }
                                        // If the value appears verbatim in the reference, delexicalize it
                                        } else {
                                            ref = ref.replace(" " + value + " ", " " + Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr) + " ");
                                            ref = ref.replaceAll("  ", " ");
                                            delexicalizedAttributeValues.get(attr).add(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr));
                                            delexicalizationMap.put(Action.TOKEN_X + attr + "_" + attrXIndeces.get(attr), value);
                                            attrXIndeces.put(attr, attrXIndeces.get(attr) + 1);
                                        }
                                    }
                                }
                            }
                        }
                        ref = ref.trim();

                        // We construct the MeaningRepresentation
                        MeaningRepresentation MR = new MeaningRepresentation(predicate, delexicalizedAttributeValues, MRstr, delexicalizationMap);
                                                
                        // Sequences of attribute/values pairs and words in the order we observe this in the reference
                        ArrayList<String> observedAttrValueSequence = new ArrayList<>();
                        ArrayList<String> observedWordSequence = new ArrayList<>();
                        
                        // The observed word sequence does not include punctuation
                        String[] words = ref.replaceAll("([,.?!;:'])", " $1").split(" ");
                        
                        // We construct the observed word sequence (and fix some orthographical errors along the way)
                        for (int w = 0; w < words.length; w++) {
                            if (!words[w].trim().isEmpty()) {
                                if (!words[w].trim().isEmpty()
                                        && (observedWordSequence.isEmpty()
                                        || !words[w].trim().equals(observedWordSequence.get(observedWordSequence.size() - 1)))) {
                                    if (words[w].trim().equals("s")
                                            && (observedWordSequence.get(observedWordSequence.size() - 1).equals("child"))) {
                                        observedWordSequence.set(observedWordSequence.size() - 1, "children");
                                    } else if (words[w].trim().equals("addres")
                                            || words[w].trim().equals("adress")) {
                                        observedWordSequence.add("address");
                                    } else if (words[w].trim().equals("mathch")) {
                                        observedWordSequence.add("match");
                                    } else if (words[w].trim().equals("prefered")) {
                                        observedWordSequence.add("preferred");
                                    } else if (words[w].trim().equals("relevent")) {
                                        observedWordSequence.add("relevant");
                                    } else if (words[w].trim().equals("alloed")) {
                                        observedWordSequence.add("allowed");
                                    } else if (words[w].trim().equals("avalible")
                                            || words[w].trim().equals("avalable")) {
                                        observedWordSequence.add("available");
                                    } else if (words[w].trim().equals("tha")
                                            || words[w].trim().equals("te")) {
                                        observedWordSequence.add("the");
                                    } else if (words[w].trim().equals("internect")) {
                                        observedWordSequence.add("internet");
                                    } else if (words[w].trim().equals("wether")) {
                                        observedWordSequence.add("whether");
                                    } else if (words[w].trim().equals("aplogize")) {
                                        observedWordSequence.add("apologize");
                                    } else if (words[w].trim().equals("accomodations")) {
                                        observedWordSequence.add("accommodations");
                                    } else if (words[w].trim().equals("whould")) {
                                        observedWordSequence.add("would");
                                    } else if (words[w].trim().equals("aceepted")) {
                                        observedWordSequence.add("accepted");
                                    } else if (words[w].trim().equals("postode")) {
                                        observedWordSequence.add("postcode");
                                    } else if (words[w].trim().equals("ive")) {
                                        observedWordSequence.add("i");
                                        observedWordSequence.add("have");
                                    } else if (words[w].trim().equals("waht")) {
                                        observedWordSequence.add("what");
                                    } else if (words[w].trim().equals("neighborhood")) {
                                        observedWordSequence.add("neighbourhood");
                                    } else if (words[w].trim().equals("prefernce")) {
                                        observedWordSequence.add("preference");
                                    } else if (words[w].trim().equals("dont")) {
                                        observedWordSequence.add("don't");
                                    } else if (words[w].trim().equals("isnt")) {
                                        observedWordSequence.add("isn't");
                                    } else if (words[w].trim().equals("intenet")
                                            || words[w].trim().equals("internetn")) {
                                        observedWordSequence.add("internet");
                                    } else if (words[w].trim().equals("cannote")) {
                                        observedWordSequence.add("cannot");
                                    } else if (words[w].trim().equals("notels")) {
                                        observedWordSequence.add("hotels");
                                    } else if (words[w].trim().equals("phne")) {
                                        observedWordSequence.add("phone");
                                    } else if (words[w].trim().equals("taht")) {
                                        observedWordSequence.add("that");
                                    } else if (words[w].trim().equals("postdocde")) {
                                        observedWordSequence.add("postcode");
                                    } else if (words[w].trim().equals("accpects")) {
                                        observedWordSequence.add("accepts");
                                    } else if (words[w].trim().equals("doesn")
                                            || words[w].trim().equals("doesnt")
                                            || words[w].trim().equals("doesn")) {
                                        observedWordSequence.add("doesn't");
                                    } else if (words[w].trim().equals("restaurnats")) {
                                        observedWordSequence.add("restarnauts");
                                    } else if (words[w].trim().equals("ther")
                                            || words[w].trim().equals("thers")) {
                                        observedWordSequence.add("there");
                                        
                                    // The dataset treats the suffixes "s" and "-ly" as separate words
                                    // We combine these suffixes with their preceding words but keep a cache with these changes to revert them before evaluation (we have to do this so that the token-based evaluation metrics are calculated in a consistent manner with Wen et al.'s)
                                    } else if (words[w].trim().equals("s")) {
                                        if (observedWordSequence.isEmpty()) {
                                            observedWordSequence.add(words[w].trim().toLowerCase());
                                        } else if (observedWordSequence.get(observedWordSequence.size() - 1).startsWith(Action.TOKEN_X)) {
                                            observedWordSequence.add(words[w].trim().toLowerCase());
                                        } else {
                                            getCompositeSuffixesInData().put(observedWordSequence.get(observedWordSequence.size() - 1) + "s", observedWordSequence.get(observedWordSequence.size() - 1) + " s");
                                            observedWordSequence.set(observedWordSequence.size() - 1, observedWordSequence.get(observedWordSequence.size() - 1) + "s");
                                        }
                                    } else if (words[w].trim().equals("-ly")) {
                                        if (observedWordSequence.isEmpty()) {
                                            observedWordSequence.add(words[w].trim().toLowerCase());
                                        } else if (observedWordSequence.get(observedWordSequence.size() - 1).startsWith(Action.TOKEN_X)) {
                                            observedWordSequence.add(words[w].trim().toLowerCase());
                                        } else {
                                            getCompositeSuffixesInData().put(observedWordSequence.get(observedWordSequence.size() - 1) + "ly", observedWordSequence.get(observedWordSequence.size() - 1) + " -ly");
                                            observedWordSequence.set(observedWordSequence.size() - 1, observedWordSequence.get(observedWordSequence.size() - 1) + "ly");
                                        }
                                    } else {
                                        observedWordSequence.add(words[w].trim().toLowerCase());
                                    }
                                }
                            }
                        }

                        //Probably deprecated, need to do some more tests
                        MR.getAttributeValues().keySet().forEach((attr) -> {
                            MR.getAttributeValues().get(attr).stream().filter((value) -> (attr.equals("name") && value.equals("none"))).forEachOrdered((value) -> {
                                observedAttrValueSequence.add(0, attr.toLowerCase() + "=" + value.toLowerCase());
                            });
                        });
                        observedAttrValueSequence.add(Action.TOKEN_END);
                        
                        // We store the maximum observed word sequence length, to use as a limit during generation
                        if (observedWordSequence.size() > getMaxWordSequenceLength()) {
                            setMaxWordSequenceLength(observedWordSequence.size());
                        }

                        
                        // We initialize the alignments between words and attribute/value pairs
                        ArrayList<String> wordToAttrValueAlignment = new ArrayList<>();
                        // And populate them with "unaligned" tokens (i.e. "[]") and punctuation alignments; we do the latter so we know to filter out punctuation when estimating the alignments in later stages
                        observedWordSequence.forEach((word) -> {
                            if (word.trim().matches("[,.?!;:']")) {
                                wordToAttrValueAlignment.add(Action.TOKEN_PUNCT);
                            } else {
                                wordToAttrValueAlignment.add("[]");
                            }
                        });
                        // And using both word sequence and initial alignments, we construct a draft sequence of word actions corresponding to the reference
                        ArrayList<Action> directReferenceSequence = new ArrayList<>();
                        for (int r = 0; r < observedWordSequence.size(); r++) {
                            directReferenceSequence.add(new Action(observedWordSequence.get(r), wordToAttrValueAlignment.get(r)));
                        }                                        
                        // Finally, we construct the DatasetInstance
                        DatasetInstance DI = new DatasetInstance(MR, directReferenceSequence, postProcessRef(MR, directReferenceSequence));
                        // We add the evaluation references of all previously constructed DatasetInstances (that are identical to this one) as available evaluation references 
                        getDatasetInstances().get(predicate).stream().filter((existingDI) -> (existingDI.getMeaningRepresentation().getAbstractMR().equals(DI.getMeaningRepresentation().getAbstractMR()))).map((existingDI) -> {
                            existingDI.getEvaluationReferences().addAll(DI.getEvaluationReferences());
                            return existingDI;
                        }).forEachOrdered((existingDI) -> {
                            // We add the direct reference of this DatasetInstance as an available evaluation reference to all previously constructed DatasetInstance that are identical to this one
                            DI.getEvaluationReferences().addAll(existingDI.getEvaluationReferences());
                        });
                        getDatasetInstances().get(predicate).add(DI);

                        // Calculate the possible alignments between (non-delexicalized) attribute values and reference subphrases
                        // We do this by comparing the values with n-gram subphrases of the reference, using character-level Levenshtein distance
                        // These are used during the estimation of naive alignments, but also for tracking which values have possibly been expressed during generation
                        HashMap<String, HashMap<String, Double>> observedValueAlignments = new HashMap<>();
                        MR.getAttributeValues().keySet().forEach((attr) -> {
                            MR.getAttributeValues().get(attr).stream().filter((value) -> (!value.equals("name=none")
                                    && !value.startsWith(Action.TOKEN_X)
                                    && !(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+")))).forEachOrdered((value) -> {
                                String valueToCompare = value;
                                if (value.equals("no")
                                        || value.equals("yes")
                                        || value.equals("yes or no")
                                        || value.equals("none")
                                        || value.equals("empty")) {
                                    // If the value is boolean or non-existant, we also compare using the attribute name
                                    valueToCompare = attr;
                                    observedValueAlignments.put(valueToCompare + ":" + value, new HashMap<String, Double>());
                                } else {
                                    observedValueAlignments.put(valueToCompare, new HashMap<String, Double>());
                                }
                                //For all n-grams in the referenec
                                for (int n = 1; n < observedWordSequence.size(); n++) {
                                    //Calculate the similaritie between them and valueToCompare
                                    for (int r = 0; r <= observedWordSequence.size() - n; r++) {
                                        boolean compareAgainstNGram = true;
                                        for (int j = 0; j < n; j++) {
                                            if (observedWordSequence.get(r + j).startsWith(Action.TOKEN_X)
                                                    || wordToAttrValueAlignment.get(r + j).equals(Action.TOKEN_PUNCT)
                                                    || StringNLPUtilities.isArticle(observedWordSequence.get(r + j))
                                                    || observedWordSequence.get(r + j).equalsIgnoreCase("and")
                                                    || observedWordSequence.get(r + j).equalsIgnoreCase("or")) {
                                                // We ignore n-grams that contain variables, punctuation, articles, or conjuctions
                                                // In other words, we do not allow values to align with such n-grams
                                                compareAgainstNGram = false;
                                            }
                                        }
                                        if (compareAgainstNGram) {
                                            String align = "";
                                            String compare = "";
                                            String backwardCompare = "";
                                            for (int j = 0; j < n; j++) {
                                                // The coordinates of the alignment
                                                align += (r + j) + " ";
                                                compare += observedWordSequence.get(r + j);
                                                backwardCompare = observedWordSequence.get(r + j) + backwardCompare;
                                            }
                                            align = align.trim();

                                            // Calculate the character-level distance between the value and the nGram (in its original and reversed order)
                                            Double distance = Levenshtein.getSimilarity(valueToCompare.toLowerCase(), compare.toLowerCase(), true);
                                            Double backwardDistance = Levenshtein.getSimilarity(valueToCompare.toLowerCase(), backwardCompare.toLowerCase(), true);

                                            // We keep the best distance score; note that the Levenshtein distance is normalized so that greater is better 
                                            if (backwardDistance > distance) {
                                                distance = backwardDistance;
                                            }
                                            // We ignore all nGrams that are less similar than a threshold
                                            if (distance > 0.3) {
                                                if (value.equals("no")
                                                        || value.equals("yes")
                                                        || value.equals("yes or no")
                                                        || value.equals("none")
                                                        || value.equals("empty")) {
                                                    observedValueAlignments.get(valueToCompare + ":" + value).put(align, distance);
                                                } else {
                                                    observedValueAlignments.get(valueToCompare).put(align, distance);
                                                }
                                            }
                                        }
                                    }
                                }
                            });
                        });

                        // We filter out any values that haven't been aligned
                        HashSet<String> toRemove = new HashSet<>();
                        for (String value : observedValueAlignments.keySet()) {
                            if (observedValueAlignments.get(value).isEmpty()) {
                                toRemove.add(value);
                            }
                        }
                        for (String value : toRemove) {
                            observedValueAlignments.remove(value);
                        }

                        // We keep the best aligned nGrams; since we do not want the aligned nGrams to be overlapping, we remove any overlapping alignments after we pick each one
                        while (!observedValueAlignments.keySet().isEmpty()) {
                            // Find the best aligned nGram
                            Double max = Double.NEGATIVE_INFINITY;
                            String[] bestAlignment = new String[2];
                            for (String value : observedValueAlignments.keySet()) {
                                for (String alignment : observedValueAlignments.get(value).keySet()) {
                                    if (observedValueAlignments.get(value).get(alignment) > max) {
                                        max = observedValueAlignments.get(value).get(alignment);
                                        bestAlignment[0] = value;
                                        bestAlignment[1] = alignment;
                                    }
                                }
                            }

                            // Find the subphrase that corresponds to the best aligned nGram, according to the coordinates
                            ArrayList<String> alignedStr = new ArrayList<>();
                            String[] coords = bestAlignment[1].split(" ");
                            if (coords.length == 1) {
                                alignedStr.add(observedWordSequence.get(Integer.parseInt(coords[0].trim())));
                            } else {
                                for (int a = Integer.parseInt(coords[0].trim()); a <= Integer.parseInt(coords[coords.length - 1].trim()); a++) {
                                    alignedStr.add(observedWordSequence.get(a));
                                }
                            }

                            // Store the best aligned nGram
                            if (!getValueAlignments().containsKey(bestAlignment[0])) {
                                getValueAlignments().put(bestAlignment[0], new HashMap<ArrayList<String>, Double>());
                            }
                            getValueAlignments().get(bestAlignment[0]).put(alignedStr, max);

                            // And remove it from the observed ones for this instance
                            observedValueAlignments.remove(bestAlignment[0]);
                            // And also remove any other aligned nGrams that are overlapping with the best aligned nGram
                            observedValueAlignments.keySet().forEach((value) -> {
                                HashSet<String> alignmentsToBeRemoved = new HashSet<>();
                                observedValueAlignments.get(value).keySet().forEach((alignment) -> {
                                    String[] othCoords = alignment.split(" ");
                                    if (Integer.parseInt(coords[0].trim()) <= Integer.parseInt(othCoords[0].trim()) && (Integer.parseInt(coords[coords.length - 1].trim()) >= Integer.parseInt(othCoords[0].trim()))
                                            || (Integer.parseInt(othCoords[0].trim()) <= Integer.parseInt(coords[0].trim()) && Integer.parseInt(othCoords[othCoords.length - 1].trim()) >= Integer.parseInt(coords[0].trim()))) {
                                        alignmentsToBeRemoved.add(alignment);
                                    }
                                });
                                alignmentsToBeRemoved.forEach((alignment) -> {
                                    observedValueAlignments.get(value).remove(alignment);
                                });
                            });
                            // We filter out any values that are no logner aligned (due to overlapping conflicts)
                            toRemove = new HashSet<>();
                            for (String value : observedValueAlignments.keySet()) {
                                if (observedValueAlignments.get(value).isEmpty()) {
                                    toRemove.add(value);
                                }
                            }
                            for (String value : toRemove) {
                                observedValueAlignments.remove(value);
                            }
                        }
                        getObservedAttrValueSequences().add(observedAttrValueSequence);
                    }
                }
            }
        } catch (JSONException ex) {
        }
    }

    /**
     * Infers the feature/cost vectors for the content and word actions.
     * @return The feature/cost vectors for the content and word actions
     */
    public Object[] inferFeatureAndCostVectors() {
        ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>> contentTrainingData = new ConcurrentHashMap<>();
        ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>> wordTrainingData = new ConcurrentHashMap<>();

        if (!getAvailableWordActions().isEmpty() && !getPredicates().isEmpty()) {
            // Initialize collections
            getTrainingData().stream().map((di) -> {
                contentTrainingData.put(di, new HashMap<String, ArrayList<Instance>>());
                return di;
            }).map((di) -> {
                wordTrainingData.put(di, new HashMap<String, HashMap<String, ArrayList<Instance>>>());
                return di;
            }).forEachOrdered((di) -> {
                getPredicates().stream().map((predicate) -> {
                    contentTrainingData.get(di).put(predicate, new ArrayList<Instance>());
                    return predicate;
                }).map((predicate) -> {
                    wordTrainingData.get(di).put(predicate, new HashMap<String, ArrayList<Instance>>());
                    return predicate;
                }).forEachOrdered((predicate) -> {
                    getAttributes().get(predicate).stream().filter((attribute) -> (!wordTrainingData.get(di).get(predicate).containsKey(attribute))).forEachOrdered((attribute) -> {
                        wordTrainingData.get(di).get(predicate).put(attribute, new ArrayList<Instance>());
                    });
                });
            });

            // Infer the vectors in parallel processes to save time
            ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);
            getTrainingData().forEach((di) -> {
                executor.execute(new InferSFXVectorsThread(di, this, contentTrainingData, wordTrainingData));
            });
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
        }
        Object[] results = new Object[2];
        results[0] = contentTrainingData;
        results[1] = wordTrainingData;
        return results;
    }

    /**
     *
     * @param trainingData
     */
    public void createRandomAlignments(ArrayList<DatasetInstance> trainingData) {
        HashMap<String, HashMap<ArrayList<Action>, HashMap<Action, Integer>>> punctPatterns = new HashMap<>();
        getPredicates().forEach((predicate) -> {
            punctPatterns.put(predicate, new HashMap<ArrayList<Action>, HashMap<Action, Integer>>());
        });
        HashMap<DatasetInstance, ArrayList<Action>> punctRealizations = new HashMap<DatasetInstance, ArrayList<Action>>();

        HashMap<ArrayList<Action>, ArrayList<Action>> calculatedRealizationsCache = new HashMap<>();
        trainingData.stream().map((di) -> {
            HashSet<ArrayList<Action>> initRealizations = new HashSet<>();
            if (!calculatedRealizationsCache.containsKey(di.getDirectReferenceSequence())) {
                initRealizations.add(di.getDirectReferenceSequence());
            }
            initRealizations.stream().map((realization) -> {
                HashMap<String, HashSet<String>> values = new HashMap<>();
                di.getMeaningRepresentation().getAttributeValues().keySet().forEach((attr) -> {
                    values.put(attr, new HashSet<>(di.getMeaningRepresentation().getAttributeValues().get(attr)));
                });
                ArrayList<Action> randomRealization = new ArrayList<Action>();
                realization.forEach((a) -> {
                    if (a.getAttribute().equals(Action.TOKEN_PUNCT)) {
                        randomRealization.add(new Action(a.getWord(), a.getAttribute()));
                    } else {
                        randomRealization.add(new Action(a.getWord(), ""));
                    }
                });
                HashSet<String> unalignedAttrs = new HashSet<>();
                if (values.keySet().isEmpty()) {
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!getAttributes().get(di.getMeaningRepresentation().getPredicate()).contains("empty")) {
                                getAttributes().get(di.getMeaningRepresentation().getPredicate()).add("empty");
                            }
                            randomRealization.get(i).setAttribute("empty=empty");
                        }
                    }
                } else {
                    values.keySet().forEach((attr) -> {
                        values.get(attr).forEach((value) -> {
                            if ((!(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+") || value.startsWith(Action.TOKEN_X)))
                                    && !value.isEmpty()) {
                                String valueToCheck = value;
                                if (valueToCheck.equals("no")
                                        || valueToCheck.equals("yes")
                                        || valueToCheck.equals("yes or no")
                                        || valueToCheck.equals("none")
                                        //|| valueToCheck.equals("dont_care")
                                        || valueToCheck.equals("empty")) {
                                    valueToCheck = attr + ":" + value;
                                    unalignedAttrs.add(attr + "=" + value);
                                }
                                if (valueToCheck.equals(attr)) {
                                    unalignedAttrs.add(attr + "=" + value);
                                }
                                if (!valueToCheck.equals("empty:empty")
                                        && getValueAlignments().containsKey(valueToCheck)) {
                                    unalignedAttrs.add(attr + "=" + valueToCheck);
                                }
                            } else {
                                unalignedAttrs.add(attr + "=" + value);
                            }
                        });
                    });
                    unalignedAttrs.forEach((attrValue) -> {
                        int index = getRandomGen().nextInt(randomRealization.size());
                        boolean change = false;
                        while (!change) {
                            if (!randomRealization.get(index).getAttribute().equals(Action.TOKEN_PUNCT)) {
                                randomRealization.get(index).setAttribute(attrValue.toLowerCase().trim());
                                change = true;
                            } else {
                                index = getRandomGen().nextInt(randomRealization.size());
                            }
                        }
                    });
                    String previousAttr = "";
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        } else {
                            previousAttr = "";
                        }
                    }
                    //System.out.println("1: " + randomRealization);
                    previousAttr = "";
                    for (int i = randomRealization.size() - 1; i >= 0; i--) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        } else {
                            previousAttr = "";
                        }
                    }
                    //System.out.println("2: " + randomRealization);
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
                    //System.out.println("3: " + randomRealization);
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
                    //System.out.println("4: " + randomRealization);
                }
                //FIX WRONG @PUNCT@
                String previousAttr = "";
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
                randomRealization.stream().filter((a) -> (!a.getAttribute().equals(Action.TOKEN_PUNCT))).forEachOrdered((a) -> {
                    cleanRandomRealization.add(a);
                });
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
                //System.out.println(di.getMeaningRepresentation().getPredicate() + ": " + endRandomRealization);
                ArrayList<String> attrValues = new ArrayList<String>();
                endRandomRealization.forEach((a) -> {
                    if (attrValues.isEmpty()) {
                        attrValues.add(a.getAttribute());
                    } else if (!attrValues.get(attrValues.size() - 1).equals(a.getAttribute())) {
                        attrValues.add(a.getAttribute());
                    }
                });
                if (attrValues.size() > getMaxContentSequenceLength()) {
                    setMaxContentSequenceLength(attrValues.size());
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
                return punctRealization;
            }).map((punctRealization) -> {
                punctRealizations.put(di, punctRealization);
                return punctRealization;
            }).forEachOrdered((punctRealization) -> {
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
                            if (!punctPatterns.get(di.getMeaningRepresentation().getPredicate()).containsKey(surroundingActions)) {
                                punctPatterns.get(di.getMeaningRepresentation().getPredicate()).put(surroundingActions, new HashMap<Action, Integer>());
                            }
                            if (!punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).containsKey(a)) {
                                punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).put(a, 1);
                            } else {
                                punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).put(a, punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).get(a) + 1);
                            }
                        }
                    }
                }
            });
            return di;
        }).map((di) -> {
            di.setDirectReferenceSequence(calculatedRealizationsCache.get(di.getDirectReferenceSequence()));
            return di;
        });
        punctRealizations.keySet().forEach((di) -> {
            ArrayList<Action> punctRealization = punctRealizations.get(di);
            punctPatterns.get(di.getMeaningRepresentation().getPredicate()).keySet().forEach((surrounds) -> {
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
                        if (!punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).containsKey(a)) {
                            punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).put(a, 1);
                        } else {
                            punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).put(a, punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).get(a) + 1);
                        }
                    }
                }
            });
        });
        punctPatterns.keySet().forEach((predicate) -> {
            punctPatterns.get(predicate).keySet().forEach((punct) -> {
                Action bestAction = null;
                int bestCount = 0;
                for (Action a : punctPatterns.get(predicate).get(punct).keySet()) {
                    if (punctPatterns.get(predicate).get(punct).get(a) > bestCount) {
                        bestAction = a;
                        bestCount = punctPatterns.get(predicate).get(punct).get(a);
                    } else if (punctPatterns.get(predicate).get(punct).get(a) == bestCount
                            && bestAction.getWord().isEmpty()) {
                        bestAction = a;
                    }
                }
                if (!getPunctuationPatterns().containsKey(predicate)) {
                    getPunctuationPatterns().put(predicate, new HashMap<ArrayList<Action>, Action>());
                }
                if (!bestAction.getWord().isEmpty()) {
                    getPunctuationPatterns().get(predicate).put(punct, bestAction);
                }
            });
        });
    }

    /**
     *
     * @param trainingData
     */
    @Override
    public void createNaiveAlignments(ArrayList<DatasetInstance> trainingData) {
        HashMap<String, HashMap<ArrayList<Action>, HashMap<Action, Integer>>> punctPatterns = new HashMap<>();
        getPredicates().forEach((predicate) -> {
            punctPatterns.put(predicate, new HashMap<ArrayList<Action>, HashMap<Action, Integer>>());
        });
        HashMap<DatasetInstance, ArrayList<Action>> punctRealizations = new HashMap<DatasetInstance, ArrayList<Action>>();

        trainingData.stream().map((di) -> {
            HashMap<ArrayList<Action>, ArrayList<Action>> calculatedRealizationsCache = new HashMap<>();
            HashSet<ArrayList<Action>> initRealizations = new HashSet<>();
            if (!calculatedRealizationsCache.containsKey(di.getDirectReferenceSequence())) {
                initRealizations.add(di.getDirectReferenceSequence());
            }
            initRealizations.stream().map((realization) -> {
                HashMap<String, HashSet<String>> values = new HashMap<>();
                di.getMeaningRepresentation().getAttributeValues().keySet().forEach((attr) -> {
                    values.put(attr, new HashSet<>(di.getMeaningRepresentation().getAttributeValues().get(attr)));
                });
                ArrayList<Action> randomRealization = new ArrayList<>();
                for (int i = 0; i < realization.size(); i++) {
                    Action a = realization.get(i);
                    if (a.getAttribute().equals(Action.TOKEN_PUNCT)) {
                        randomRealization.add(new Action(a.getWord(), a.getAttribute()));
                    } else {
                        randomRealization.add(new Action(a.getWord(), ""));
                    }
                }
                if (values.keySet().isEmpty()) {
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!getAttributes().get(di.getMeaningRepresentation().getPredicate()).contains("empty")) {
                                getAttributes().get(di.getMeaningRepresentation().getPredicate()).add("empty");
                            }
                            randomRealization.get(i).setAttribute("empty=empty");
                        }
                    }
                } else {
                    HashMap<Double, HashMap<String, ArrayList<Integer>>> indexAlignments = new HashMap<>();
                    HashSet<String> noValueAttrs = new HashSet<String>();
                    values.keySet().forEach((attr) -> {
                        values.get(attr).stream().filter((value) -> ((!(value.matches("\"[xX][0-9]+\"") || value.matches("[xX][0-9]+") || value.startsWith(Action.TOKEN_X)))
                                && !value.isEmpty())).map((value) -> {
                            String valueToCheck = value;
                            if (valueToCheck.equals("no")
                                    || valueToCheck.equals("yes")
                                    || valueToCheck.equals("yes or no")
                                    || valueToCheck.equals("none")
                                    //|| attr.equals("dont_care")
                                    || valueToCheck.equals("empty")) {
                                valueToCheck = attr + ":" + value;
                                noValueAttrs.add(attr + "=" + value);
                            }
                            if (valueToCheck.equals(attr)) {
                                noValueAttrs.add(attr + "=" + value);
                            }
                            return valueToCheck;
                        }).filter((valueToCheck) -> (!valueToCheck.equals("empty:empty")
                                && getValueAlignments().containsKey(valueToCheck))).forEachOrdered((valueToCheck) -> {
                            for (ArrayList<String> align : getValueAlignments().get(valueToCheck).keySet()) {
                                int n = align.size();
                                for (int i = 0; i <= randomRealization.size() - n; i++) {
                                    ArrayList<String> compare = new ArrayList<String>();
                                    ArrayList<Integer> indexAlignment = new ArrayList<Integer>();
                                    for (int j = 0; j < n; j++) {
                                        compare.add(randomRealization.get(i + j).getWord());
                                        indexAlignment.add(i + j);
                                    }
                                    if (compare.equals(align)) {
                                        if (!indexAlignments.containsKey(getValueAlignments().get(valueToCheck).get(align))) {
                                            indexAlignments.put(getValueAlignments().get(valueToCheck).get(align), new HashMap());
                                        }
                                        indexAlignments.get(getValueAlignments().get(valueToCheck).get(align)).put(attr + "=" + valueToCheck, indexAlignment);
                                    }
                                }
                            }
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
                    //System.out.println("-1: " + randomRealization);
                    randomRealization.stream().filter((a) -> (a.getWord().startsWith(Action.TOKEN_X))).forEachOrdered((a) -> {
                        String attr = a.getWord().substring(3, a.getWord().lastIndexOf('_')).toLowerCase().trim();
                        a.setAttribute(attr + "=" + a.getWord());
                    });
                    HashSet<String> unalignedNoValueAttrs = new HashSet<>();
                    noValueAttrs.forEach((noValueAttr) -> {
                        boolean assigned = false;
                        for (Action a : randomRealization) {
                            if (a.getAttribute().equals(noValueAttr)) {
                                assigned = true;
                            }
                        }
                        if (!assigned) {
                            unalignedNoValueAttrs.add(noValueAttr);
                        }
                    });
                    boolean isAllEmpty = true;
                    boolean hasSpace = false;
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (!randomRealization.get(i).getAttribute().isEmpty()
                                && !randomRealization.get(i).getAttribute().equals("[]")
                                && !randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                            isAllEmpty = false;
                        }
                        if (randomRealization.get(i).getAttribute().isEmpty()
                                || randomRealization.get(i).getAttribute().equals("[]")) {
                            hasSpace = true;
                        }
                    }
                    if (isAllEmpty && hasSpace && !unalignedNoValueAttrs.isEmpty()) {
                        unalignedNoValueAttrs.forEach((attrValue) -> {
                            int index = getRandomGen().nextInt(randomRealization.size());
                            boolean change = false;
                            while (!change) {
                                if (!randomRealization.get(index).getAttribute().equals(Action.TOKEN_PUNCT)) {
                                    randomRealization.get(index).setAttribute(attrValue.toLowerCase().trim());
                                    change = true;
                                } else {
                                    index = getRandomGen().nextInt(randomRealization.size());
                                }
                            }
                        });
                    }
                    //System.out.println(isAllEmpty + " " + hasSpace + " " + unalignedNoValueAttrs);
                    //System.out.println(">> " + noValueAttrs);
                    //System.out.println(">> " + values);
                    //System.out.println("0: " + randomRealization);
                    String previousAttr = "";
                    int start = -1;
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
                    }
                    //System.out.println("1: " + randomRealization);
                    previousAttr = "";
                    for (int i = randomRealization.size() - 1; i >= 0; i--) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        } else {
                            previousAttr = "";
                        }
                    }
                    //System.out.println("2: " + randomRealization);
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
                    //System.out.println("3: " + randomRealization);
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
                    //System.out.println("4: " + randomRealization);
                }
                //FIX WRONG @PUNCT@
                String previousAttr = "";
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
                randomRealization.stream().filter((a) -> (!a.getAttribute().equals(Action.TOKEN_PUNCT))).forEachOrdered((a) -> {
                    cleanRandomRealization.add(a);
                });
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
                //System.out.println(di.getMeaningRepresentation().getPredicate() + ": " + endRandomRealization);
                ArrayList<String> attrValues = new ArrayList<String>();
                endRandomRealization.forEach((a) -> {
                    if (attrValues.isEmpty()) {
                        attrValues.add(a.getAttribute());
                    } else if (!attrValues.get(attrValues.size() - 1).equals(a.getAttribute())) {
                        attrValues.add(a.getAttribute());
                    }
                });
                if (attrValues.size() > getMaxContentSequenceLength()) {
                    setMaxContentSequenceLength(attrValues.size());
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
                return punctRealization;
            }).map((punctRealization) -> {
                punctRealizations.put(di, punctRealization);
                return punctRealization;
            }).forEachOrdered((punctRealization) -> {
                for (int i = 0; i < punctRealization.size(); i++) {
                    Action a = punctRealization.get(i);
                    if (a.getAttribute().equals(Action.TOKEN_PUNCT)) {
                        boolean legal = true;
                        ArrayList<Action> surroundingActions = new ArrayList<>();
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
                            if (!punctPatterns.get(di.getMeaningRepresentation().getPredicate()).containsKey(surroundingActions)) {
                                punctPatterns.get(di.getMeaningRepresentation().getPredicate()).put(surroundingActions, new HashMap<Action, Integer>());
                            }
                            if (!punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).containsKey(a)) {
                                punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).put(a, 1);
                            } else {
                                punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).put(a, punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).get(a) + 1);
                            }
                        }
                    }
                }
            });
            di.setDirectReferenceSequence(calculatedRealizationsCache.get(di.getDirectReferenceSequence()));
            return di;
        }).forEachOrdered((di) -> {
            HashSet<String> attrValuesToBeMentioned = new HashSet<>();
            di.getMeaningRepresentation().getAttributeValues().keySet().forEach((attribute) -> {
                int a = 0;
                for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                    if (value.startsWith("\"x")) {
                        value = "x" + a;
                        a++;
                    } else if (value.startsWith("\"")) {
                        value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                    }
                    attrValuesToBeMentioned.add(attribute + "=" + value);
                }
            });
            di.getDirectReferenceSequence().stream().map((key) -> {
                attrValuesToBeMentioned.remove(key.getAttribute());
                return key;
            });
        });
        punctRealizations.keySet().forEach((di) -> {
            ArrayList<Action> punctRealization = punctRealizations.get(di);
            punctPatterns.get(di.getMeaningRepresentation().getPredicate()).keySet().forEach((surrounds) -> {
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
                        if (!punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).containsKey(a)) {
                            punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).put(a, 1);
                        } else {
                            punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).put(a, punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).get(a) + 1);
                        }
                    }
                }
            });
        });
        punctPatterns.keySet().forEach((predicate) -> {
            punctPatterns.get(predicate).keySet().forEach((punct) -> {
                Action bestAction = null;
                int bestCount = 0;
                for (Action a : punctPatterns.get(predicate).get(punct).keySet()) {
                    if (punctPatterns.get(predicate).get(punct).get(a) > bestCount) {
                        bestAction = a;
                        bestCount = punctPatterns.get(predicate).get(punct).get(a);
                    } else if (punctPatterns.get(predicate).get(punct).get(a) == bestCount
                            && bestAction.getWord().isEmpty()) {
                        bestAction = a;
                    }
                }
                if (!getPunctuationPatterns().containsKey(predicate)) {
                    getPunctuationPatterns().put(predicate, new HashMap<ArrayList<Action>, Action>());
                }
                if (!bestAction.getWord().isEmpty()) {
                    getPunctuationPatterns().get(predicate).put(punct, bestAction);
                }
            });
        });
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
    @Override
    public Instance createContentInstance(String predicate, String bestAction, ArrayList<String> previousGeneratedAttrs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, MeaningRepresentation MR, HashMap<String, HashSet<String>> availableAttributeActions) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

        HashSet<String> attrsToBeMentioned = new HashSet<>();
        attrValuesToBeMentioned.stream().map((attrValue) -> {
            String attr = attrValue;
            if (attr.contains("=")) {
                attr = attrValue.substring(0, attrValue.indexOf('='));
            }
            return attr;
        }).forEachOrdered((attr) -> {
            attrsToBeMentioned.add(attr);
        });
        if (!bestAction.isEmpty()) {
            //COSTS
            if (bestAction.equals(Action.TOKEN_END)) {
                costs.put(Action.TOKEN_END, 0.0);
                availableAttributeActions.get(predicate).forEach((action) -> {
                    costs.put(action, 1.0);
                });
            } else if (!bestAction.equals("@TOK@")) {
                costs.put(Action.TOKEN_END, 1.0);
                availableAttributeActions.get(predicate).forEach((action) -> {
                    String attr = bestAction;
                    if (bestAction.contains("=")) {
                        attr = bestAction.substring(0, bestAction.indexOf('=')).toLowerCase().trim();
                    }
                    if (action.equals(attr)) {
                        costs.put(action, 0.0);
                    } else {
                        costs.put(action, 1.0);
                    }
                });
            }
        }
        return createContentInstanceWithCosts(predicate, costs, previousGeneratedAttrs, attrValuesAlreadyMentioned, attrValuesToBeMentioned, availableAttributeActions, MR);
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
    @Override
    public Instance createContentInstanceWithCosts(String predicate, TObjectDoubleHashMap<String> costs, ArrayList<String> previousGeneratedAttrs, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned, HashMap<String, HashSet<String>> availableAttributeActions, MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        if (availableAttributeActions.containsKey(predicate)) {
            availableAttributeActions.get(predicate).forEach((action) -> {
                valueSpecificFeatures.put(action, new TObjectDoubleHashMap<String>());
            });
        }

        ArrayList<String> mentionedAttrValues = new ArrayList<>();
        previousGeneratedAttrs.stream().filter((attrValue) -> (!attrValue.equals(Action.TOKEN_START)
                && !attrValue.equals(Action.TOKEN_END))).forEachOrdered((attrValue) -> {
            mentionedAttrValues.add(attrValue);
        });

        for (int j = 1; j <= 1; j++) {
            String previousAttrValue = "@@";
            if (mentionedAttrValues.size() - j >= 0) {
                previousAttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - j).trim();
            }
            generalFeatures.put("feature_attrValue_" + j + "_" + previousAttrValue, 1.0);
        }
        //Word N-Grams
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

        //If arguments have been generated or not
        for (int i = 0; i < mentionedAttrValues.size(); i++) {
            generalFeatures.put("feature_attrValue_allreadyMentioned_" + mentionedAttrValues.get(i), 1.0);
        }
        //If arguments should still be generated or not
        attrValuesToBeMentioned.forEach((attrValue) -> {
            generalFeatures.put("feature_attrValue_toBeMentioned_" + attrValue, 1.0);
        }); //Which attrs are in the MR and which are not

        if (availableAttributeActions.containsKey(predicate)) {
            availableAttributeActions.get(predicate).forEach((attribute) -> {
                if (MR.getAttributeValues().keySet().contains(attribute)) {
                    generalFeatures.put("feature_attr_inMR_" + attribute, 1.0);
                } else {
                    generalFeatures.put("feature_attr_notInMR_" + attribute, 1.0);
                }
            });
        }

        ArrayList<String> mentionedAttrs = new ArrayList<>();
        for (int i = 0; i < mentionedAttrValues.size(); i++) {
            String attr = mentionedAttrValues.get(i);
            if (attr.contains("=")) {
                attr = mentionedAttrValues.get(i).substring(0, mentionedAttrValues.get(i).indexOf('='));
            }
            mentionedAttrs.add(attr);
        }
        HashSet<String> attrsToBeMentioned = new HashSet<>();
        attrValuesToBeMentioned.stream().map((attrValue) -> {
            String attr = attrValue;
            if (attr.contains("=")) {
                attr = attrValue.substring(0, attrValue.indexOf('='));
            }
            return attr;
        }).forEachOrdered((attr) -> {
            attrsToBeMentioned.add(attr);
        });

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

        //If arguments have been generated or not
        attrValuesAlreadyMentioned.forEach((attr) -> {
            generalFeatures.put("feature_attr_alreadyMentioned_" + attr, 1.0);
        });
        //If arguments should still be generated or not
        attrsToBeMentioned.forEach((attr) -> {
            generalFeatures.put("feature_attr_toBeMentioned_" + attr, 1.0);
        });

        //Attr specific features (and global features)
        if (availableAttributeActions.containsKey(predicate)) {
            for (String action : availableAttributeActions.get(predicate)) {
                if (action.equals(Action.TOKEN_END)) {
                    if (attrsToBeMentioned.isEmpty()) {
                        valueSpecificFeatures.get(action).put("global_feature_specific_allAttrValuesMentioned", 1.0);
                    } else {
                        valueSpecificFeatures.get(action).put("global_feature_specific_allAttrValuesNotMentioned", 1.0);
                    }
                } else {
                    //Is attr in MR?
                    if (MR.getAttributeValues().get(action) != null) {
                        valueSpecificFeatures.get(action).put("global_feature_specific_isInMR", 1.0);
                    } else {
                        valueSpecificFeatures.get(action).put("global_feature_specific_isNotInMR", 1.0);
                    }
                    //Is attr already mentioned right before
                    if (prevAttr.equals(action)) {
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrFollowingSameAttr", 1.0);
                    } else {
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrNotFollowingSameAttr", 1.0);
                    }
                    //Is attr already mentioned
                    attrValuesAlreadyMentioned.stream().map((attrValue) -> {
                        if (attrValue.indexOf('=') == -1) {
                        }
                        return attrValue;
                    }).filter((attrValue) -> (attrValue.substring(0, attrValue.indexOf('=')).equals(action))).forEachOrdered((_item) -> {
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrAlreadyMentioned", 1.0);
                    });
                    //Is attr to be mentioned (has value to express)
                    boolean toBeMentioned = false;
                    for (String attrValue : attrValuesToBeMentioned) {
                        if (attrValue.substring(0, attrValue.indexOf('=')).equals(action)) {
                            toBeMentioned = true;
                            valueSpecificFeatures.get(action).put("global_feature_specific_attrToBeMentioned", 1.0);
                        }
                    }
                    if (!toBeMentioned) {
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrNotToBeMentioned", 1.0);
                    }
                }
                HashSet<String> keys = new HashSet<>(valueSpecificFeatures.get(action).keySet());
                keys.forEach((feature1) -> {
                    keys.stream().filter((feature2) -> (valueSpecificFeatures.get(action).get(feature1) == 1.0
                            && valueSpecificFeatures.get(action).get(feature2) == 1.0
                            && feature1.compareTo(feature2) < 0)).forEachOrdered((feature2) -> {
                        valueSpecificFeatures.get(action).put(feature1 + "&&" + feature2, 1.0);
                    });
                });

                String nextValue = chooseNextValue(action, attrValuesToBeMentioned);
                if (nextValue.isEmpty() && !action.equals(Action.TOKEN_END)) {
                    valueSpecificFeatures.get(action).put("global_feature_LMAttr_score", 0.0);
                } else {
                    ArrayList<String> fullGramLM = new ArrayList<>();
                    for (int i = 0; i < mentionedAttrValues.size(); i++) {
                        fullGramLM.add(mentionedAttrValues.get(i));
                    }
                    ArrayList<String> prev5attrValueGramLM = new ArrayList<>();
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
                    while (prev5attrValueGramLM.size() < 4) {
                        prev5attrValueGramLM.add(0, "@@");
                    }

                    double afterLMScore = getContentLMsPerPredicate().get(predicate).getProbability(prev5attrValueGramLM);
                    valueSpecificFeatures.get(action).put("global_feature_LMAttr_score", afterLMScore);

                    afterLMScore = getContentLMsPerPredicate().get(predicate).getProbability(fullGramLM);
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
    @Override
    public Instance createWordInstance(String predicate, Action bestAction, ArrayList<String> previousGeneratedAttributes, ArrayList<Action> previousGeneratedWords, ArrayList<String> nextGeneratedAttributes, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesThatFollow, boolean wasValueMentioned, HashMap<String, HashSet<Action>> availableWordActions) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
        if (!bestAction.getWord().trim().isEmpty()) {
            //COSTS
            String attr = bestAction.getAttribute();
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
    @Override
    public Instance createWordInstanceWithCosts(String predicate, String currentAttrValue, TObjectDoubleHashMap<String> costs, ArrayList<String> generatedAttributes, ArrayList<Action> previousGeneratedWords, ArrayList<String> nextGeneratedAttributes, HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesThatFollow, boolean wasValueMentioned, HashMap<String, HashSet<Action>> availableWordActions) {
        String currentAttr = currentAttrValue;
        String currentValue = "";
        if (currentAttr.contains("=")) {
            currentAttr = currentAttrValue.substring(0, currentAttrValue.indexOf('='));
            currentValue = currentAttrValue.substring(currentAttrValue.indexOf('=') + 1);
        }
        if (currentValue.contains(":")) {
            currentValue = currentAttrValue.substring(currentAttrValue.indexOf(':') + 1);
        }
        if (currentValue.isEmpty()) {
            //System.exit(0);
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
        attrValuesThatFollow.stream().map((attrValue) -> {
            generalFeatures.put("feature_attrValuesThatFollow_" + attrValue.toLowerCase(), 1.0);
            return attrValue;
        }).forEachOrdered((attrValue) -> {
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
        });
        if (valuesThatFollow.isEmpty()) {
            generalFeatures.put("feature_noAttrsFollow", 1.0);
        } else {
            generalFeatures.put("feature_noAttrsFollow", 0.0);
        }
        HashSet<String> mentionedValues = new HashSet<>();
        attrValuesAlreadyMentioned.stream().map((attrValue) -> {
            generalFeatures.put("feature_attrValuesAlreadyMentioned_" + attrValue.toLowerCase(), 1.0);
            return attrValue;
        }).forEachOrdered((attrValue) -> {
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
        });

        /*System.out.println("currentAttrValue: " + currentAttrValue);
        System.out.println("5W: " + prev5gram);
        System.out.println("5AW: " + prevAttrWord5gram);
        System.out.println("5A: " + prevAttr5gram);
        System.out.println("VM: " + wasValueMentioned);
        System.out.println("A_TF: " + attrValuesThatFollow);
        System.out.println("==============================");*/
        if (currentValue.equals("no")
                || currentValue.equals("yes")
                || currentValue.equals("yes or no")
                || currentValue.equals("none")
                || currentValue.equals("empty") //|| currentValue.equals("dont_care")
                ) {
            generalFeatures.put("feature_emptyValue", 1.0);
        }

        //Word specific features (and also global features)
        for (Action action : availableWordActions.get(currentAttr)) {
            //Is word same as previous word
            if (prevWord.equals(action.getWord())) {
                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_sameAsPreviousWord", 1.0);
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_sameAsPreviousWord", 1.0);
            } else {
                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_notSameAsPreviousWord", 1.0);
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_notSameAsPreviousWord", 1.0);
            }
            //Has word appeared in the same attrValue before
            generatedWords.forEach((previousAction) -> {
                if (previousAction.getWord().equals(action.getWord())
                        && previousAction.getAttribute().equals(currentAttrValue)) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_appearedInSameAttrValue", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_appearedInSameAttrValue", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_notAppearedInSameAttrValue", 1.0);
                    //valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_notAppearedInSameAttrValue", 1.0);
                }
            });
            //Has word appeared before
            generatedWords.forEach((previousAction) -> {
                if (previousAction.getWord().equals(action.getWord())) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_appeared", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_appeared", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_notAppeared", 1.0);
                    //valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_notAppeared", 1.0);
                }
            });
            if (currentValue.equals("no")
                    || currentValue.equals("yes")
                    || currentValue.equals("yes or no")
                    || currentValue.equals("none")
                    || currentValue.equals("empty") //|| currentValue.equals("dont_care")
                    ) {
                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_emptyValue", 1.0);
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_emptyValue", 1.0);
            } else {
                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_notEmptyValue", 1.0);
                //valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_notEmptyValue", 1.0);
            }

            HashSet<String> keys = new HashSet<>(valueSpecificFeatures.get(action.getAction()).keySet());
            keys.forEach((feature1) -> {
                keys.stream().filter((feature2) -> (valueSpecificFeatures.get(action.getAction()).get(feature1) == 1.0
                        && valueSpecificFeatures.get(action.getAction()).get(feature2) == 1.0
                        && feature1.compareTo(feature2) < 0)).forEachOrdered((feature2) -> {
                    valueSpecificFeatures.get(action.getAction()).put(feature1 + "&&" + feature2, 1.0);
                });
            });

            if (!action.getWord().startsWith(Action.TOKEN_X)
                    && !currentValue.equals("no")
                    && !currentValue.equals("yes")
                    && !currentValue.equals("yes or no")
                    && !currentValue.equals("none")
                    && !currentValue.equals("empty") //&& !currentValue.equals("dont_care")
                    ) {
                for (String value : getValueAlignments().keySet()) {
                    for (ArrayList<String> alignedStr : getValueAlignments().get(value).keySet()) {
                        if (alignedStr.get(0).equals(action.getWord())) {
                            if (mentionedValues.contains(value)) {
                                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_beginsValue_alreadyMentioned", 1.0);
                                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_beginsValue_alreadyMentioned", 1.0);

                            } else if (currentValue.equals(value)) {
                                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_beginsValue_current", 1.0);
                                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_beginsValue_current", 1.0);

                            } else if (valuesThatFollow.contains(value)) {
                                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_beginsValue_thatFollows", 1.0);
                                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_beginsValue_thatFollows", 1.0);

                            } else {
                                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_beginsValue_notInMR", 1.0);
                                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_beginsValue_notInMR", 1.0);

                            }
                        } else {
                            for (int i = 1; i < alignedStr.size(); i++) {
                                if (alignedStr.get(i).equals(action.getWord())) {
                                    if (endsWith(generatedPhrase, new ArrayList<String>(alignedStr.subList(0, i + 1)))) {
                                        if (mentionedValues.contains(value)) {
                                            //valueSpecificFeatures.get(action.getAction()).put("feature_specific_inValue_alreadyMentioned", 1.0);
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_inValue_alreadyMentioned", 1.0);

                                        } else if (currentValue.equals(value)) {
                                            //valueSpecificFeatures.get(action.getAction()).put("feature_specific_inValue_current", 1.0);
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_inValue_current", 1.0);

                                        } else if (valuesThatFollow.contains(value)) {
                                            //valueSpecificFeatures.get(action.getAction()).put("feature_specific_inValue_thatFollows", 1.0);
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_inValue_thatFollows", 1.0);

                                        } else {
                                            //valueSpecificFeatures.get(action.getAction()).put("feature_specific_inValue_notInMR", 1.0);
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_inValue_notInMR", 1.0);

                                        }
                                    } else {
                                        /*if (mentionedValues.contains(value)) {
                                        valueSpecificFeatures.get(action.getAction()).put("feature_specific_outOfValue_alreadyMentioned", 1.0);
                                        } else if (currentValue.equals(value)) {
                                        valueSpecificFeatures.get(action.getAction()).put("feature_specific_outOfValue_current", 1.0);
                                        } else if (valuesThatFollow.contains(value)) {
                                        valueSpecificFeatures.get(action.getAction()).put("feature_specific_outOfValue_thatFollows", 1.0);
                                        } else {
                                        valueSpecificFeatures.get(action.getAction()).put("feature_specific_outOfValue_notInMR", 1.0);
                                        }*/
                                        //valueSpecificFeatures.get(action.getAction()).put("feature_specific_outOfValue", 1.0);
                                        valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_outOfValue", 1.0);
                                    }
                                }
                            }
                        }
                    }
                }
                if (action.getWord().equals(Action.TOKEN_END)) {
                    if (generatedWordsInSameAttrValue.isEmpty()) {
                        //valueSpecificFeatures.get(action.getAction()).put("feature_specific_closingEmptyAttr", 1.0);
                        valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_closingEmptyAttr", 1.0);
                    }
                    if (!wasValueMentioned) {
                        //valueSpecificFeatures.get(action.getAction()).put("feature_specific_closingAttrWithValueNotMentioned", 1.0);
                        valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_closingAttrWithValueNotMentioned", 1.0);
                    }

                    // if (!prevCurrentAttrValueWord.equals("@@")) {
                    if (!prevWord.equals("@@")) {
                        boolean alignmentIsOpen = false;
                        for (String value : getValueAlignments().keySet()) {
                            for (ArrayList<String> alignedStr : getValueAlignments().get(value).keySet()) {
                                for (int i = 0; i < alignedStr.size() - 1; i++) {
                                    if (alignedStr.get(i).equals(prevWord)
                                            && endsWith(generatedPhrase, new ArrayList<>(alignedStr.subList(0, i + 1)))) {
                                        alignmentIsOpen = true;
                                    }
                                }
                            }
                        }
                        if (alignmentIsOpen) {
                            // valueSpecificFeatures.get(action.getAction()).put("feature_specific_closingAttrWhileValueIsNotConcluded", 1.0);
                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_closingAttrWhileValueIsNotConcluded", 1.0);
                        }
                    }
                }
            } else if (currentValue.equals("no")
                    || currentValue.equals("yes")
                    || currentValue.equals("yes or no")
                    || currentValue.equals("none")
                    || currentValue.equals("empty") //|| currentValue.equals("dont_care")
                    ) {
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_notInMR", 1.0);
            } else {
                String currentValueVariant = "";
                if (currentValue.matches("[xX][0-9]+")) {
                    currentValueVariant = Action.TOKEN_X + currentAttr + "_" + currentValue.substring(1);
                }

                if (mentionedValues.contains(action.getWord())) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_XValue_alreadyMentioned", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_alreadyMentioned", 1.0);
                } else if (currentValueVariant.equals(action.getWord())
                        && !currentValueVariant.isEmpty()) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_XValue_current", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_current", 1.0);

                } else if (valuesThatFollow.contains(action.getWord())) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_XValue_thatFollows", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_thatFollows", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_XValue_notInMR", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_notInMR", 1.0);
                }
            }
            /*for (int i : nGrams.keySet()) {
            for (String nGram : nGrams.get(i)) {
            if (i == 2) {
            if (nGram.startsWith(prevWord + "|")
            && nGram.endsWith("|" + action.getAction())) {
            valueSpecificFeatures.get(action.getAction()).put("feature_specific_valuesFollowsPreviousWord", 1.0);
            }
            } else if (i == 3) {
            if (nGram.startsWith(prevBigram + "|")
            && nGram.endsWith("|" + action.getAction())) {
            valueSpecificFeatures.get(action.getAction()).put("feature_specific_valuesFollowsPreviousBigram", 1.0);
            }
            } else if (i == 4) {
            if (nGram.startsWith(prevTrigram + "|")
            && nGram.endsWith("|" + action.getAction())) {
            valueSpecificFeatures.get(action.getAction()).put("feature_specific_valuesFollowsPreviousTrigram", 1.0);
            }
            } else if (i == 5) {
            if (nGram.startsWith(prev4gram + "|")
            && nGram.endsWith("|" + action.getAction())) {
            valueSpecificFeatures.get(action.getAction()).put("feature_specific_valuesFollowsPrevious4gram", 1.0);
            }
            } else if (i == 6) {
            if (nGram.startsWith(prev5gram + "|")
            && nGram.endsWith("|" + action.getAction())) {
            valueSpecificFeatures.get(action.getAction()).put("feature_specific_valuesFollowsPrevious5gram", 1.0);
            }
            }
            }
            }*/

            //valueSpecificFeatures.get(action.getAction()).put("global_feature_abstractMR_" + mr.getAbstractMR(), 1.0);
            valueSpecificFeatures.get(action.getAction()).put("global_feature_currentValue_" + currentValue.toLowerCase(), 1.0);

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
            while (prev5wordGramLM.size() < 4) {
                prev5wordGramLM.add(0, "@@");
            }

            double afterLMScorePerPred5Gram = getWordLMsPerPredicate().get(predicate).getProbability(prev5wordGramLM);
            valueSpecificFeatures.get(action.getAction()).put("global_feature_LMWord_perPredicate_5gram_score", afterLMScorePerPred5Gram);

            double afterLMScorePerPred = getWordLMsPerPredicate().get(predicate).getProbability(fullGramLM);
            valueSpecificFeatures.get(action.getAction()).put("global_feature_LMWord_perPredicate_score", afterLMScorePerPred);
        }

        /*HashSet<String> keys = new HashSet<>(generalFeatures.keySet());
        for (String feature1 : keys) {
            if (generalFeatures.get(feature1) == 1.0) {
                generalFeatures.put("global_feature_attr_" + currentValue.toLowerCase() + "&&" + feature1, 1.0);
            }
        }*/
        //generalFeatures.put("feature_abstractMR_" + mr.getAbstractMR(), 1.0);

        /*HashSet<String> keys = new HashSet<>(generalFeatures.keySet());
        for (String feature1 : keys) {
        for (String feature2 : keys) {
        if (generalFeatures.get(feature1) == 1.0
        && generalFeatures.get(feature2) == 1.0
        && feature1.compareTo(feature2) < 0) {
        generalFeatures.put(feature1 + "&&" + feature2, 1.0);
        }
        }
        }*/
        return new Instance(generalFeatures, valueSpecificFeatures, costs);
    }

    /**
     *
     * @param di
     * @param wordSequence
     * @return
     */
    @Override
    public String postProcessWordSequence(DatasetInstance di, ArrayList<Action> wordSequence) {
        HashSet<ArrayList<Action>> matched = new HashSet<>();
        ArrayList<Action> processedWordSequence = new ArrayList<>();
        wordSequence.forEach((act) -> {
            processedWordSequence.add(new Action(act));
        });
        if (!processedWordSequence.isEmpty()
                && processedWordSequence.get(processedWordSequence.size() - 1).getWord().equals(Action.TOKEN_END)
                && processedWordSequence.get(processedWordSequence.size() - 1).getAttribute().equals(Action.TOKEN_END)) {
            processedWordSequence.remove(processedWordSequence.size() - 1);
        }
        if (getPunctuationPatterns().containsKey(di.getMeaningRepresentation().getPredicate())) {
            getPunctuationPatterns().get(di.getMeaningRepresentation().getPredicate()).keySet().forEach((surrounds) -> {
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
                        processedWordSequence.add(i + 2, getPunctuationPatterns().get(di.getMeaningRepresentation().getPredicate()).get(surrounds));
                    }
                }
            });
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
        processedWordSequence.stream().filter((action) -> (!action.getWord().equals(Action.TOKEN_START)
                && !action.getWord().equals(Action.TOKEN_END))).forEachOrdered((action) -> {
            cleanActionList.add(action);
        });

        String predictedWordSequence = " ";
        boolean previousIsTokenX = false;
        for (Action action : cleanActionList) {
            if (action.getWord().startsWith(Action.TOKEN_X)) {
                predictedWordSequence += " " + di.getMeaningRepresentation().getDelexicalizationMap().get(action.getWord());
                previousIsTokenX = true;
            } else {
                if (action.getWord().equals("-ly") && previousIsTokenX) {
                    predictedWordSequence += "ly";
                } else if (action.getWord().equals("s") && previousIsTokenX) {
                    predictedWordSequence += action.getWord();
                } else {
                    predictedWordSequence += " " + action.getWord();
                }
                previousIsTokenX = false;
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
        }
        return predictedWordSequence;
    }

    /**
     *
     * @param mr
     * @param refSeq
     * @return
     */
    @Override
    public String postProcessRef(MeaningRepresentation mr, ArrayList<Action> refSeq) {
        String cleanedWords = "";
        boolean previousIsTokenX = false;
        for (Action nlWord : refSeq) {
            if (!nlWord.equals(new Action(Action.TOKEN_START, ""))
                    && !nlWord.equals(new Action(Action.TOKEN_END, ""))) {
                if (nlWord.getWord().startsWith(Action.TOKEN_X)) {
                    cleanedWords += " " + mr.getDelexicalizationMap().get(nlWord.getWord());
                    previousIsTokenX = true;
                } else {
                    if (nlWord.getWord().equals("-ly") && previousIsTokenX) {
                        cleanedWords += "ly";
                    } else if (nlWord.getWord().equals("s") && previousIsTokenX) {
                        cleanedWords += nlWord.getWord();
                    } else {
                        cleanedWords += " " + nlWord.getWord();
                    }
                    previousIsTokenX = false;
                }
            }
        }
        cleanedWords = cleanedWords.trim();
        if (mr.getPredicate().startsWith("?")
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

    /**
     *
     * @param wordSequence
     * @return
     */
    public ArrayList<String> getPredictedAttrList(ArrayList<Action> wordSequence) {
        ArrayList<Action> cleanActionList = new ArrayList<>();
        wordSequence.stream().filter((action) -> (!action.getWord().equals(Action.TOKEN_START)
                && !action.getWord().equals(Action.TOKEN_END))).forEachOrdered((action) -> {
            cleanActionList.add(action);
        });

        ArrayList<String> predictedAttrList = new ArrayList<>();
        cleanActionList.forEach((action) -> {
            if (predictedAttrList.isEmpty()) {
                predictedAttrList.add(action.getAttribute());
            } else if (!predictedAttrList.get(predictedAttrList.size() - 1).equals(action.getAttribute())) {
                predictedAttrList.add(action.getAttribute());
            }
        });
        return predictedAttrList;
    }

    /**
     *
     * @return
     */
    public boolean loadLists() {
        String file1 = "cache/getPredicates()_SF_" + getDataset();
        String file2 = "cache/attributes_SF_" + getDataset();
        String file3 = "cache/attributeValuePairs_SF_" + getDataset();
        String file4 = "cache/getValueAlignments()_SF_" + getDataset();
        String file5 = "cache/getDatasetInstances()_SF_" + getDataset();
        String file6 = "cache/maxLengths_SF_" + getDataset();
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
                if (getPredicates() == null) {
                    if (o1 instanceof ArrayList) {
                        setPredicates(new ArrayList<String>((Collection<? extends String>) o1));
                    }
                } else if (o1 instanceof ArrayList) {
                    getPredicates().addAll((Collection<? extends String>) o1);
                }
                ///////////////////
                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (getAttributes() == null) {
                    if (o2 instanceof HashMap) {
                        setAttributes(new HashMap<String, HashSet<String>>((Map<? extends String, ? extends HashSet<String>>) o2));
                    }
                } else if (o2 instanceof HashMap) {
                    getAttributes().putAll((Map<? extends String, ? extends HashSet<String>>) o2);
                }
                ///////////////////
                fin3 = new FileInputStream(file3);
                ois3 = new ObjectInputStream(fin3);
                Object o3 = ois3.readObject();
                if (getAttributeValuePairs() == null) {
                    if (o3 instanceof HashMap) {
                        setAttributeValuePairs(new HashMap<String, HashSet<String>>((Map<? extends String, ? extends HashSet<String>>) o3));
                    }
                } else if (o3 instanceof HashMap) {
                    getAttributeValuePairs().putAll((Map<? extends String, ? extends HashSet<String>>) o3);
                }
                ///////////////////
                fin4 = new FileInputStream(file4);
                ois4 = new ObjectInputStream(fin4);
                Object o4 = ois4.readObject();
                if (getValueAlignments() == null) {
                    if (o4 instanceof HashMap) {
                        setValueAlignments(new HashMap<String, HashMap<ArrayList<String>, Double>>((Map<? extends String, ? extends HashMap<ArrayList<String>, Double>>) o4));
                    }
                } else if (o4 instanceof HashMap) {
                    getValueAlignments().putAll((Map<? extends String, ? extends HashMap<ArrayList<String>, Double>>) o4);
                }
                ///////////////////
                fin5 = new FileInputStream(file5);
                ois5 = new ObjectInputStream(fin5);
                Object o5 = ois5.readObject();
                if (getDatasetInstances() == null) {
                    if (o5 instanceof HashMap) {
                        setDatasetInstances(new HashMap<String, ArrayList<DatasetInstance>>((Map<? extends String, ? extends ArrayList<DatasetInstance>>) o5));
                    }
                } else if (o5 instanceof HashMap) {
                    getDatasetInstances().putAll((Map<? extends String, ? extends ArrayList<DatasetInstance>>) o5);
                }
                ///////////////////
                fin6 = new FileInputStream(file6);
                ois6 = new ObjectInputStream(fin6);
                Object o6 = ois6.readObject();
                ArrayList<Integer> lengths = new ArrayList<Integer>((Collection<? extends Integer>) o6);
                setMaxContentSequenceLength(lengths.get(0));
                setMaxWordSequenceLength(lengths.get(1));

                System.out.println("done!");
            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin1.close();
                } catch (IOException ex) {
                }
                try {
                    ois1.close();
                } catch (IOException ex) {
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
        String file1 = "cache/getPredicates()_SF_" + getDataset();
        String file2 = "cache/attributes_SF_" + getDataset();
        String file3 = "cache/attributeValuePairs_SF_" + getDataset();
        String file4 = "cache/getValueAlignments()_SF_" + getDataset();
        String file5 = "cache/getDatasetInstances()_SF_" + getDataset();
        String file6 = "cache/maxLengths_SF_" + getDataset();
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
            oos1.writeObject(getPredicates());
            ///////////////////
            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(getAttributes());
            ///////////////////
            fout3 = new FileOutputStream(file3);
            oos3 = new ObjectOutputStream(fout3);
            oos3.writeObject(getAttributeValuePairs());
            ///////////////////
            fout4 = new FileOutputStream(file4);
            oos4 = new ObjectOutputStream(fout4);
            oos4.writeObject(getValueAlignments());
            ///////////////////
            fout5 = new FileOutputStream(file5);
            oos5 = new ObjectOutputStream(fout5);
            oos5.writeObject(getDatasetInstances());
            ///////////////////
            fout6 = new FileOutputStream(file6);
            oos6 = new ObjectOutputStream(fout6);
            ArrayList<Integer> lengths = new ArrayList<Integer>();
            lengths.add(getMaxContentSequenceLength());
            lengths.add(getMaxWordSequenceLength());
            oos6.writeObject(lengths);
        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
                fout3.close();
                fout4.close();
                fout5.close();
                fout6.close();
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
                oos3.close();
                oos4.close();
                oos5.close();
                oos6.close();
            } catch (IOException ex) {
            }
        }
    }

    /**
     *
     * @return
     */
    public boolean loadLMs() {
        String file2 = "cache/wordLMs_SF_" + getDataset();
        String file3 = "cache/attrLMs_SF_" + getDataset();
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        FileInputStream fin3 = null;
        ObjectInputStream ois3 = null;
        if ((new File(file2)).exists()
                && (new File(file3)).exists()) {
            try {
                System.out.print("Load language models...");
                
                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (getWordLMsPerPredicate() == null) {
                    if (o2 instanceof HashMap) {
                        setWordLMsPerPredicate(new HashMap<String, SimpleLM>((Map<? extends String, ? extends SimpleLM>) o2));
                    }
                } else if (o2 instanceof HashMap) {
                    getWordLMsPerPredicate().putAll((Map<? extends String, ? extends SimpleLM>) o2);
                }

                fin3 = new FileInputStream(file3);
                ois3 = new ObjectInputStream(fin3);
                Object o3 = ois3.readObject();
                if (getContentLMsPerPredicate() == null) {
                    if (o3 instanceof HashMap) {
                        setContentLMsPerPredicate(new HashMap<String, SimpleLM>((Map<? extends String, ? extends SimpleLM>) o3));
                    }
                } else if (o3 instanceof HashMap) {
                    getContentLMsPerPredicate().putAll((Map<? extends String, ? extends SimpleLM>) o3);
                }

            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin2.close();
                    fin3.close();
                } catch (IOException ex) {
                }
                try {
                    ois2.close();
                    ois3.close();
                } catch (IOException ex) {
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
        String file2 = "cache/wordLMs_SF_" + getDataset();
        String file3 = "cache/attrLMs_SF_" + getDataset();
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        FileOutputStream fout3 = null;
        ObjectOutputStream oos3 = null;
        try {
            System.out.print("Write LMs...");
            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(getWordLMsPerPredicate());

            fout3 = new FileOutputStream(file3);
            oos3 = new ObjectOutputStream(fout3);
            oos3.writeObject(getContentLMsPerPredicate());
        } catch (IOException ex) {
        } finally {
            try {
                fout2.close();
                fout3.close();
            } catch (IOException ex) {
            }
            try {
                oos2.close();
                oos3.close();
            } catch (IOException ex) {
            }
        }
    }

    /**
     *
     * @param dataSize
     * @return
     */
    public boolean loadTrainingData(int dataSize) {
        String file1 = "cache/attrTrainingData" + getDataset() + "_" + dataSize;
        String file2 = "cache/wordTrainingData" + getDataset() + "_" + dataSize;
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
                if (getPredicateContentTrainingData() == null) {
                    if (o1 instanceof HashMap) {
                        setPredicateContentTrainingData(new HashMap<String, ArrayList<Instance>>((Map<? extends String, ? extends ArrayList<Instance>>) o1));
                    }
                } else if (o1 instanceof HashMap) {
                    getPredicateContentTrainingData().putAll((Map<? extends String, ? extends ArrayList<Instance>>) o1);
                }

                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (getPredicateWordTrainingData() == null) {
                    if (o2 instanceof HashMap) {
                        setPredicateWordTrainingData(new HashMap<String, HashMap<String, ArrayList<Instance>>>((Map<? extends String, ? extends HashMap<String, ArrayList<Instance>>>) o2));
                    }
                } else if (o2 instanceof HashMap) {
                    getPredicateWordTrainingData().putAll((Map<? extends String, ? extends HashMap<String, ArrayList<Instance>>>) o2);
                }

            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                } catch (IOException ex) {
                }
                try {
                    ois1.close();
                    ois2.close();
                } catch (IOException ex) {
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
     */
    public void writeTrainingData(int dataSize) {
        String file1 = "cache/attrTrainingData" + getDataset() + "_" + dataSize;
        String file2 = "cache/wordTrainingData" + getDataset() + "_" + dataSize;
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        try {
            System.out.print("Write Training Data...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(getPredicateContentTrainingData());

            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(getPredicateWordTrainingData());

        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
            } catch (IOException ex) {
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
    @Override
    public boolean loadInitClassifiers(int dataSize, HashMap<String, JAROW> trainedAttrClassifiers_0, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0) {
        String file1 = "cache/attrInitClassifiers" + getDataset() + "_" + dataSize;
        String file2 = "cache/wordInitClassifiers" + getDataset() + "_" + dataSize;
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
                if (o1 instanceof HashMap) {
                    trainedAttrClassifiers_0.putAll((Map<? extends String, ? extends JAROW>) o1);
                }

                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (o2 instanceof HashMap) {
                    trainedWordClassifiers_0.putAll((Map<? extends String, ? extends HashMap<String, JAROW>>) o2);
                }

            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                } catch (IOException ex) {
                }
                try {
                    ois1.close();
                    ois2.close();
                } catch (IOException ex) {
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
    @Override
    public void writeInitClassifiers(int dataSize, HashMap<String, JAROW> trainedAttrClassifiers_0, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0) {
        String file1 = "cache/attrInitClassifiers" + getDataset() + "_" + dataSize;
        String file2 = "cache/wordInitClassifiers" + getDataset() + "_" + dataSize;
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

        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
            } catch (IOException ex) {
            }
        }
    }

}

/*
 * Thread that infers the cost and feature vectors for a particular DatasetInstance action sequence
*/
class InferSFXVectorsThread extends Thread {
    DatasetInstance di;
    SFX SFX;
    ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>> predicateContentTrainingData;
    ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData;

    InferSFXVectorsThread(DatasetInstance di, SFX SFX, ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>> predicateContentTrainingData, ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData) {
        this.di = di;
        this.SFX = SFX;

        this.predicateContentTrainingData = predicateContentTrainingData;
        this.predicateWordTrainingData = predicateWordTrainingData;
    }

    /**
     * This method goes through the ActionSequence one time-step at the time, and creates a feature and cost vector for each one.
     * Meanwhile it tracks the context information that the feature vector requires.
     */
    @Override
    public void run() {
        String predicate = di.getMeaningRepresentation().getPredicate();
        ArrayList<Action> refSequence = di.getDirectReferenceSequence();
        
        //Collections to track which attribute/value pairs have already be mentioned in the sequence and which are yet to be mentioned
        HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
        HashSet<String> attrValuesToBeMentioned = new HashSet<>();
        for (String attribute : di.getMeaningRepresentation().getAttributeValues().keySet()) {
            for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
            }
        }
        if (attrValuesToBeMentioned.isEmpty()) {
            attrValuesToBeMentioned.add("empty=empty");
        }

        // First we create the feature and cost vectors for the content actions
        ArrayList<String> attributeSequence = new ArrayList<>();
        String attrValue = "";
        // For every step of the sequence
        for (int w = 0; w < refSequence.size(); w++) {
            if (!refSequence.get(w).getAttribute().equals(Action.TOKEN_PUNCT)
                    && !refSequence.get(w).getAttribute().equals(attrValue)) {
                if (!attrValue.isEmpty()) {
                    attrValuesToBeMentioned.remove(attrValue);
                }
                // Create the feature and cost vector
                Instance contentTrainingVector = SFX.createContentInstance(predicate, refSequence.get(w).getAttribute(), attributeSequence, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), SFX.getAvailableContentActions());
                if (contentTrainingVector != null) {
                    predicateContentTrainingData.get(di).get(predicate).add(contentTrainingVector);
                }
                attributeSequence.add(refSequence.get(w).getAttribute());

                attrValue = refSequence.get(w).getAttribute();
                if (!attrValue.isEmpty()) {
                    attrValuesAlreadyMentioned.add(attrValue);
                    attrValuesToBeMentioned.remove(attrValue);
                }
            }
        }

        // Reset the tracking collections
        attrValuesAlreadyMentioned = new HashSet<>();
        attrValuesToBeMentioned = new HashSet<>();
        for (String attribute : di.getMeaningRepresentation().getAttributeValues().keySet()) {
            for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
            }
        }
        if (attrValuesToBeMentioned.isEmpty()) {
            attrValuesToBeMentioned.add("empty=empty");
        }

        // Then we create the feature and cost vectors for the word actions
        // Each word action corresponds to a content action, so we need to keep track of which content action we are "generating" from at each timestep
        ArrayList<String> attrs = new ArrayList<>();
        boolean isValueMentioned = false;
        // The value that we currently need to mention
        String valueTBM = "";
        // These track the content (attribute/value pairs)
        attrValue = "";
        // Time-step counter
        int a = -1;
        // This tracks the subphrase consisting of the words generated for the current content action
        ArrayList<String> subPhrase = new ArrayList<>();
        // For every step of the sequence
        for (int w = 0; w < refSequence.size(); w++) {
            if (!refSequence.get(w).getAttribute().equals(Action.TOKEN_PUNCT)) {
                // If this action does not belong to the current content, we need to update the trackers and switch to the new content action
                if (!refSequence.get(w).getAttribute().equals(attrValue)) {
                    a++;
                    if (!attrValue.isEmpty()) {
                        attrValuesToBeMentioned.remove(attrValue);
                    }
                    attrs.add(refSequence.get(w).getAttribute());

                    attrValue = refSequence.get(w).getAttribute();
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

                // If it's not the end of the ActionSequence
                if (!attrValue.equals(Action.TOKEN_END)) {
                    // The subsequence of content actions we have generated for so far
                    ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                    for (int i = 0; i < attrs.size() - 1; i++) {
                        predictedAttributesForInstance.add(attrs.get(i));
                    }
                    // ...exclusive of the current content action
                    if (!attrs.get(attrs.size() - 1).equals(attrValue)) {
                        predictedAttributesForInstance.add(attrs.get(attrs.size() - 1));
                    }
                    // The subsequence of content actions we will generated for after the current content action
                    ArrayList<String> nextAttributesForInstance = new ArrayList<>(attributeSequence.subList(a + 1, attributeSequence.size()));
                    // Create the feature and cost vector
                    Instance wordTrainingVector = SFX.createWordInstance(predicate, refSequence.get(w), predictedAttributesForInstance, new ArrayList<>(refSequence.subList(0, w)), nextAttributesForInstance, attrValuesAlreadyMentioned, attrValuesToBeMentioned, isValueMentioned, SFX.getAvailableWordActions().get(predicate));

                    if (wordTrainingVector != null) {
                        String attribute = attrValue;
                        if (attribute.contains("=")) {
                            attribute = attrValue.substring(0, attrValue.indexOf('='));
                        }
                        if (!predicateWordTrainingData.get(di).containsKey(predicate)) {
                            predicateWordTrainingData.get(di).put(predicate, new HashMap<String, ArrayList<Instance>>());
                        }
                        if (!predicateWordTrainingData.get(di).get(predicate).containsKey(attribute)) {
                            predicateWordTrainingData.get(di).get(predicate).put(attribute, new ArrayList<Instance>());
                        }
                        predicateWordTrainingData.get(di).get(predicate).get(attribute).add(wordTrainingVector);
                        if (!refSequence.get(w).getWord().equals(Action.TOKEN_START)
                                && !refSequence.get(w).getWord().equals(Action.TOKEN_END)) {
                            subPhrase.add(refSequence.get(w).getWord());
                        }
                    }
                    
                    // Check if we have mentioned the value of the current content action
                    if (!isValueMentioned) {
                        // If the value is a variable, we just check if the word action we just generated is that variable
                        if (refSequence.get(w).getWord().startsWith(Action.TOKEN_X)
                                && (valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\"")
                                || valueTBM.startsWith(Action.TOKEN_X))) {
                            isValueMentioned = true;
                        // Otherwise
                        } else if (!refSequence.get(w).getWord().startsWith(Action.TOKEN_X)
                                && !(valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\"")
                                || valueTBM.startsWith(Action.TOKEN_X))) {
                            // We form the key for the value, as it appears in the valueAlignments collection
                            String valueToCheck = valueTBM;
                            if (valueToCheck.equals("no")
                                    || valueToCheck.equals("yes")
                                    || valueToCheck.equals("yes or no")
                                    || valueToCheck.equals("none")
                                    || valueToCheck.equals("empty")) {
                                String attribute = attrValue;
                                if (attribute.contains("=")) {
                                    attribute = attrValue.substring(0, attrValue.indexOf('='));
                                }
                                valueToCheck = attribute + ":" + valueTBM;
                            }
                            // We look up the value in all the value alignments we have made during the parsing of the data, and see if it is mentioned in the subphrase
                            // Note that the value may be formed by multiple word actions
                            if (!valueToCheck.equals("empty:empty")
                                    && SFX.getValueAlignments().containsKey(valueToCheck)) {
                                for (ArrayList<String> alignedStr : SFX.getValueAlignments().get(valueToCheck).keySet()) {
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
                    // We also check if we have inadvertedly mentioned some other pending value (not the current one)
                    String mentionedAttrValue = "";
                    if (!refSequence.get(w).getWord().startsWith(Action.TOKEN_X)) {
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
                                            || valueToCheck.equals("empty")) {
                                        valueToCheck = attrValueTBM.replace("=", ":");
                                    }
                                    if (!valueToCheck.equals("empty:empty")
                                            && SFX.getValueAlignments().containsKey(valueToCheck)) {
                                        for (ArrayList<String> alignedStr : SFX.getValueAlignments().get(valueToCheck).keySet()) {
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
        }
    }
}
