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
package jarow;

import imitationNLG.RoboCup;
import imitationNLG.Bagel;
import gnu.trove.map.hash.THashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

public class JAROW implements Serializable {

    private boolean probabilities;
    private THashMap<String, TObjectDoubleHashMap<String>> currentWeightVectors;
    private THashMap<String, TObjectDoubleHashMap<String>> currentVarianceVectors;
    private ArrayList<THashMap<String, TObjectDoubleHashMap<String>>> probWeightVectors;
    private THashMap<String, TObjectDoubleHashMap<String>> averagedWeightVectors;
    private TObjectDoubleHashMap<String> currentGlobalWeightVectors;
    private TObjectDoubleHashMap<String> currentGlobalVarianceVectors;
    private ArrayList<TObjectDoubleHashMap<String>> probGlobalWeightVectors;
    private TObjectDoubleHashMap<String> averagedGlobalWeightVectors;
    private int averagingUpdates = 0;
    private double param;

    public JAROW() {
        this.probabilities = false;
        this.currentWeightVectors = new THashMap<>();
        this.currentVarianceVectors = new THashMap<>();

        //GLOBAL FEATURES ARE FEATURES WHOSE WEIGHTS ARE CALCULATED REGARDLESS THE LABEL
        //FOR FEATURES TO BE TREATED AS GLOBAL THEIR IDENTIFIERS MUST START WITH "global_"
        this.currentGlobalWeightVectors = new TObjectDoubleHashMap<>();
        this.currentGlobalVarianceVectors = new TObjectDoubleHashMap<>();

        this.averagedGlobalWeightVectors = new TObjectDoubleHashMap<>();
    }

    public JAROW(JAROW j) {
        this.probabilities = j.probabilities;
        this.currentWeightVectors = new THashMap<>();
        for (String key : j.currentWeightVectors.keySet()) {
            this.currentWeightVectors.put(key, new TObjectDoubleHashMap<String>(j.currentWeightVectors.get(key)));
        }
        this.currentVarianceVectors = new THashMap<>();
        for (String key : j.currentVarianceVectors.keySet()) {
            this.currentVarianceVectors.put(key, new TObjectDoubleHashMap<String>(j.currentVarianceVectors.get(key)));
        }
        this.currentGlobalWeightVectors = new TObjectDoubleHashMap<String>(j.currentGlobalWeightVectors);
        this.currentGlobalVarianceVectors = new TObjectDoubleHashMap<String>(j.currentGlobalVarianceVectors);

        if (j.probWeightVectors != null) {
            this.probWeightVectors = new ArrayList<>();
            for (THashMap<String, TObjectDoubleHashMap<String>> map : j.probWeightVectors) {
                THashMap<String, TObjectDoubleHashMap<String>> nMap = new THashMap<String, TObjectDoubleHashMap<String>>();
                for (String key : map.keySet()) {
                    nMap.put(key, new TObjectDoubleHashMap<String>(map.get(key)));
                }
                this.probWeightVectors.add(nMap);
            }
        }
        if (j.averagedWeightVectors != null) {
            this.averagedWeightVectors = new THashMap<>();
            for (String key : j.averagedWeightVectors.keySet()) {
                this.averagedWeightVectors.put(key, new TObjectDoubleHashMap<String>(j.averagedWeightVectors.get(key)));
            }
        }
        if (j.probGlobalWeightVectors != null) {
            this.probGlobalWeightVectors = new ArrayList<>();
            for (TObjectDoubleHashMap<String> map : j.probGlobalWeightVectors) {
                TObjectDoubleHashMap<String> nMap = new TObjectDoubleHashMap<String>(map);
                this.probGlobalWeightVectors.add(nMap);
            }

        }
        if (j.averagedGlobalWeightVectors != null) {
            this.averagedGlobalWeightVectors = new TObjectDoubleHashMap<>(j.averagedGlobalWeightVectors);
        }
        this.averagingUpdates = j.averagingUpdates;
        this.param = j.param;
    }

    public TObjectDoubleHashMap<String> getWeightVector(String label) {
        TObjectDoubleHashMap<String> weightVector = new TObjectDoubleHashMap<String>();
        if (currentWeightVectors.containsKey(label)) {
            weightVector.putAll(currentWeightVectors.get(label));
        }
        weightVector.putAll(getCurrentGlobalWeightVectors());

        return weightVector;
    }

    // This predicts always using the current weight vectors
    public Prediction predict(Instance instance) {
        return predict(instance, false, false);
    }
    
    public Prediction predict(Instance instance, boolean verbose/* = false*/, boolean probabilities/* = false*/) {
        //# always add the bias
        instance.getGeneralFeatureVector().put("biasAutoAdded", 1.0);

        Prediction prediction = new Prediction();

        for (String label : currentWeightVectors.keySet()) {
            Double score = dotProduct(instance.getFeatureVector(label), this.getWeightVector(label));
            prediction.getLabel2Score().put(label, score);
            if (Double.compare(score, prediction.getScore()) > 0) {
                prediction.setScore(score);
                prediction.setLabel(label);
            }
        }

        if (verbose) {
            TObjectDoubleHashMap<String> featureVector = instance.getFeatureVector(prediction.getLabel());
            for (String feature : featureVector.keySet()) {
                // keep the feature weights for the predicted label
                ArrayList<Object> fvw = new ArrayList<>();
                fvw.add(feature);
                fvw.add(featureVector.get(feature));
                fvw.add(currentWeightVectors.get(prediction.getLabel()).get(feature));
                fvw.add(currentGlobalWeightVectors.get(feature));

                prediction.getFeatureValueWeights().add(fvw);
            }
        }
        if (probabilities) {
            // if we have probabilistic training
            if (this.probabilities) {
                TObjectDoubleHashMap<String> probPredictions = new TObjectDoubleHashMap<>();
                for (String label : this.probWeightVectors.get(0).keySet()) {
                    // smoothing the probabilities with add 0.01 of 1 out of the vectors
                    probPredictions.put(label, 0.01 / this.probWeightVectors.size());
                }
                // for each of the weight vectors obtained get its prediction
                for (THashMap<String, TObjectDoubleHashMap<String>> probWeightVector : this.probWeightVectors) {
                    Double maxScore = Double.NEGATIVE_INFINITY;
                    String maxLabel = null;
                    for (String label : probWeightVector.keySet()) {
                        TObjectDoubleHashMap<String> weightVector = probWeightVector.get(label);
                        Double score = dotProduct(instance.getFeatureVector(label), weightVector);
                        if (Double.compare(score, maxScore) > 0) {
                            maxScore = score;
                            maxLabel = label;
                        }
                    }
                    // so the winning label adds one vote
                    probPredictions.put(maxLabel, probPredictions.get(maxLabel) + 1);
                }
                // now let's normalize:
                for (String label : probPredictions.keySet()) {
                    Double score = probPredictions.get(label);
                    prediction.getLabel2prob().put(label, score / (double) this.probWeightVectors.size());
                }
                // Also compute the entropy:
                for (Double prob : prediction.getLabel2prob().values()) {
                    if (Double.compare(prob, 0.0) > 0) {
                        prediction.setEntropy(prediction.getEntropy() - prob * (Math.log(prob) / Math.log(2)));
                    }
                }
                // normalize it:
                prediction.setEntropy(prediction.getEntropy() / (Math.log(prediction.getLabel2prob().size()) / Math.log(2)));
            } else {
                System.out.println("Need to obtain weight samples for probability estimates first");
            }
        }

        return prediction;
    }

    // This is just used to optimize the params
    // if probabilities is True we return the ratio for the average entropies, otherwise the loss
    public double batchPredict(ArrayList<Instance> instances) {
        return batchPredict(instances, false);
    }

    public double batchPredict(ArrayList<Instance> instances, boolean probabilities) {
        Double totalCost = 0.0;
        Double sumCorrectEntropies = 0.0;
        Double sumIncorrectEntropies = 0.0;
        Double sumLogProbCorrect = 0.0;
        Double totalCorrects = 0.0;
        Double totalIncorrects = 0.0;
        Double sumEntropies = 0.0;
        Double errors = 0.0;
        for (Instance instance : instances) {
            Prediction prediction = predict(instance, false, probabilities);
            // This is without probabilities, with probabilities we want the average entropy*cost 
            if (probabilities) {
                if (Double.compare(instance.getCosts().get(prediction.getLabel()), 0.0) == 0) {
                    sumLogProbCorrect -= Math.log(prediction.getLabel2prob().get(prediction.getLabel())) / Math.log(2);
                    totalCorrects += instance.getMaxCost();
                    sumEntropies += instance.getMaxCost() * prediction.getEntropy();
                    sumCorrectEntropies += instance.getMaxCost() * prediction.getEntropy();
                } else {
                    Double maxCorrectProb = 0.0;
                    for (String correctLabel : instance.getCorrectLabels()) {
                        if (Double.compare(prediction.getLabel2prob().get(correctLabel), maxCorrectProb) > 0) {
                            maxCorrectProb = prediction.getLabel2prob().get(correctLabel);
                        }
                    }
                    // if maxCorrectProb > 0.0:
                    sumLogProbCorrect -= Math.log(maxCorrectProb) / Math.log(2);
                    // else:
                    //    sumLogProbCorrect = float("inf")
                    totalIncorrects += instance.getMaxCost();
                    sumEntropies += instance.getMaxCost() * (1 - prediction.getEntropy());
                    sumIncorrectEntropies += instance.getMaxCost() * prediction.getEntropy();
                }
            } else {
                // no probs, just keep track of the cost incurred
                if (instance.getCosts().get(prediction.getLabel()) > 0) {
                    totalCost += instance.getCosts().get(prediction.getLabel());
                    //totalCost ++;
                }
            }
        }

        if (probabilities) {
            Double avgCorrectEntropy = sumCorrectEntropies / totalCorrects;
            System.out.println(avgCorrectEntropy);
            Double avgIncorrectEntropy = sumIncorrectEntropies / totalIncorrects;
            System.out.println(avgIncorrectEntropy);
            System.out.println(sumLogProbCorrect);
            return sumLogProbCorrect;
        } else {
            return totalCost;
        }
    }

    public static JAROW trainWithRandomRestarts(ArrayList<Instance> trainingWordInstances, int restarts, boolean averaging, int rounds, Double param, boolean adapt) {
        double minCost = Integer.MAX_VALUE;
        JAROW bestClassifier = null;

        for (int i = 0; i < restarts; i++) {
            JAROW classifierWords = new JAROW();
            Double cost = classifierWords.train(trainingWordInstances, true, true, RoboCup.rounds, param, adapt);

            if (cost < minCost) {
                minCost = cost;
                bestClassifier = classifierWords;
            }
        }
        return bestClassifier;
    }

    // the parameter here is for AROW learning
    // adapt if True is AROW, if False it is passive aggressive-II with prediction-based updates 
    public Double train(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, Double param) {
        return train(instances, averaging, shuffling, rounds, param, true);
    }

    public Double train(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, Double param, boolean adapt) {
        // we first need to go through the dataset to find how many classes

        // Initialize the weight vectors in the beginning of training"
        // we have one variance and one weight vector per class        
        this.param = param;
        this.currentWeightVectors = new THashMap<>();
        if (adapt) {
            this.currentVarianceVectors = new THashMap<>();
        }
        this.currentGlobalWeightVectors = new TObjectDoubleHashMap<>();
        this.currentGlobalVarianceVectors = new TObjectDoubleHashMap<>();
        this.averagedGlobalWeightVectors = new TObjectDoubleHashMap<>();
        if (averaging) {
            this.averagedWeightVectors = new THashMap<>();
            averagingUpdates = 0;
        }
        if (!instances.isEmpty()) {
            HashSet<String> labels = new HashSet<>();
            for (String label : instances.get(0).getCosts().keySet()) {
                labels.add(label);
            }
            for (String label : labels) {
                this.currentWeightVectors.put(label, new TObjectDoubleHashMap<String>());
                // remember: this is sparse in the sense that everything that doesn't have a value, is 1
                // everytime we to do something with it, remember to add 1
                if (adapt) {
                    this.currentVarianceVectors.put(label, new TObjectDoubleHashMap<String>());
                }
                // keep the averaged weight vector
                if (averaging && averagedWeightVectors != null) {
                    averagedWeightVectors.put(label, new TObjectDoubleHashMap<String>());
                }
            }
        }
        // in each iteration        
        for (int r = 0; r < rounds; r++) {
            // shuffle
            if (shuffling) {
                Collections.shuffle(instances, Bagel.r);
            }
            int errorsInRound = 0;
            Double costInRound = 0.0;
            // for each instance
            for (Instance instance : instances) {
                Prediction prediction = this.predict(instance);

                // so if the prediction was incorrect
                // we are no longer large margin, since we are using the loss from the cost-sensitive PA
                if (Double.compare(instance.getCosts().get(prediction.getLabel()), 0) > 0) {
                    errorsInRound += 1;
                    costInRound += instance.getCosts().get(prediction.getLabel());

                    // first we need to get the score for the correct answer
                    // if the instance has more than one correct answer then pick the min
                    Double minCorrectLabelScore = Double.POSITIVE_INFINITY;
                    String minCorrectLabel = null;
                    for (String label : instance.getCorrectLabels()) {
                        Double score = dotProduct(instance.getFeatureVector(label), this.getWeightVector(label));
                        if (Double.compare(score, minCorrectLabelScore) < 0) {
                            minCorrectLabelScore = score;
                            minCorrectLabel = label;
                        }
                    }
                    if (minCorrectLabelScore == Double.POSITIVE_INFINITY) {
                        System.out.println("No correct labels error!");
                        System.out.println(instance.getCorrectLabels());
                        System.exit(0);
                    }

                    // the loss is the scaled margin loss also used by Mejer and Crammer 2010
                    Double loss = prediction.getScore() - minCorrectLabelScore + Math.sqrt(instance.getCosts().get(prediction.getLabel()));
                    //System.out.println(loss + " <> " + prediction.getScore() + " = " + minCorrectLabelScore + " = " + Math.sqrt(instance.getCosts().get(prediction.getLabel())) + " = " + instance.getCosts().get(prediction.getLabel()));
                    if (adapt) {
                        // Calculate the confidence values
                        // first for the predicted label 
                        TObjectDoubleHashMap<String> zVectorPredicted = new TObjectDoubleHashMap<>();
                        TObjectDoubleHashMap<String> zVectorMinCorrect = new TObjectDoubleHashMap<>();
                        TObjectDoubleHashMap<String> featureVectorPredicted = instance.getFeatureVector(prediction.getLabel());
                        for (String feature : featureVectorPredicted.keySet()) {
                            // the variance is either some value that is in the dict or just 1
                            if (!feature.startsWith("global_")) {
                                if (currentVarianceVectors.get(prediction.getLabel()).containsKey(feature)) {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * currentVarianceVectors.get(prediction.getLabel()).get(feature));
                                } else {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                                }
                            } else {
                                if (currentGlobalVarianceVectors.containsKey(feature)) {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * currentGlobalVarianceVectors.get(feature));
                                } else {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                                }
                            }
                        }
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            // then for the minCorrect:
                            if (!feature.startsWith("global_")) {
                                if (currentVarianceVectors.get(minCorrectLabel).containsKey(feature)) {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * currentVarianceVectors.get(minCorrectLabel).get(feature));
                                } else {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                                }
                            } else {
                                if (currentGlobalVarianceVectors.containsKey(feature)) {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * currentGlobalVarianceVectors.get(feature));
                                } else {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        Double confidence = dotProduct(zVectorPredicted, featureVectorPredicted) + dotProduct(zVectorMinCorrect, featureVectorMinCorrect);
                        Double beta = 1.0 / (confidence + param);
                        Double alpha = loss * beta;

                        // update the current weight vectors
                        for (String feature : zVectorPredicted.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.currentWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, -(alpha * zVectorPredicted.get(feature)), -(alpha * zVectorPredicted.get(feature)));
                            } else {
                                this.currentGlobalWeightVectors.adjustOrPutValue(feature, -(alpha * zVectorPredicted.get(feature)), -(alpha * zVectorPredicted.get(feature)));
                            }
                        }
                        for (String feature : zVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.currentWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, (alpha * zVectorMinCorrect.get(feature)), (alpha * zVectorMinCorrect.get(feature)));
                            } else {
                                this.currentGlobalWeightVectors.adjustOrPutValue(feature, (alpha * zVectorMinCorrect.get(feature)), (alpha * zVectorMinCorrect.get(feature)));
                            }
                        }
                        if (averaging && averagedWeightVectors != null) {
                            for (String feature : zVectorPredicted.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    averagedWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                } else {
                                    averagedGlobalWeightVectors.adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                }
                            }
                            for (String feature : zVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    averagedWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                } else {
                                    averagedGlobalWeightVectors.adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        // update the diagonal covariance
                        for (String feature : featureVectorPredicted.keySet()) {
                            // for the predicted
                            if (!feature.startsWith("global_")) {
                                currentVarianceVectors.get(prediction.getLabel()).adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            } else {
                                currentGlobalVarianceVectors.adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            }
                        }
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            // for the minCorrect
                            if (!feature.startsWith("global_")) {
                                currentVarianceVectors.get(minCorrectLabel).adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            } else {
                                currentGlobalVarianceVectors.adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            }
                        }
                    } else {
                        // the squared norm is twice the square of the features since they are the same per class 
                        //The value specific features can be safely ignored since they are not shared between values and would result in 0s anyway
                        Double norm = 2.0 * dotProduct(instance.getGeneralFeatureVector(), instance.getGeneralFeatureVector());
                        Double factor = loss / (norm + 1.0 / (2.0 * param));

                        TObjectDoubleHashMap<String> featureVectorPredicted = instance.getFeatureVector(prediction.getLabel());
                        for (String feature : featureVectorPredicted.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.currentWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            } else {
                                this.currentGlobalWeightVectors.adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            }
                        }
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.currentWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            } else {
                                this.currentGlobalWeightVectors.adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            }
                        }
                        if (averaging && averagedWeightVectors != null) {
                            for (String feature : featureVectorPredicted.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    averagedWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                } else {
                                    averagedGlobalWeightVectors.adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                }
                            }
                            for (String feature : featureVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    averagedWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                } else {
                                    averagedGlobalWeightVectors.adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                    }
                }
                if (averaging) {
                    averagingUpdates++;
                }
            }
        }

        if (averaging && averagedWeightVectors != null) {
            for (String label : this.currentWeightVectors.keySet()) {
                for (String feature : averagedWeightVectors.get(label).keySet()) {
                    this.currentWeightVectors.get(label).put(feature, averagedWeightVectors.get(label).get(feature) / ((double) averagingUpdates));
                }
            }
            for (String feature : averagedGlobalWeightVectors.keySet()) {
                this.currentGlobalWeightVectors.put(feature, averagedGlobalWeightVectors.get(feature) / ((double) averagingUpdates));
            }
        }

        // Compute the final training error:
        int finalTrainingErrors = 0;
        Double finalTrainingCost = 0.0;
        for (Instance instance : instances) {
            Prediction prediction = this.predict(instance);
            if (Double.compare(instance.getCosts().get(prediction.getLabel()), 0.0) > 0) {
                finalTrainingErrors += 1;
                finalTrainingCost += instance.getCosts().get(prediction.getLabel());
            }
        }

        Double finalTrainingErrorRate = (double) finalTrainingErrors / (double) instances.size();
        //System.out.println("Final training error rate=" + finalTrainingErrorRate.toString());
        //System.out.println("Final training cost=" + finalTrainingCost.toString());

        return finalTrainingCost;
    }

    public boolean isInstanceLeadingToFix(Instance instance) {
        Prediction prediction = this.predict(instance);

        if (Double.compare(instance.getCosts().get(prediction.getLabel()), 0) > 0) {
            return true;
        }
        return false;
    }

    public Double trainAdditional(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, double param) {
        return trainAdditional(instances, averaging, shuffling, rounds, true, param);
    }

    public Double trainAdditional(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, boolean adapt, double param) {
        // We do not initialize the weight vectors in the beginning of training, rather keep them as they have already been trained

        // in each iteration        
        for (int r = 0; r < rounds; r++) {
            // shuffle
            if (shuffling) {
                Collections.shuffle(instances, Bagel.r);
            }
            boolean flag2 = false;
            int errorsInRound = 0;
            Double costInRound = 0.0;
            String checkWrong = "";
            String checkCorrect = "";
            // for each instance
            for (Instance instance : instances) {
                Prediction prediction = this.predict(instance);

                // so if the prediction was incorrect
                // we are no longer large margin, since we are using the loss from the cost-sensitive PA
                if (Double.compare(instance.getCosts().get(prediction.getLabel()), 0) > 0) {
                    errorsInRound += 1;
                    costInRound += instance.getCosts().get(prediction.getLabel());

                    // first we need to get the score for the correct answer
                    // if the instance has more than one correct answer then pick the min
                    Double minCorrectLabelScore = Double.POSITIVE_INFINITY;
                    HashMap<String, Double> correctLabels2Score = new HashMap<>();
                    String minCorrectLabel = null;
                    for (String label : instance.getCorrectLabels()) {
                        Double score = dotProduct(instance.getFeatureVector(label), this.getWeightVector(label));
                        correctLabels2Score.put(label, score);
                        if (Double.compare(score, minCorrectLabelScore) < 0) {
                            minCorrectLabelScore = score;
                            minCorrectLabel = label;
                        }
                    }
                    HashSet<String> minCorrectLabels = new HashSet<>();
                    for (String label : correctLabels2Score.keySet()) {
                        if (correctLabels2Score.get(label) == minCorrectLabelScore) {
                            minCorrectLabels.add(label);
                        }
                    }
                    if (minCorrectLabelScore == Double.POSITIVE_INFINITY) {
                        System.out.println("No correct labels error!");
                        System.out.println(instance.getCorrectLabels());
                        System.exit(0);
                    }

                    // the loss is the scaled margin loss also used by Mejer and Crammer 2010
                    Double loss = prediction.getScore() - minCorrectLabelScore + Math.sqrt(instance.getCosts().get(prediction.getLabel()));

                    if (adapt) {
                        // Calculate the confidence values
                        // first for the predicted label 
                        TObjectDoubleHashMap<String> zVectorPredicted = new TObjectDoubleHashMap<>();
                        TObjectDoubleHashMap<String> zVectorMinCorrect = new TObjectDoubleHashMap<>();

                        TObjectDoubleHashMap<String> featureVectorPredicted = instance.getFeatureVector(prediction.getLabel());
                        for (String feature : featureVectorPredicted.keySet()) {
                            // the variance is either some value that is in the dict or just 1
                            if (!feature.startsWith("global_")) {
                                if (currentVarianceVectors.get(prediction.getLabel()).containsKey(feature)) {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * currentVarianceVectors.get(prediction.getLabel()).get(feature));
                                } else {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                                }
                            } else {
                                if (currentGlobalVarianceVectors.containsKey(feature)) {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * currentGlobalVarianceVectors.get(feature));
                                } else {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                                }
                            }
                        }
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getGeneralFeatureVector();
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            // then for the minCorrect:
                            if (!feature.startsWith("global_")) {
                                if (currentVarianceVectors.get(minCorrectLabel).containsKey(feature)) {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * currentVarianceVectors.get(minCorrectLabel).get(feature));
                                } else {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                                }
                            } else {
                                if (currentGlobalVarianceVectors.containsKey(feature)) {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * currentGlobalVarianceVectors.get(feature));
                                } else {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        Double confidence = dotProduct(zVectorPredicted, featureVectorPredicted) + dotProduct(zVectorMinCorrect, featureVectorMinCorrect);
                        Double beta = 1.0 / (confidence + param);
                        Double alpha = loss * beta;

                        // update the current weight vectors
                        for (String feature : zVectorPredicted.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.currentWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, -alpha * zVectorPredicted.get(feature), -alpha * zVectorPredicted.get(feature));
                            } else {
                                this.currentGlobalWeightVectors.adjustOrPutValue(feature, -alpha * zVectorPredicted.get(feature), -alpha * zVectorPredicted.get(feature));
                            }
                        }
                        for (String feature : zVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.currentWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                            } else {
                                this.currentGlobalWeightVectors.adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                            }
                        }
                        if (averaging && averagedWeightVectors != null) {
                            for (String feature : zVectorPredicted.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    averagedWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                } else {
                                    averagedGlobalWeightVectors.adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                }
                            }
                            for (String feature : zVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    averagedWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                } else {
                                    averagedGlobalWeightVectors.adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        // update the diagonal covariance
                        for (String feature : featureVectorPredicted.keySet()) {
                            // for the predicted
                            if (!feature.startsWith("global_")) {
                                currentVarianceVectors.get(prediction.getLabel()).adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            } else {
                                currentGlobalVarianceVectors.adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            }
                        }
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            // for the minCorrect
                            if (!feature.startsWith("global_")) {
                                currentVarianceVectors.get(minCorrectLabel).adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            } else {
                                currentGlobalVarianceVectors.adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            }
                        }
                        for (String s : currentVarianceVectors.get(prediction.getLabel()).keySet()) {
                            if (currentVarianceVectors.get(prediction.getLabel()).get(s) < 0) {
                                System.out.println(currentVarianceVectors.get(prediction.getLabel()));
                            }
                        }
                        for (String s : currentVarianceVectors.get(minCorrectLabel).keySet()) {
                            if (currentVarianceVectors.get(minCorrectLabel).get(s) < 0) {
                                System.out.println(currentVarianceVectors.get(minCorrectLabel));
                            }
                        }
                        for (String s : currentGlobalVarianceVectors.keySet()) {
                            if (currentGlobalVarianceVectors.get(s) < 0) {
                                System.out.println(currentGlobalVarianceVectors);
                            }
                        }
                    } else {
                        // the squared norm is twice the square of the features since they are the same per class 
                        Double norm = 2.0 * dotProduct(instance.getGeneralFeatureVector(), instance.getGeneralFeatureVector());
                        Double factor = loss / (norm + 1.0 / (2.0 * param));

                        TObjectDoubleHashMap<String> featureVectorPredicted = instance.getFeatureVector(prediction.getLabel());
                        for (String feature : featureVectorPredicted.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.currentWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            } else {
                                this.currentGlobalWeightVectors.adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            }
                        }

                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.currentWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            } else {
                                this.currentGlobalWeightVectors.adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            }
                        }
                        if (averaging && averagedWeightVectors != null) {
                            for (String feature : featureVectorPredicted.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    averagedWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                } else {
                                    averagedGlobalWeightVectors.adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                }
                            }
                            for (String feature : featureVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    averagedWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                } else {
                                    averagedGlobalWeightVectors.adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                    }
                }
                if (averaging) {
                    averagingUpdates++;
                }
            }
            //System.out.println("Training error rate in round " + r + " : " + (double) errorsInRound / (double) instances.size());
        }

        if (averaging && averagedWeightVectors != null) {
            for (String label : this.currentWeightVectors.keySet()) {
                for (String feature : averagedWeightVectors.get(label).keySet()) {
                    this.currentWeightVectors.get(label).put(feature, averagedWeightVectors.get(label).get(feature) / ((double) averagingUpdates));
                }
            }
            for (String feature : averagedGlobalWeightVectors.keySet()) {
                this.currentGlobalWeightVectors.put(feature, averagedGlobalWeightVectors.get(feature) / ((double) averagingUpdates));
            }
        }

        // Compute the final training error:
        int finalTrainingErrors = 0;
        Double finalTrainingCost = 0.0;
        for (Instance instance : instances) {
            Prediction prediction = this.predict(instance);
            if (Double.compare(instance.getCosts().get(prediction.getLabel()), 0.0) > 0) {
                finalTrainingErrors += 1;
                finalTrainingCost += instance.getCosts().get(prediction.getLabel());
            }
        }

        Double finalTrainingErrorRate = (double) finalTrainingErrors / (double) instances.size();
        //System.out.println("Final training error rate=" + finalTrainingErrorRate.toString());
        //System.out.println("Final training cost=" + finalTrainingCost.toString());

        return finalTrainingCost;
    }

    public void averageOverClassifiers(ArrayList<JAROW> classifiers) {
        if (!classifiers.isEmpty()) {
            for (String label : classifiers.get(0).getCurrentWeightVectors().keySet()) {
                this.currentWeightVectors.put(label, new TObjectDoubleHashMap<String>());
            }
            this.currentGlobalWeightVectors = new TObjectDoubleHashMap<>();

            HashMap<String, HashMap<String, Integer>> updates = new HashMap<>();
            HashMap<String, Integer> globalUpdates = new HashMap<>();
            for (JAROW classifier : classifiers) {
                this.param = classifier.getParam();
                for (String label : classifier.getCurrentWeightVectors().keySet()) {
                    if (!updates.containsKey(label)) {
                        updates.put(label, new HashMap<String, Integer>());
                    }
                    for (String feature : classifier.getCurrentWeightVectors().get(label).keySet()) {
                        this.currentWeightVectors.get(label).adjustOrPutValue(feature, classifier.getCurrentWeightVectors().get(label).get(feature), classifier.getCurrentWeightVectors().get(label).get(feature));
                        if (updates.get(label).containsKey(feature)) {
                            updates.get(label).put(feature, updates.get(label).get(feature) + 1);
                        } else {                        
                            updates.get(label).put(feature, 1);
                        }
                    }
                }
                for (String feature : classifier.getCurrentGlobalWeightVectors().keySet()) {
                    this.currentGlobalWeightVectors.adjustOrPutValue(feature, classifier.getCurrentGlobalWeightVectors().get(feature), classifier.getCurrentGlobalWeightVectors().get(feature));
                    if (globalUpdates.containsKey(feature)) {
                        globalUpdates.put(feature, globalUpdates.get(feature) + 1);
                    } else {                        
                        globalUpdates.put(feature, 1);
                    }
                }
            } 

            for (String label : this.currentWeightVectors.keySet()) {
                for (String feature : this.currentWeightVectors.get(label).keySet()) {
                    double currentWeightUpdates = (double) updates.get(label).get(feature);
                    this.currentWeightVectors.get(label).put(feature, this.currentWeightVectors.get(label).get(feature) * (1.0 / currentWeightUpdates));
                }
            }
            for (String feature : this.currentGlobalWeightVectors.keySet()) {
                double currentGlobalWeightUpdates = (double) globalUpdates.get(feature);
                this.currentGlobalWeightVectors.put(feature, this.currentGlobalWeightVectors.get(feature) * (1.0 / currentGlobalWeightUpdates));
            }
        }
    }

    public void probGeneration() {
        probGeneration(1.0, 100);
    }

    public void probGeneration(Double scale) {
        probGeneration(scale, 100);
    }

    public void probGeneration(int noWeightVectors) {
        probGeneration(1.0, noWeightVectors);
    }

    public void probGeneration(Double scale, int noWeightVectors) {
        // initialize the weight vectors
        System.out.println("Generating samples for the weight vectors to obtain probability estimates");
        this.probWeightVectors = new ArrayList<>();
        for (int i = 0; i < noWeightVectors; i++) {
            this.probWeightVectors.add(new THashMap<String, TObjectDoubleHashMap<String>>());
            for (String label : this.currentWeightVectors.keySet()) {
                this.probWeightVectors.get(i).put(label, new TObjectDoubleHashMap<String>());
            }
        }

        RandomGaussian gaussian = new RandomGaussian();
        for (String label : this.currentWeightVectors.keySet()) {
            // We are ignoring features that never got their weight set 
            for (String feature : this.currentWeightVectors.keySet()) {
                // note that if the weight was updated, then the variance must have been updated too, i.e. we shouldn't have 0s
                ArrayList<Double> weights = new ArrayList<>();

                for (int i = 0; i < noWeightVectors; i++) {
                    weights.add(gaussian.getGaussian(this.currentWeightVectors.get(label).get(feature), scale * this.currentVarianceVectors.get(label).get(feature)));
                }
                // we got the samples, now let's put them in the right places
                for (int i = 0; i < weights.size(); i++) {
                    this.probWeightVectors.get(i).get(label).put(feature, weights.get(i));
                }
            }
        }
        for (String label : this.currentGlobalWeightVectors.keySet()) {
            // We are ignoring features that never got their weight set 
            for (String feature : this.currentGlobalWeightVectors.keySet()) {
                // note that if the weight was updated, then the variance must have been updated too, i.e. we shouldn't have 0s
                ArrayList<Double> weights = new ArrayList<>();

                for (int i = 0; i < noWeightVectors; i++) {
                    weights.add(gaussian.getGaussian(this.currentGlobalWeightVectors.get(feature), scale * this.currentGlobalVarianceVectors.get(feature)));
                }
                // we got the samples, now let's put them in the right places
                for (int i = 0; i < weights.size(); i++) {
                    this.probWeightVectors.get(i).get(label).put(feature, weights.get(i));
                }
            }
        }

        System.out.println("done");
        this.probabilities = true;
    }

    // train by optimizing the c parametr
    public static JAROW trainOpt(ArrayList<Instance> instances) {
        Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
        return trainOpt(instances, 10, params, 0.1, true, false);
    }

    public static JAROW trainOpt(ArrayList<Instance> instances, int rounds, Double[] paramValues, Double heldout, boolean adapt, boolean optimizeProbs) {
        System.out.println("===Training with " + instances.size() + " instances===");

        // this value will be kept if nothing seems to work better
        double bestParam = 1;
        Double lowestCost = Double.POSITIVE_INFINITY;
        JAROW bestClassifier = null;
        ArrayList<Instance> trainingInstances = new ArrayList<>(instances.subList(0, (int) Math.round(instances.size() * (1 - heldout))));
        int to = ((int) Math.round(instances.size() * (1 - heldout))) + 1;
        if (to >= instances.size()) {
            to = instances.size() - 1;
        }
        ArrayList<Instance> testingInstances = new ArrayList<>(instances.subList(to, instances.size()));
        int changes = 0;
        for (Double param : paramValues) {
            System.out.println("Training with param=" + param + " on " + trainingInstances.size() + " instances");
            // Keep the weight vectors produced in each round
            JAROW classifier = new JAROW();
            classifier.train(trainingInstances, true, false, rounds, param, adapt);
            System.out.println("testing on " + testingInstances.size() + " instances");
            // Test on the dev for the weight vector produced in each round
            Double devCost = classifier.batchPredict(testingInstances);
            System.out.println("Dev cost:" + devCost + " avg cost per instance " + devCost / (double) testingInstances.size());

            if (devCost < lowestCost) {
                bestParam = param;
                lowestCost = devCost;
                bestClassifier = classifier;
                changes++;
            }
        }
        if (changes == 1) {
            bestParam = 100;
        }

        // OK, now we got the best C, so it's time to train the final model with it
        // Do the probs
        // So we need to pick a value between 
        Double bestScale = 1.0;
        if (optimizeProbs && bestClassifier != null) {
            System.out.println("optimizing the scale parameter for probability estimation");
            Double lowestEntropy = Double.POSITIVE_INFINITY;
            int steps = 20;
            for (int i = 0; i < steps; i++) {
                Double scale = 1.0 - (double) i / (double) steps;
                System.out.println("scale= " + scale);
                bestClassifier.probGeneration(scale);
                Double entropy = bestClassifier.batchPredict(testingInstances, true);
                System.out.println("entropy sums: " + entropy);

                if (Double.compare(entropy, lowestEntropy) < 0) {
                    bestScale = scale;
                    lowestEntropy = entropy;
                }
            }
        }

        // Now train the final model:
        System.out.println("Training with best param=" + bestParam + " on all the data");

        JAROW finalClassifier = new JAROW();
        finalClassifier.train(instances, true, false, rounds, bestParam, adapt);
        if (optimizeProbs) {
            System.out.println("Adding weight samples for probability estimates with scale " + bestScale);
            finalClassifier.probGeneration(bestScale);
        }

        return finalClassifier;
    }

    // save function for the parameters:
    public static void save(JAROW classifier, String filename) throws IOException {
        FileOutputStream fout = new FileOutputStream(filename);
        ObjectOutputStream oos = new ObjectOutputStream(fout);

        oos.writeObject(classifier);
        oos.close();
    }

    // load function for the parameters:
    public static JAROW load(String filename) throws IOException, ClassNotFoundException {
        FileInputStream fin = new FileInputStream(filename);
        ObjectInputStream ois = new ObjectInputStream(fin);

        JAROW classifier = (JAROW) ois.readObject();
        ois.close();

        return classifier;
    }

    public Double dotProduct(TObjectDoubleHashMap<String> a1, TObjectDoubleHashMap<String> a2) {
        //BigDecimal product = BigDecimal.ZERO;
        Double product = 0.0;
        for (String label : a1.keySet()) {
            if (a2.contains(label)) {
                Double v1 = a1.get(label);
                if (v1 == Double.POSITIVE_INFINITY) {
                    v1 = Double.MAX_VALUE;
                    System.exit(0);
                } else if (v1 == Double.NEGATIVE_INFINITY) {
                    v1 = -Double.MAX_VALUE;
                    System.exit(0);
                }
                Double v2 = a2.get(label);
                if (v2 == Double.POSITIVE_INFINITY) {
                    v2 = Double.MAX_VALUE;
                    System.exit(0);
                } else if (v2 == Double.NEGATIVE_INFINITY) {
                    v2 = -Double.MAX_VALUE;
                    System.exit(0);
                }
                try {
                    product += v1 * v2;
                } catch (java.lang.NumberFormatException io) {
                    System.out.println("!!! " + a1.get(label) + "," + a2.get(label));
                    System.out.println("!!! " + v1 + "," + v2);
                    System.out.println("!!! " + a1.containsKey(label) + "," + a2.containsKey(label));
                    System.out.println("!!! " + a1.contains(label) + "," + a2.contains(label));
                    System.out.println("!!! " + a1.getNoEntryValue() + "," + a2.getNoEntryValue());
                    System.out.println("!!! " + label + "," + label);
                    System.exit(0);
                }
            }
        }
        return product.doubleValue();
    }

    public boolean isProbabilities() {
        return probabilities;
    }

    public THashMap<String, TObjectDoubleHashMap<String>> getCurrentWeightVectors() {
        return currentWeightVectors;
    }

    public TObjectDoubleHashMap<String> getCurrentGlobalWeightVectors() {
        return currentGlobalWeightVectors;
    }

    public THashMap<String, TObjectDoubleHashMap<String>> getCurrentVarianceVectors() {
        return currentVarianceVectors;
    }

    public ArrayList<THashMap<String, TObjectDoubleHashMap<String>>> getProbWeightVectors() {
        return probWeightVectors;
    }

    public double getParam() {
        return param;
    }
}
