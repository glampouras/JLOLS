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
import java.util.Random;
import java.util.logging.Logger;


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

    /**
     *
     */
    public JAROW() {
        this.probabilities = false;
        this.currentWeightVectors = new THashMap<>();
        this.currentVarianceVectors = new THashMap<>();

        //GLOBAL FEATURES ARE FEATURES WHOSE WEIGHTS ARE CALCULATED REGARDLESS THE LABEL
        //FOR FEATURES TO BE TREATED AS GLOBAL THEIR IDENTIFIERS MUST START WITH "global_"
        this.currentGlobalWeightVectors = new TObjectDoubleHashMap<>();
        this.currentGlobalVarianceVectors = new TObjectDoubleHashMap<>();
        
        this.probWeightVectors = new ArrayList<>();
        this.averagedWeightVectors = new THashMap<>();

        this.probGlobalWeightVectors = new ArrayList<>();
        this.averagedGlobalWeightVectors = new TObjectDoubleHashMap<>();        
    }

    /**
     *
     * @param j
     */
    public JAROW(JAROW j) {
        this.probabilities = j.probabilities;
        this.currentWeightVectors = new THashMap<>();
        j.currentWeightVectors.keySet().forEach((key) -> {
            this.currentWeightVectors.put(key, new TObjectDoubleHashMap<String>(j.getCurrentWeightVectors().get(key)));
        });
        this.currentVarianceVectors = new THashMap<>();
        j.currentVarianceVectors.keySet().forEach((key) -> {
            this.currentVarianceVectors.put(key, new TObjectDoubleHashMap<String>(j.getCurrentVarianceVectors().get(key)));
        });
        this.currentGlobalWeightVectors = new TObjectDoubleHashMap<String>(j.getCurrentGlobalWeightVectors());
        this.currentGlobalVarianceVectors = new TObjectDoubleHashMap<String>(j.getCurrentGlobalVarianceVectors());

        if (j.probWeightVectors != null) {
            this.probWeightVectors = new ArrayList<>();
            j.probWeightVectors.stream().map((map) -> {
                THashMap<String, TObjectDoubleHashMap<String>> nMap = new THashMap<String, TObjectDoubleHashMap<String>>();
                map.keySet().forEach((key) -> {
                    nMap.put(key, new TObjectDoubleHashMap<String>(map.get(key)));
                });
                return nMap;
            }).forEachOrdered((nMap) -> {
                this.probWeightVectors.add(nMap);
            });
        }
        if (j.averagedWeightVectors != null) {
            this.averagedWeightVectors = new THashMap<>();
            j.averagedWeightVectors.keySet().forEach((key) -> {
                this.averagedWeightVectors.put(key, new TObjectDoubleHashMap<String>(j.getAveragedWeightVectors().get(key)));
            });
        }
        if (j.probGlobalWeightVectors != null) {
            this.probGlobalWeightVectors = new ArrayList<>();
            j.probGlobalWeightVectors.stream().map((map) -> new TObjectDoubleHashMap<String>(map)).forEachOrdered((nMap) -> {
                this.probGlobalWeightVectors.add(nMap);
            });

        }
        if (j.averagedGlobalWeightVectors != null) {
            this.averagedGlobalWeightVectors = new TObjectDoubleHashMap<>(j.getAveragedGlobalWeightVectors());
        }
        this.averagingUpdates = j.averagingUpdates;
        this.param = j.param;
    }

    /**
     *
     * @param label
     * @return
     */
    public TObjectDoubleHashMap<String> getWeightVector(String label) {
        TObjectDoubleHashMap<String> weightVector = new TObjectDoubleHashMap<String>();
        if (getCurrentWeightVectors().containsKey(label)) {
            weightVector.putAll(getCurrentWeightVectors().get(label));
        }
        weightVector.putAll(getCurrentGlobalWeightVectors());

        return weightVector;
    }

    // This predicts always using the current weight vectors

    /**
     *
     * @param instance
     * @return
     */
    public Prediction predict(Instance instance) {
        return predict(instance, false, false);
    }
    
    /**
     *
     * @param instance
     * @param verbose
     * @param probabilities
     * @return
     */
    public Prediction predict(Instance instance, boolean verbose/* = false*/, boolean probabilities/* = false*/) {
        //# always add the bias
        instance.getGeneralFeatureVector().put("biasAutoAdded", 1.0);

        Prediction prediction = new Prediction();

        getCurrentWeightVectors().keySet().forEach((label) -> {
            Double score = dotProduct(instance.getFeatureVector(label), this.getWeightVector(label));
            prediction.getLabel2Score().put(label, score);
            if (Double.compare(score, prediction.getScore()) > 0) {
                prediction.setScore(score);
                prediction.setLabel(label);
            }
        });

        if (verbose) {
            TObjectDoubleHashMap<String> featureVector = instance.getFeatureVector(prediction.getLabel());
            featureVector.keySet().stream().map((feature) -> {
                // keep the feature weights for the predicted label
                ArrayList<Object> fvw = new ArrayList<>();
                fvw.add(feature);
                fvw.add(featureVector.get(feature));
                fvw.add(getCurrentWeightVectors().get(prediction.getLabel()).get(feature));
                fvw.add(getCurrentGlobalWeightVectors().get(feature));
                return fvw;
            }).forEachOrdered((fvw) -> {
                prediction.getFeatureValueWeights().add(fvw);
            });
        }
        if (probabilities) {
            // if we have probabilistic training
            if (this.isProbabilities()) {
                TObjectDoubleHashMap<String> probPredictions = new TObjectDoubleHashMap<>();
                this.getProbWeightVectors().get(0).keySet().forEach((label) -> {
                    // smoothing the probabilities with add 0.01 of 1 out of the vectors
                    probPredictions.put(label, 0.01 / this.getProbWeightVectors().size());
                });
                // for each of the weight vectors obtained get its prediction
                this.getProbWeightVectors().stream().map((probWeightVector) -> {
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
                    return maxLabel;
                }).forEachOrdered((maxLabel) -> {
                    // so the winning label adds one vote
                    probPredictions.put(maxLabel, probPredictions.get(maxLabel) + 1);
                });
                // now let's normalize:
                probPredictions.keySet().forEach((label) -> {
                    Double score = probPredictions.get(label);
                    prediction.getLabel2prob().put(label, score / this.getProbWeightVectors().size());
                });
                // Also compute the entropy:
                for (Double prob : prediction.getLabel2prob().values()) {
                    if (Double.compare(prob, 0.0) > 0) {
                        prediction.setEntropy(prediction.getEntropy() - prob * (Math.log(prob) / Math.log(2)));
                    }
                }
                // normalize it:
                prediction.setEntropy(prediction.getEntropy() / (Math.log(prediction.getLabel2prob().size()) / Math.log(2)));
            } else {
            }
        }

        return prediction;
    }

    // This is just used to optimize the params
    // if probabilities is True we return the ratio for the average entropies, otherwise the loss

    /**
     *
     * @param instances
     * @return
     */
    public double batchPredict(ArrayList<Instance> instances) {
        return batchPredict(instances, false);
    }

    /**
     *
     * @param instances
     * @param probabilities
     * @return
     */
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
            return sumLogProbCorrect;
        } else {
            return totalCost;
        }
    }

    // the parameter here is for AROW learning
    // adapt if True is AROW, if False it is passive aggressive-II with prediction-based updates 

    /**
     *
     * @param instances
     * @param averaging
     * @param shuffling
     * @param rounds
     * @param param
     * @return
     */
    public Double train(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, Double param) {
        return train(instances, averaging, shuffling, rounds, param, true);
    }

    /**
     *
     * @param instances
     * @param averaging
     * @param shuffling
     * @param rounds
     * @param param
     * @param adapt
     * @return
     */
    public Double train(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, Double param, boolean adapt) {
        // we first need to go through the dataset to find how many classes

        // Initialize the weight vectors in the beginning of training"
        // we have one variance and one weight vector per class        
        
        this.setParam(param);
        this.setCurrentWeightVectors(new THashMap<String, TObjectDoubleHashMap<String>>());
        if (adapt) {
            this.setCurrentVarianceVectors(new THashMap<String, TObjectDoubleHashMap<String>>());
        }
        this.setCurrentGlobalWeightVectors(new TObjectDoubleHashMap<String>());
        this.setCurrentGlobalVarianceVectors(new TObjectDoubleHashMap<String>());
        this.setAveragedGlobalWeightVectors(new TObjectDoubleHashMap<String>());
        if (averaging) {
            this.setAveragedWeightVectors(new THashMap<String, TObjectDoubleHashMap<String>>());
            this.setAveragingUpdates(0);
        }
        if (!instances.isEmpty()) {
            HashSet<String> labels = new HashSet<>();
            instances.forEach((in) -> {
                in.getCosts().keySet().forEach((label) -> {
                    labels.add(label);
                });
            });
            labels.stream().map((label) -> {
                this.getCurrentWeightVectors().put(label, new TObjectDoubleHashMap<String>());
                return label;
            }).map((label) -> {
                // remember: this is sparse in the sense that everything that doesn't have a value, is 1
                // everytime we to do something with it, remember to add 1
                if (adapt) {
                    this.getCurrentVarianceVectors().put(label, new TObjectDoubleHashMap<String>());
                }
                // keep the averaged weight vector
                return label;
            }).filter((label) -> (averaging && this.getAveragedWeightVectors() != null)).forEachOrdered((label) -> {
                this.getAveragedWeightVectors().put(label, new TObjectDoubleHashMap<String>());
            });
        }
        // in each iteration        
        for (int r = 0; r < rounds; r++) {
            // shuffle
            if (shuffling) {
                Collections.shuffle(instances, new Random(13));
            }
            // for each instance
            instances.stream().map((instance) -> {
                Prediction prediction = this.predict(instance);
                // so if the prediction was incorrect
                // we are no longer large margin, since we are using the loss from the cost-sensitive PA
                if (Double.compare(instance.getCosts().get(prediction.getLabel()), 0) > 0) {
                    // first we need to get the score for the correct answer
                    // if the instance has more than one correct answer then pick the min
                    Double minCorrectLabelScore = Double.POSITIVE_INFINITY;
                    String minCorrectLabel = null;
                    for (String label : instance.getCorrectLabels()) {
                        Double score = this.dotProduct(instance.getFeatureVector(label), this.getWeightVector(label));
                        if (Double.compare(score, minCorrectLabelScore) < 0) {
                            minCorrectLabelScore = score;
                            minCorrectLabel = label;
                        }
                    }
                    if (minCorrectLabelScore == Double.POSITIVE_INFINITY) {
                        System.out.println("No correct labels error!");
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
                        featureVectorPredicted.keySet().forEach((feature) -> {
                            // the variance is either some value that is in the dict or just 1
                            if (!feature.startsWith("global_")) {
                                if (this.getCurrentVarianceVectors().get(prediction.getLabel()).containsKey(feature)) {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * this.getCurrentVarianceVectors().get(prediction.getLabel()).get(feature));
                                } else {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                                }
                            } else if (this.getCurrentGlobalVarianceVectors().containsKey(feature)) {
                                zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * this.getCurrentGlobalVarianceVectors().get(feature));
                            } else {
                                zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                            }
                        });
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            // then for the minCorrect:
                            if (!feature.startsWith("global_")) {
                                if (this.getCurrentVarianceVectors().get(minCorrectLabel).containsKey(feature)) {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * this.getCurrentVarianceVectors().get(minCorrectLabel).get(feature));
                                } else {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                                }
                            } else if (this.getCurrentGlobalVarianceVectors().containsKey(feature)) {
                                zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * this.getCurrentGlobalVarianceVectors().get(feature));
                            } else {
                                zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                            }
                        }
                        Double confidence = this.dotProduct(zVectorPredicted, featureVectorPredicted) + this.dotProduct(zVectorMinCorrect, featureVectorMinCorrect);
                        Double beta = 1.0 / (confidence + param);
                        Double alpha = loss * beta;
                        // update the current weight vectors
                        zVectorPredicted.keySet().forEach((feature) -> {
                            if (!feature.startsWith("global_")) {
                                this.getCurrentWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -(alpha * zVectorPredicted.get(feature)), -(alpha * zVectorPredicted.get(feature)));
                            } else {
                                this.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, -(alpha * zVectorPredicted.get(feature)), -(alpha * zVectorPredicted.get(feature)));
                            }
                        });
                        for (String feature : zVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.getCurrentWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, (alpha * zVectorMinCorrect.get(feature)), (alpha * zVectorMinCorrect.get(feature)));
                            } else {
                                this.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, (alpha * zVectorMinCorrect.get(feature)), (alpha * zVectorMinCorrect.get(feature)));
                            }
                        }
                        if (averaging && this.getAveragedWeightVectors() != null) {
                            zVectorPredicted.keySet().forEach((feature) -> {
                                if (!feature.startsWith("global_")) {
                                    this.getAveragedWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                } else {
                                    this.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                }
                            });
                            for (String feature : zVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    this.getAveragedWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                } else {
                                    this.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        // update the diagonal covariance
                        featureVectorPredicted.keySet().forEach((feature) -> {
                            // for the predicted
                            if (!feature.startsWith("global_")) {
                                this.getCurrentVarianceVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            } else {
                                this.getCurrentGlobalVarianceVectors().adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            }
                        });
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            // for the minCorrect
                            if (!feature.startsWith("global_")) {
                                this.getCurrentVarianceVectors().get(minCorrectLabel).adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            } else {
                                this.getCurrentGlobalVarianceVectors().adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            }
                        }
                    } else {
                        // the squared norm is twice the square of the features since they are the same per class 
                        //The value specific features can be safely ignored since they are not shared between values and would result in 0s anyway
                        Double norm = 2.0 * this.dotProduct(instance.getGeneralFeatureVector(), instance.getGeneralFeatureVector());
                        Double factor = loss / (norm + 1.0 / (2.0 * param));
                        TObjectDoubleHashMap<String> featureVectorPredicted = instance.getFeatureVector(prediction.getLabel());
                        featureVectorPredicted.keySet().forEach((feature) -> {
                            if (!feature.startsWith("global_")) {
                                this.getCurrentWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            } else {
                                this.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            }
                        });
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.getCurrentWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            } else {
                                this.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            }
                        }
                        if (averaging && this.getAveragedWeightVectors() != null) {
                            featureVectorPredicted.keySet().forEach((feature) -> {
                                if (!feature.startsWith("global_")) {
                                    this.getAveragedWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                } else {
                                    this.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                }
                            });
                            for (String feature : featureVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    this.getAveragedWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                } else {
                                    this.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                    }
                }
                return instance;
            }).filter((_item) -> (averaging)).forEachOrdered((_item) -> {
                this.setAveragingUpdates(this.getAveragingUpdates() + 1);
            });
        }

        if (averaging && this.getAveragedWeightVectors() != null) {
            this.getCurrentWeightVectors().keySet().forEach((label) -> {
                this.getAveragedWeightVectors().get(label).keySet().forEach((feature) -> {
                    this.getCurrentWeightVectors().get(label).put(feature, this.getAveragedWeightVectors().get(label).get(feature) / this.getAveragingUpdates());
                });
            });
            this.getAveragedGlobalWeightVectors().keySet().forEach((feature) -> {
                this.getCurrentGlobalWeightVectors().put(feature, this.getAveragedGlobalWeightVectors().get(feature) / this.getAveragingUpdates());
            });
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

        Double finalTrainingErrorRate = finalTrainingErrors / (double) instances.size();
        //System.out.println("Final training error rate=" + finalTrainingErrorRate.toString());
        //System.out.println("Final training cost=" + finalTrainingCost.toString());

        return finalTrainingCost;
    }

    /**
     *
     * @param instance
     * @return
     */
    public boolean isInstanceLeadingToFix(Instance instance) {
        Prediction prediction = this.predict(instance);
        return Double.compare(instance.getCosts().get(prediction.getLabel()), 0) > 0;
    }

    /**
     *
     * @param instances
     * @param averaging
     * @param shuffling
     * @param rounds
     * @param param
     * @return
     */
    public Double trainAdditional(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, double param) {
        return trainAdditional(instances, averaging, shuffling, rounds, true, param);
    }

    /**
     *
     * @param instances
     * @param averaging
     * @param shuffling
     * @param rounds
     * @param adapt
     * @param param
     * @return
     */
    public Double trainAdditional(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, boolean adapt, double param) {
        // We do not initialize the weight vectors in the beginning of training, rather keep them as they have already been trained

        // in each iteration        
        for (int r = 0; r < rounds; r++) {
            // shuffle
            if (shuffling) {
                Collections.shuffle(instances, new Random(13));
            }
            // for each instance
            instances.stream().map((instance) -> {
                Prediction prediction = this.predict(instance);
                // so if the prediction was incorrect
                // we are no longer large margin, since we are using the loss from the cost-sensitive PA
                if (Double.compare(instance.getCosts().get(prediction.getLabel()), 0) > 0) {
                    // first we need to get the score for the correct answer
                    // if the instance has more than one correct answer then pick the min
                    Double minCorrectLabelScore = Double.POSITIVE_INFINITY;
                    String minCorrectLabel = null;
                    for (String label : instance.getCorrectLabels()) {
                        Double score = this.dotProduct(instance.getFeatureVector(label), this.getWeightVector(label));
                        if (Double.compare(score, minCorrectLabelScore) < 0) {
                            minCorrectLabelScore = score;
                            minCorrectLabel = label;
                        }
                    }
                    if (minCorrectLabelScore == Double.POSITIVE_INFINITY) {
                        System.out.println("No correct labels error!");
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
                                if (!this.getCurrentVarianceVectors().containsKey(minCorrectLabel)) {
                                    System.out.println(this.getCurrentVarianceVectors().keySet());
                                }
                                if (this.getCurrentVarianceVectors().get(prediction.getLabel()).containsKey(feature)) {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * this.getCurrentVarianceVectors().get(prediction.getLabel()).get(feature));
                                } else {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                                }
                            } else {
                                if (this.getCurrentGlobalVarianceVectors().containsKey(feature)) {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * this.getCurrentGlobalVarianceVectors().get(feature));
                                } else {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                                }
                            }
                        }
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            // then for the minCorrect:
                            if (!feature.startsWith("global_")) {
                                if (this.getCurrentVarianceVectors().get(minCorrectLabel).containsKey(feature)) {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * this.getCurrentVarianceVectors().get(minCorrectLabel).get(feature));
                                } else {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                                }
                            } else {
                                if (this.getCurrentGlobalVarianceVectors().containsKey(feature)) {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * this.getCurrentGlobalVarianceVectors().get(feature));
                                } else {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        Double confidence = this.dotProduct(zVectorPredicted, featureVectorPredicted) + this.dotProduct(zVectorMinCorrect, featureVectorMinCorrect);
                        Double beta = 1.0 / (confidence + param);
                        Double alpha = loss * beta;
                        // update the current weight vectors
                        zVectorPredicted.keySet().forEach((feature) -> {
                            if (!feature.startsWith("global_")) {
                                this.getCurrentWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -alpha * zVectorPredicted.get(feature), -alpha * zVectorPredicted.get(feature));
                            } else {
                                this.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, -alpha * zVectorPredicted.get(feature), -alpha * zVectorPredicted.get(feature));
                            }
                        });
                        for (String feature : zVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.getCurrentWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                            } else {
                                this.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                            }
                        }
                        if (averaging && this.getAveragedWeightVectors() != null) {
                            zVectorPredicted.keySet().forEach((feature) -> {
                                if (!feature.startsWith("global_")) {
                                    this.getAveragedWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                } else {
                                    this.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                }
                            });
                            for (String feature : zVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    this.getAveragedWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                } else {
                                    this.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        // update the diagonal covariance
                        featureVectorPredicted.keySet().forEach((feature) -> {
                            // for the predicted
                            if (!feature.startsWith("global_")) {
                                this.getCurrentVarianceVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            } else {
                                this.getCurrentGlobalVarianceVectors().adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            }
                        });
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            // for the minCorrect
                            if (!feature.startsWith("global_")) {
                                this.getCurrentVarianceVectors().get(minCorrectLabel).adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            } else {
                                this.getCurrentGlobalVarianceVectors().adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            }
                        }
                        this.getCurrentVarianceVectors().get(prediction.getLabel()).keySet().stream().filter((s) -> (this.getCurrentVarianceVectors().get(prediction.getLabel()).get(s) < 0)).forEachOrdered((_item) -> {
                            System.out.println(this.getCurrentVarianceVectors().get(prediction.getLabel()));
                        });
                        for (String s : this.getCurrentVarianceVectors().get(minCorrectLabel).keySet()) {
                            if (this.getCurrentVarianceVectors().get(minCorrectLabel).get(s) < 0) {
                            }
                        }
                        this.getCurrentGlobalVarianceVectors().keySet().stream().filter((s) -> (this.getCurrentGlobalVarianceVectors().get(s) < 0)).forEachOrdered((_item) -> {
                            System.out.println(this.getCurrentGlobalVarianceVectors());
                        });
                    } else {
                        // the squared norm is twice the square of the features since they are the same per class 
                        Double norm = 2.0 * this.dotProduct(instance.getGeneralFeatureVector(), instance.getGeneralFeatureVector());
                        Double factor = loss / (norm + 1.0 / (2.0 * param));
                        TObjectDoubleHashMap<String> featureVectorPredicted = instance.getFeatureVector(prediction.getLabel());
                        featureVectorPredicted.keySet().forEach((feature) -> {
                            if (!feature.startsWith("global_")) {
                                this.getCurrentWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            } else {
                                this.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            }
                        });
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                this.getCurrentWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            } else {
                                this.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            }
                        }
                        if (averaging && this.getAveragedWeightVectors() != null) {
                            featureVectorPredicted.keySet().forEach((feature) -> {
                                if (!feature.startsWith("global_")) {
                                    this.getAveragedWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                } else {
                                    this.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                }
                            });
                            for (String feature : featureVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    this.getAveragedWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                } else {
                                    this.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                    }
                }
                return instance;
            }).filter((_item) -> (averaging)).forEachOrdered((_item) -> {
                this.setAveragingUpdates(this.getAveragingUpdates() + 1);
            }); //System.out.println("Training error rate in round " + r + " : " + (double) errorsInRound / (double) instances.size());
        }

        if (averaging && this.getAveragedWeightVectors() != null) {
            this.getCurrentWeightVectors().keySet().forEach((label) -> {
                this.getAveragedWeightVectors().get(label).keySet().forEach((feature) -> {
                    this.getCurrentWeightVectors().get(label).put(feature, this.getAveragedWeightVectors().get(label).get(feature) / this.getAveragingUpdates());
                });
            });
            this.getAveragedGlobalWeightVectors().keySet().forEach((feature) -> {
                this.getCurrentGlobalWeightVectors().put(feature, this.getAveragedGlobalWeightVectors().get(feature) / this.getAveragingUpdates());
            });
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

        Double finalTrainingErrorRate = finalTrainingErrors / (double) instances.size();
        //System.out.println("Final training error rate=" + finalTrainingErrorRate.toString());
        //System.out.println("Final training cost=" + finalTrainingCost.toString());

        return finalTrainingCost;
    }

    /**
     *
     * @param classifiers
     */
    public void averageOverClassifiers(ArrayList<JAROW> classifiers) {
        if (!classifiers.isEmpty()) {
            classifiers.get(0).getCurrentWeightVectors().keySet().forEach((label) -> {
                this.getCurrentWeightVectors().put(label, new TObjectDoubleHashMap<String>());
            });
            this.setCurrentGlobalWeightVectors(new TObjectDoubleHashMap<String>());

            HashMap<String, HashMap<String, Integer>> updates = new HashMap<>();
            HashMap<String, Integer> globalUpdates = new HashMap<>();
            classifiers.stream().map((classifier) -> {
                this.setParam(classifier.getParam());
                return classifier;
            }).map((classifier) -> {
                classifier.getCurrentWeightVectors().keySet().stream().map((label) -> {
                    if (!updates.containsKey(label)) {
                        updates.put(label, new HashMap<String, Integer>());
                    }
                    return label;
                }).forEachOrdered((label) -> {
                    classifier.getCurrentWeightVectors().get(label).keySet().stream().map((feature) -> {
                        this.getCurrentWeightVectors().get(label).adjustOrPutValue(feature, classifier.getCurrentWeightVectors().get(label).get(feature), classifier.getCurrentWeightVectors().get(label).get(feature));
                        return feature;
                    }).forEachOrdered((feature) -> {
                        if (updates.get(label).containsKey(feature)) {
                            updates.get(label).put(feature, updates.get(label).get(feature) + 1);
                        } else {                        
                            updates.get(label).put(feature, 1);
                        }
                    });
                });
                return classifier;
            }).forEachOrdered((classifier) -> {
                classifier.getCurrentGlobalWeightVectors().keySet().stream().map((feature) -> {
                    this.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, classifier.getCurrentGlobalWeightVectors().get(feature), classifier.getCurrentGlobalWeightVectors().get(feature));
                    return feature;
                }).forEachOrdered((feature) -> {
                    if (globalUpdates.containsKey(feature)) {
                        globalUpdates.put(feature, globalUpdates.get(feature) + 1);
                    } else {                        
                        globalUpdates.put(feature, 1);
                    }
                });
            });
            this.getCurrentWeightVectors().keySet().forEach((label) -> {
                this.getCurrentWeightVectors().get(label).keySet().forEach((feature) -> {
                    double currentWeightUpdates = updates.get(label).get(feature);
                    this.getCurrentWeightVectors().get(label).put(feature, this.getCurrentWeightVectors().get(label).get(feature) * (1.0 / currentWeightUpdates));
                });
            });
            this.getCurrentGlobalWeightVectors().keySet().forEach((feature) -> {
                double currentGlobalWeightUpdates = globalUpdates.get(feature);
                this.getCurrentGlobalWeightVectors().put(feature, this.getCurrentGlobalWeightVectors().get(feature) * (1.0 / currentGlobalWeightUpdates));
            });
        }
    }

    /**
     *
     */
    public void probGeneration() {
        probGeneration(1.0, 100);
    }

    /**
     *
     * @param scale
     */
    public void probGeneration(Double scale) {
        probGeneration(scale, 100);
    }

    /**
     *
     * @param noWeightVectors
     */
    public void probGeneration(int noWeightVectors) {
        probGeneration(1.0, noWeightVectors);
    }

    /**
     *
     * @param scale
     * @param noWeightVectors
     */
    public void probGeneration(Double scale, int noWeightVectors) {
        // initialize the weight vectors
        System.out.println("Generating samples for the weight vectors to obtain probability estimates");
// initialize the weight vectors
                this.setProbWeightVectors(new ArrayList<THashMap<String, TObjectDoubleHashMap<String>>>());
        for (int i = 0; i < noWeightVectors; i++) {
            this.getProbWeightVectors().add(new THashMap<String, TObjectDoubleHashMap<String>>());
            for (String label : this.getCurrentWeightVectors().keySet()) {
                this.getProbWeightVectors().get(i).put(label, new TObjectDoubleHashMap<String>());
            }
        }

        RandomGaussian gaussian = new RandomGaussian();
        this.getCurrentWeightVectors().keySet().forEach((label) -> {
            // We are ignoring features that never got their weight set
            this.getCurrentWeightVectors().keySet().forEach((feature) -> {
                // note that if the weight was updated, then the variance must have been updated too, i.e. we shouldn't have 0s
                ArrayList<Double> weights = new ArrayList<>();

                for (int i = 0; i < noWeightVectors; i++) {
                    weights.add(gaussian.getGaussian(this.getCurrentWeightVectors().get(label).get(feature), scale * this.getCurrentVarianceVectors().get(label).get(feature)));
                }
                // we got the samples, now let's put them in the right places
                for (int i = 0; i < weights.size(); i++) {
                    this.getProbWeightVectors().get(i).get(label).put(feature, weights.get(i));
                }
            });
        });
        this.getCurrentGlobalWeightVectors().keySet().forEach((label) -> {
            // We are ignoring features that never got their weight set
            this.getCurrentGlobalWeightVectors().keySet().forEach((feature) -> {
                // note that if the weight was updated, then the variance must have been updated too, i.e. we shouldn't have 0s
                ArrayList<Double> weights = new ArrayList<>();

                for (int i = 0; i < noWeightVectors; i++) {
                    weights.add(gaussian.getGaussian(this.getCurrentGlobalWeightVectors().get(feature), scale * this.getCurrentGlobalVarianceVectors().get(feature)));
                }
                // we got the samples, now let's put them in the right places
                for (int i = 0; i < weights.size(); i++) {
                    this.getProbWeightVectors().get(i).get(label).put(feature, weights.get(i));
                }
            });
        });
        this.setProbabilities(true);
    }

    // train by optimizing the c parametr

    /**
     *
     * @param instances
     * @return
     */
    public static JAROW trainOpt(ArrayList<Instance> instances) {
        Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
        return trainOpt(instances, 10, params, 0.1, true, false);
    }

    /**
     *
     * @param instances
     * @param rounds
     * @param paramValues
     * @param heldout
     * @param adapt
     * @param optimizeProbs
     * @return
     */
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
            Double lowestEntropy = Double.POSITIVE_INFINITY;
            int steps = 20;
            for (int i = 0; i < steps; i++) {
                Double scale = 1.0 - i / (double) steps;
                System.out.println("scale= " + scale);
                bestClassifier.probGeneration(scale);
                Double entropy = bestClassifier.batchPredict(testingInstances, true);

                if (Double.compare(entropy, lowestEntropy) < 0) {
                    bestScale = scale;
                    lowestEntropy = entropy;
                }
            }
        }

        // Now train the final model:

        JAROW finalClassifier = new JAROW();
        finalClassifier.train(instances, true, false, rounds, bestParam, adapt);
        if (optimizeProbs) {
            finalClassifier.probGeneration(bestScale);
        }

        return finalClassifier;
    }

    // save function for the parameters:

    /**
     *
     * @param classifier
     * @param filename
     * @throws IOException
     */
    public static void save(JAROW classifier, String filename) throws IOException {
        FileOutputStream fout = new FileOutputStream(filename);
        try (ObjectOutputStream oos = new ObjectOutputStream(fout)) {
            oos.writeObject(classifier);
        }
    }

    // load function for the parameters:

    /**
     *
     * @param filename
     * @return
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public static JAROW load(String filename) throws IOException, ClassNotFoundException {
        FileInputStream fin = new FileInputStream(filename);
        JAROW classifier;
        try (ObjectInputStream ois = new ObjectInputStream(fin)) {
            classifier = (JAROW) ois.readObject();
        }

        return classifier;
    }

    /**
     *
     * @param a1
     * @param a2
     * @return
     */
    public Double dotProduct(TObjectDoubleHashMap<String> a1, TObjectDoubleHashMap<String> a2) {
        Double product = 0.0;        
        for (String label : a1.keySet()) {
            if (a2.contains(label)) {
                Double v1 = a1.get(label);
                Double v2 = a2.get(label);
                product += v1 * v2;
            }
        }
        return product.doubleValue();
    }

    /**
     *
     * @return
     */
    public boolean isProbabilities() {
        return probabilities;
    }

    /**
     *
     * @return
     */
    public THashMap<String, TObjectDoubleHashMap<String>> getCurrentWeightVectors() {
        return currentWeightVectors;
    }

    /**
     *
     * @return
     */
    public TObjectDoubleHashMap<String> getCurrentGlobalWeightVectors() {
        return currentGlobalWeightVectors;
    }

    /**
     *
     * @return
     */
    public THashMap<String, TObjectDoubleHashMap<String>> getCurrentVarianceVectors() {
        return currentVarianceVectors;
    }

    /**
     *
     * @return
     */
    public ArrayList<THashMap<String, TObjectDoubleHashMap<String>>> getProbWeightVectors() {
        return probWeightVectors;
    }

    /**
     *
     * @return
     */
    public double getParam() {
        return param;
    }

    /**
     * @param probabilities the probabilities to set
     */
    public void setProbabilities(boolean probabilities) {
        this.probabilities = probabilities;
    }

    /**
     * @param currentWeightVectors the currentWeightVectors to set
     */
    public void setCurrentWeightVectors(THashMap<String, TObjectDoubleHashMap<String>> currentWeightVectors) {
        this.currentWeightVectors = currentWeightVectors;
    }

    /**
     * @param currentVarianceVectors the currentVarianceVectors to set
     */
    public void setCurrentVarianceVectors(THashMap<String, TObjectDoubleHashMap<String>> currentVarianceVectors) {
        this.currentVarianceVectors = currentVarianceVectors;
    }

    /**
     * @param probWeightVectors the probWeightVectors to set
     */
    public void setProbWeightVectors(ArrayList<THashMap<String, TObjectDoubleHashMap<String>>> probWeightVectors) {
        this.probWeightVectors = probWeightVectors;
    }

    /**
     * @return the averagedWeightVectors
     */
    public THashMap<String, TObjectDoubleHashMap<String>> getAveragedWeightVectors() {
        return averagedWeightVectors;
    }

    /**
     * @param averagedWeightVectors the averagedWeightVectors to set
     */
    public void setAveragedWeightVectors(THashMap<String, TObjectDoubleHashMap<String>> averagedWeightVectors) {
        this.averagedWeightVectors = averagedWeightVectors;
    }

    /**
     * @param currentGlobalWeightVectors the currentGlobalWeightVectors to set
     */
    public void setCurrentGlobalWeightVectors(TObjectDoubleHashMap<String> currentGlobalWeightVectors) {
        this.currentGlobalWeightVectors = currentGlobalWeightVectors;
    }

    /**
     * @return the currentGlobalVarianceVectors
     */
    public TObjectDoubleHashMap<String> getCurrentGlobalVarianceVectors() {
        return currentGlobalVarianceVectors;
    }

    /**
     * @param currentGlobalVarianceVectors the currentGlobalVarianceVectors to set
     */
    public void setCurrentGlobalVarianceVectors(TObjectDoubleHashMap<String> currentGlobalVarianceVectors) {
        this.currentGlobalVarianceVectors = currentGlobalVarianceVectors;
    }

    /**
     * @return the probGlobalWeightVectors
     */
    public ArrayList<TObjectDoubleHashMap<String>> getProbGlobalWeightVectors() {
        return probGlobalWeightVectors;
    }

    /**
     * @param probGlobalWeightVectors the probGlobalWeightVectors to set
     */
    public void setProbGlobalWeightVectors(ArrayList<TObjectDoubleHashMap<String>> probGlobalWeightVectors) {
        this.probGlobalWeightVectors = probGlobalWeightVectors;
    }

    /**
     * @return the averagedGlobalWeightVectors
     */
    public TObjectDoubleHashMap<String> getAveragedGlobalWeightVectors() {
        return averagedGlobalWeightVectors;
    }

    /**
     * @param averagedGlobalWeightVectors the averagedGlobalWeightVectors to set
     */
    public void setAveragedGlobalWeightVectors(TObjectDoubleHashMap<String> averagedGlobalWeightVectors) {
        this.averagedGlobalWeightVectors = averagedGlobalWeightVectors;
    }

    /**
     * @return the averagingUpdates
     */
    public int getAveragingUpdates() {
        return averagingUpdates;
    }

    /**
     * @param averagingUpdates the averagingUpdates to set
     */
    public void setAveragingUpdates(int averagingUpdates) {
        this.averagingUpdates = averagingUpdates;
    }

    /**
     * @param param the param to set
     */
    public void setParam(double param) {
        this.param = param;
    }
    private static final Logger LOG = Logger.getLogger(JAROW.class.getName());
}
