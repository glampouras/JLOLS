/*
 * Copyright (C) 2016 lampouras
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

import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.logging.Logger;

/**
 *
 * @author Gerasimos Lampouras
 */
public class TrainAdditionalThread extends Thread {

    JAROW classifier;

    ArrayList<Instance> instances;
    boolean averaging;
    boolean shuffling;
    int rounds;
    Double param;
    boolean adapt;

    /**
     *
     * @param classifier
     * @param instances
     * @param averaging
     * @param shuffling
     * @param rounds
     * @param param
     * @param adapt
     */
    public TrainAdditionalThread(JAROW classifier, ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, Double param, boolean adapt) {
        this.classifier = classifier;

        this.instances = instances;

        this.averaging = averaging;
        this.shuffling = shuffling;
        this.rounds = rounds;
        this.param = param;
        this.adapt = adapt;
    }

    /**
     *
     */
    @Override
    public void run() {
        for (int r = 0; r < rounds; r++) {
            // shuffle
            if (shuffling) {
                Collections.shuffle(instances, new Random(13));
            }
            // for each instance
            instances.stream().map((instance) -> {
                Prediction prediction = classifier.predict(instance);
                // so if the prediction was incorrect
                // we are no longer large margin, since we are using the loss from the cost-sensitive PA
                if (Double.compare(instance.getCosts().get(prediction.getLabel()), 0) > 0) {
                    // first we need to get the score for the correct answer
                    // if the instance has more than one correct answer then pick the min
                    Double minCorrectLabelScore = Double.POSITIVE_INFINITY;
                    String minCorrectLabel = null;
                    for (String label : instance.getCorrectLabels()) {
                        Double score = classifier.dotProduct(instance.getFeatureVector(label), classifier.getWeightVector(label));
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
                        featureVectorPredicted.keySet().stream().filter((feature) -> (featureVectorPredicted.get(feature) != 0.0)).forEachOrdered((feature) -> {
                            // the variance is either some value that is in the dict or just 1
                            if (!feature.startsWith("global_")) {
                                if (!classifier.getCurrentVarianceVectors().containsKey(prediction.getLabel())) {
                                    classifier.getCurrentVarianceVectors().put(prediction.getLabel(), new TObjectDoubleHashMap<String>());
                                }
                                if (classifier.getCurrentVarianceVectors().get(prediction.getLabel()).containsKey(feature)) {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * classifier.getCurrentVarianceVectors().get(prediction.getLabel()).get(feature));
                                } else {
                                    zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                                }
                            } else if (classifier.getCurrentGlobalVarianceVectors().containsKey(feature)) {
                                zVectorPredicted.put(feature, featureVectorPredicted.get(feature) * classifier.getCurrentGlobalVarianceVectors().get(feature));
                            } else {
                                zVectorPredicted.put(feature, featureVectorPredicted.get(feature));
                            }
                        });
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            if (featureVectorMinCorrect.get(feature) != 0.0) {
                                // then for the minCorrect:
                                if (!feature.startsWith("global_")) {
                                    if (!classifier.getCurrentVarianceVectors().containsKey(minCorrectLabel)) {
                                        classifier.getCurrentVarianceVectors().put(minCorrectLabel, new TObjectDoubleHashMap<String>());
                                    }
                                    if (classifier.getCurrentVarianceVectors().get(minCorrectLabel).containsKey(feature)) {
                                        zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * classifier.getCurrentVarianceVectors().get(minCorrectLabel).get(feature));
                                    } else {
                                        zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                                    }
                                } else if (classifier.getCurrentGlobalVarianceVectors().containsKey(feature)) {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature) * classifier.getCurrentGlobalVarianceVectors().get(feature));
                                } else {
                                    zVectorMinCorrect.put(feature, featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        Double confidence = classifier.dotProduct(zVectorPredicted, featureVectorPredicted) + classifier.dotProduct(zVectorMinCorrect, featureVectorMinCorrect);
                        Double beta = 1.0 / (confidence + param);
                        Double alpha = loss * beta;
                        // update the current weight vectors
                        zVectorPredicted.keySet().forEach((feature) -> {
                            if (!feature.startsWith("global_")) {
                                classifier.getCurrentWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -alpha * zVectorPredicted.get(feature), -alpha * zVectorPredicted.get(feature));
                            } else {
                                classifier.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, -alpha * zVectorPredicted.get(feature), -alpha * zVectorPredicted.get(feature));
                            }
                        });
                        for (String feature : zVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                classifier.getCurrentWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                            } else {
                                classifier.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                            }
                        }
                        if (averaging && classifier.getAveragedWeightVectors() != null) {
                            zVectorPredicted.keySet().forEach((feature) -> {
                                if (!feature.startsWith("global_")) {
                                    classifier.getAveragedWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                } else {
                                    classifier.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                }
                            });
                            for (String feature : zVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    classifier.getAveragedWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                } else {
                                    classifier.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        // update the diagonal covariance
                        featureVectorPredicted.keySet().forEach((feature) -> {
                            // for the predicted
                            if (!feature.startsWith("global_")) {
                                classifier.getCurrentVarianceVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            } else {
                                classifier.getCurrentGlobalVarianceVectors().adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                            }
                        });
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            // for the minCorrect
                            if (!feature.startsWith("global_")) {
                                classifier.getCurrentVarianceVectors().get(minCorrectLabel).adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            } else {
                                classifier.getCurrentGlobalVarianceVectors().adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                            }
                        }
                        classifier.getCurrentVarianceVectors().get(prediction.getLabel()).keySet().stream().filter((s) -> (classifier.getCurrentVarianceVectors().get(prediction.getLabel()).get(s) < 0)).forEachOrdered((_item) -> {
                            System.out.println(classifier.getCurrentVarianceVectors().get(prediction.getLabel()));
                        });
                        for (String s : classifier.getCurrentVarianceVectors().get(minCorrectLabel).keySet()) {
                            if (classifier.getCurrentVarianceVectors().get(minCorrectLabel).get(s) < 0) {
                            }
                        }
                        classifier.getCurrentGlobalVarianceVectors().keySet().stream().filter((s) -> (classifier.getCurrentGlobalVarianceVectors().get(s) < 0)).forEachOrdered((_item) -> {
                            System.out.println(classifier.getCurrentGlobalVarianceVectors());
                        });
                    } else {
                        // the squared norm is twice the square of the features since they are the same per class 
                        Double norm = 2.0 * classifier.dotProduct(instance.getGeneralFeatureVector(), instance.getGeneralFeatureVector());
                        Double factor = loss / (norm + 1.0 / (2.0 * param));
                        TObjectDoubleHashMap<String> featureVectorPredicted = instance.getFeatureVector(prediction.getLabel());
                        featureVectorPredicted.keySet().forEach((feature) -> {
                            if (!feature.startsWith("global_")) {
                                classifier.getCurrentWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            } else {
                                classifier.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                            }
                        });
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                classifier.getCurrentWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            } else {
                                classifier.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                            }
                        }
                        if (averaging && classifier.getAveragedWeightVectors() != null) {
                            featureVectorPredicted.keySet().forEach((feature) -> {
                                if (!feature.startsWith("global_")) {
                                    classifier.getAveragedWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                } else {
                                    classifier.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                }
                            });
                            for (String feature : featureVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    classifier.getAveragedWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                } else {
                                    classifier.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                    }
                }
                return instance;
            }).filter((_item) -> (averaging)).forEachOrdered((_item) -> {
                classifier.setAveragingUpdates(classifier.getAveragingUpdates() + 1);
            }); //System.out.println("Training error rate in round " + r + " : " + (double) errorsInRound / (double) instances.size());
        }

        if (averaging && classifier.getAveragedWeightVectors() != null) {
            classifier.getCurrentWeightVectors().keySet().forEach((label) -> {
                classifier.getAveragedWeightVectors().get(label).keySet().forEach((feature) -> {
                    classifier.getCurrentWeightVectors().get(label).put(feature, classifier.getAveragedWeightVectors().get(label).get(feature) / classifier.getAveragingUpdates());
                });
            });
            classifier.getAveragedGlobalWeightVectors().keySet().forEach((feature) -> {
                classifier.getCurrentGlobalWeightVectors().put(feature, classifier.getAveragedGlobalWeightVectors().get(feature) / classifier.getAveragingUpdates());
            });
        }
    }
    private static final Logger LOG = Logger.getLogger(TrainAdditionalThread.class.getName());
}
