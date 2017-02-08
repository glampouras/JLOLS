/*
 * Copyright (C) 2016 Black Fox
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

/**
 *
 * @author Gerasimos Lampouras
 */
public class TrainThread extends Thread {

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
    public TrainThread(JAROW classifier, ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, Double param, boolean adapt) {
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
    public void run() {
        classifier.setParam((double) param);
        synchronized (classifier.getCurrentWeightVectors()) {
            classifier.setCurrentWeightVectors(new THashMap<String, TObjectDoubleHashMap<String>>());
        }
        if (adapt) {
            synchronized (classifier.getCurrentVarianceVectors()) {
                classifier.setCurrentVarianceVectors(new THashMap<String, TObjectDoubleHashMap<String>>());
            }
        }
        synchronized (classifier.getCurrentGlobalWeightVectors()) {
            classifier.setCurrentGlobalWeightVectors(new TObjectDoubleHashMap<String>());
        }
        synchronized (classifier.getCurrentGlobalVarianceVectors()) {
            classifier.setCurrentGlobalVarianceVectors(new TObjectDoubleHashMap<String>());
        }
        synchronized (classifier.getAveragedGlobalWeightVectors()) {
            classifier.setAveragedGlobalWeightVectors(new TObjectDoubleHashMap<String>());
        }
        if (averaging) {
            synchronized (classifier.getAveragedWeightVectors()) {
                classifier.setAveragedWeightVectors(new THashMap<String, TObjectDoubleHashMap<String>>());
            }
            classifier.setAveragingUpdates(0);
        }
        if (!instances.isEmpty()) {
            HashSet<String> labels = new HashSet<>();
            for (Instance in : instances) {
                for (String label : in.getCosts().keySet()) {
                    labels.add(label);
                }
            }
            for (String label : labels) {
                synchronized (classifier.getCurrentWeightVectors()) {
                    classifier.getCurrentWeightVectors().put(label, new TObjectDoubleHashMap<String>());
                }
                // remember: this is sparse in the sense that everything that doesn't have a value, is 1
                // everytime we to do something with it, remember to add 1
                if (adapt) {
                    synchronized (classifier.getCurrentVarianceVectors()) {
                        classifier.getCurrentVarianceVectors().put(label, new TObjectDoubleHashMap<String>());
                    }
                }
                // keep the averaged weight vector
                if (averaging && classifier.getAveragedWeightVectors() != null) {
                    synchronized (classifier.getAveragedWeightVectors()) {
                        classifier.getAveragedWeightVectors().put(label, new TObjectDoubleHashMap<String>());
                    }
                }
            }
        }
        // in each iteration        
        for (int r = 0; r < rounds; r++) {
            // shuffle
            if (shuffling) {
                Collections.shuffle(instances, new Random(13));
            }
            // for each instance
            for (Instance instance : instances) {
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
                            if (featureVectorPredicted.get(feature) != 0.0) {
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
                            }
                        }
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            if (featureVectorMinCorrect.get(feature) != 0.0) {
                                // then for the minCorrect:
                                if (!feature.startsWith("global_")) {
                                    if (!classifier.getCurrentVarianceVectors().containsKey(minCorrectLabel)) {
                                        classifier.getCurrentVarianceVectors().put(prediction.getLabel(), new TObjectDoubleHashMap<String>());
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
                        for (String feature : zVectorPredicted.keySet()) {
                            if (!feature.startsWith("global_")) {
                                synchronized (classifier.getCurrentWeightVectors()) {
                                    classifier.getCurrentWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -(alpha * zVectorPredicted.get(feature)), -(alpha * zVectorPredicted.get(feature)));
                                }
                            } else {
                                synchronized (classifier.getCurrentGlobalWeightVectors()) {
                                    classifier.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, -(alpha * zVectorPredicted.get(feature)), -(alpha * zVectorPredicted.get(feature)));
                                }
                            }
                        }
                        for (String feature : zVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                synchronized (classifier.getCurrentWeightVectors()) {
                                    classifier.getCurrentWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, (alpha * zVectorMinCorrect.get(feature)), (alpha * zVectorMinCorrect.get(feature)));
                                }
                            } else {
                                synchronized (classifier.getCurrentGlobalWeightVectors()) {
                                    classifier.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, (alpha * zVectorMinCorrect.get(feature)), (alpha * zVectorMinCorrect.get(feature)));
                                }
                            }
                        }
                        if (averaging && classifier.getAveragedWeightVectors() != null) {
                            for (String feature : zVectorPredicted.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    synchronized (classifier.getAveragedWeightVectors()) {
                                        classifier.getAveragedWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));
                                    }
                                } else {
                                    synchronized (classifier.getAveragedGlobalWeightVectors()) {
                                        classifier.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, (-alpha * zVectorPredicted.get(feature)), (-alpha * zVectorPredicted.get(feature)));

                                    }
                                }
                            }
                            for (String feature : zVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    synchronized (classifier.getAveragedWeightVectors()) {
                                        classifier.getAveragedWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                    }
                                } else {
                                    synchronized (classifier.getAveragedGlobalWeightVectors()) {
                                        classifier.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                                    }
                                }
                            }
                        }
                        // update the diagonal covariance
                        for (String feature : featureVectorPredicted.keySet()) {
                            if (featureVectorPredicted.get(feature) != 0.0) {
                                // for the predicted
                                if (!feature.startsWith("global_")) {
                                    synchronized (classifier.getCurrentVarianceVectors()) {
                                        classifier.getCurrentVarianceVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                                    }
                                } else {
                                    synchronized (classifier.getCurrentGlobalVarianceVectors()) {
                                        classifier.getCurrentGlobalVarianceVectors().adjustOrPutValue(feature, -beta * Math.pow(zVectorPredicted.get(feature), 2), 1 - beta * Math.pow(zVectorPredicted.get(feature), 2));
                                    }
                                }
                            }
                        }
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            if (featureVectorMinCorrect.get(feature) != 0.0) {
                                // for the minCorrect
                                if (!feature.startsWith("global_")) {
                                    synchronized (classifier.getCurrentVarianceVectors()) {
                                        classifier.getCurrentVarianceVectors().get(minCorrectLabel).adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                                    }
                                } else {
                                    synchronized (classifier.getCurrentGlobalVarianceVectors()) {
                                        classifier.getCurrentGlobalVarianceVectors().adjustOrPutValue(feature, -beta * Math.pow(zVectorMinCorrect.get(feature), 2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature), 2));
                                    }
                                }
                            }
                        }
                    } else {
                        // the squared norm is twice the square of the features since they are the same per class 
                        //The value specific features can be safely ignored since they are not shared between values and would result in 0s anyway
                        Double norm = 2.0 * classifier.dotProduct(instance.getGeneralFeatureVector(), instance.getGeneralFeatureVector());
                        Double factor = loss / (norm + 1.0 / (2.0 * param));

                        TObjectDoubleHashMap<String> featureVectorPredicted = instance.getFeatureVector(prediction.getLabel());
                        for (String feature : featureVectorPredicted.keySet()) {
                            if (!feature.startsWith("global_")) {
                                synchronized (classifier.getCurrentWeightVectors()) {
                                    classifier.getCurrentWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                }
                            } else {
                                synchronized (classifier.getCurrentGlobalWeightVectors()) {
                                    classifier.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                }
                            }
                        }
                        TObjectDoubleHashMap<String> featureVectorMinCorrect = instance.getFeatureVector(minCorrectLabel);
                        for (String feature : featureVectorMinCorrect.keySet()) {
                            if (!feature.startsWith("global_")) {
                                synchronized (classifier.getCurrentWeightVectors()) {
                                    classifier.getCurrentWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                }
                            } else {
                                synchronized (classifier.getCurrentGlobalWeightVectors()) {
                                    classifier.getCurrentGlobalWeightVectors().adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                }
                            }
                        }
                        if (averaging && classifier.getAveragedWeightVectors() != null) {
                            for (String feature : featureVectorPredicted.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    synchronized (classifier.getAveragedWeightVectors()) {
                                        classifier.getAveragedWeightVectors().get(prediction.getLabel()).adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                    }
                                } else {
                                    synchronized (classifier.getAveragedGlobalWeightVectors()) {
                                        classifier.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, -factor * featureVectorPredicted.get(feature), -factor * featureVectorPredicted.get(feature));
                                    }
                                }
                            }
                            for (String feature : featureVectorMinCorrect.keySet()) {
                                if (!feature.startsWith("global_")) {
                                    synchronized (classifier.getAveragedWeightVectors()) {
                                        classifier.getAveragedWeightVectors().get(minCorrectLabel).adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));
                                    }
                                } else {
                                    synchronized (classifier.getAveragedGlobalWeightVectors()) {
                                        classifier.getAveragedGlobalWeightVectors().adjustOrPutValue(feature, factor * featureVectorMinCorrect.get(feature), factor * featureVectorMinCorrect.get(feature));

                                    }
                                }
                            }
                        }
                    }
                }
                if (averaging) {
                    classifier.setAveragingUpdates(classifier.getAveragingUpdates() + 1);
                }
            }
        }

        if (averaging && classifier.getAveragedWeightVectors() != null) {
            for (String label : classifier.getCurrentWeightVectors().keySet()) {
                for (String feature : classifier.getAveragedWeightVectors().get(label).keySet()) {
                    synchronized (classifier.getCurrentWeightVectors()) {
                        classifier.getCurrentWeightVectors().get(label).put(feature, classifier.getAveragedWeightVectors().get(label).get(feature) / ((double) classifier.getAveragingUpdates()));
                    }
                }
            }
            for (String feature : classifier.getAveragedGlobalWeightVectors().keySet()) {
                synchronized (classifier.getCurrentGlobalWeightVectors()) {
                    classifier.getCurrentGlobalWeightVectors().put(feature, classifier.getAveragedGlobalWeightVectors().get(feature) / ((double) classifier.getAveragingUpdates()));
                }
            }
        }
    }
}