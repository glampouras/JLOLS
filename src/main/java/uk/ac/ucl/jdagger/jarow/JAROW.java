package uk.ac.ucl.jdagger.jarow;

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
import java.util.Random;

public class JAROW implements Serializable {
    private boolean probabilities;
    private THashMap<String, TObjectDoubleHashMap<String>> currentWeightVectors;
    private THashMap<String, TObjectDoubleHashMap<String>> currentVarianceVectors;
    private ArrayList<THashMap<String, TObjectDoubleHashMap<String>>> probWeightVectors;
    
    public JAROW() {
        this.probabilities = false;
        this.currentWeightVectors = new THashMap<>();
        this.currentVarianceVectors = new THashMap<>();
    }
        
    // This predicts always using the current weight vectors
    
    public Prediction predict(Instance instance) {
        return predict(instance, false, false);
    }
    
    public Prediction predict(Instance instance, boolean verbose/* = false*/, boolean probabilities/* = false*/) {
        //# always add the bias
        instance.getFeatureVector().put("biasAutoAdded", 1.0);

        Prediction prediction = new Prediction();
        
        for (String label : currentWeightVectors.keySet()) {
            TObjectDoubleHashMap<String> weightVector = currentWeightVectors.get(label);
            
            double score = dotProduct(instance.getFeatureVector(), weightVector);
            prediction.getLabel2score().put(label, score);
            if (Double.compare(score, prediction.getScore()) > 0) {
                prediction.setScore(score);
                prediction.setLabel(label);
            }
        }

        if (verbose) {
            for (String feature : instance.getFeatureVector().keySet()) {
                // keep the feature weights for the predicted label
                ArrayList<Object> fvw = new ArrayList<>();
                fvw.add(feature);
                fvw.add(instance.getFeatureVector().get(feature));
                fvw.add(this.currentWeightVectors.get(prediction.getLabel()).get(feature));
                        
                prediction.getFeatureValueWeights().add(fvw);
            }
            // order them from the most positive to the most negative
            // TO-DO: This!
            //prediction.featureValueWeights = sorted(prediction.featureValueWeights, key=itemgetter(2))
        }
        if (probabilities) {
            // if we have probabilistic training
            if (this.probabilities) {
                TObjectDoubleHashMap<String> probPredictions = new TObjectDoubleHashMap<>();
                for (String label : this.probWeightVectors.get(0).keySet()) {
                    // smoothing the probabilities with add 0.01 of 1 out of the vectors
                    probPredictions.put(label, 0.01/this.probWeightVectors.size());
                }
                // for each of the weight vectors obtained get its prediction
                for (THashMap<String, TObjectDoubleHashMap<String>> probWeightVector : this.probWeightVectors) {
                    Double maxScore = Double.NEGATIVE_INFINITY;
                    String maxLabel = null;
                    for (String label : probWeightVector.keySet()) {
                        TObjectDoubleHashMap<String> weightVector = probWeightVector.get(label);
                        Double score = dotProduct(instance.getFeatureVector(), weightVector);
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
                    prediction.getLabel2prob().put(label, score/(double)this.probWeightVectors.size());
                }
                // Also compute the entropy:
                for (Double prob : prediction.getLabel2prob().values()) {
                    if (Double.compare(prob, 0.0) > 0) {
                        prediction.setEntropy(prediction.getEntropy() - prob * (Math.log(prob) / Math.log(2)));
                    }
                }
                // normalize it:
                prediction.setEntropy(prediction.getEntropy() / (Math.log(prediction.getLabel2prob().size()) / Math.log(2)));
                //print prediction.label2probs
                //print prediction.entropy
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
    
    // the parameter here is for AROW learning
    // adapt if True is AROW, if False it is passive aggressive-II with prediction-based updates 
    public Double train(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, Double param) {
        return train(instances, averaging, shuffling, rounds, param, true);
    }
    
    public Double train(ArrayList<Instance> instances, boolean averaging, boolean shuffling, int rounds, Double param, boolean adapt) {
        // we first need to go through the dataset to find how many classes

        // Initialize the weight vectors in the beginning of training"
        // we have one variance and one weight vector per class
        this.currentWeightVectors = new THashMap<>();
        if (adapt) {
            this.currentVarianceVectors = new THashMap<>();
        }        
        THashMap<String, TObjectDoubleHashMap<String>> averagedWeightVectors = null;
        int updatesLeft = 0;
        if (averaging) {
            averagedWeightVectors = new THashMap<>();
            updatesLeft = rounds * instances.size();
        }
        for (String label : instances.get(0).getCosts().keySet()) {
            this.currentWeightVectors.put(label, new TObjectDoubleHashMap<String>());
            // TO-DO: deal with sparse
            // remember: this is sparse in the sense that everething that doesn't have a value is 1
            // everytime we to do something with it, remember to add 1
            if (adapt) {
                this.currentVarianceVectors.put(label, new TObjectDoubleHashMap<String>());
            }
            // keep the averaged weight vector
            if (averaging && averagedWeightVectors != null) {
                averagedWeightVectors.put(label, new TObjectDoubleHashMap<String>());
            }
        }

        // in each iteration        
        for (int r = 0; r < rounds; r++) {
            // shuffle
            if (shuffling) {
                Collections.shuffle(instances, new Random());
            }
            int errorsInRound = 0;
            // TO-DO find out what that is for?
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
                        if (this.currentWeightVectors.get(label) == null) {
                            System.out.println("WTF " + label);
                        }
                        Double score = dotProduct(instance.getFeatureVector(), this.currentWeightVectors.get(label));
                        if (Double.compare(score, minCorrectLabelScore) < 0) {
                            minCorrectLabelScore = score;
                            minCorrectLabel = label;
                        }
                    }
                        
                    // the loss is the scaled margin loss also used by Mejer and Crammer 2010
                    Double loss = prediction.getScore() - minCorrectLabelScore + Math.sqrt(instance.getCosts().get(prediction.getLabel()));
                    if (adapt) {
                        // Calculate the confidence values
                        // first for the predicted label 
                        TObjectDoubleHashMap<String> zVectorPredicted = new TObjectDoubleHashMap<>();
                        TObjectDoubleHashMap<String> zVectorMinCorrect = new TObjectDoubleHashMap<>();
                        for (String feature : instance.getFeatureVector().keySet()) {
                            // the variance is either some value that is in the dict or just 1
                            if (this.currentVarianceVectors.get(prediction.getLabel()).containsKey(feature)) {
                                zVectorPredicted.put(feature, instance.getFeatureVector().get(feature) * this.currentVarianceVectors.get(prediction.getLabel()).get(feature));
                            } else {
                                zVectorPredicted.put(feature, instance.getFeatureVector().get(feature));
                            }
                            // then for the minCorrect:
                            if (this.currentVarianceVectors.get(minCorrectLabel).containsKey(feature)) {
                                zVectorMinCorrect.put(feature, instance.getFeatureVector().get(feature) * this.currentVarianceVectors.get(minCorrectLabel).get(feature));
                            } else {
                                zVectorMinCorrect.put(feature, instance.getFeatureVector().get(feature));
                            }
                        }
                        Double confidence = dotProduct(zVectorPredicted, instance.getFeatureVector()) + dotProduct(zVectorMinCorrect, instance.getFeatureVector());
                        Double beta = 1.0/(confidence + param);
                        Double alpha = loss * beta;

                        // update the current weight vectors
                        for (String feature : zVectorPredicted.keySet()) {
                            //this.currentWeightVectors.get(prediction.getLabel()).put(feature, this.currentWeightVectors.get(prediction.getLabel()).get(feature) - alpha * zVectorPredicted.get(feature));
                            this.currentWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, - alpha * zVectorPredicted.get(feature), - alpha * zVectorPredicted.get(feature));
                        }                    
                        for (String feature : zVectorMinCorrect.keySet()) {
                            //this.currentWeightVectors.get(minCorrectLabel).put(feature, this.currentWeightVectors.get(minCorrectLabel).get(feature) + alpha * zVectorMinCorrect.get(feature));
                            this.currentWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, alpha * zVectorMinCorrect.get(feature), alpha * zVectorMinCorrect.get(feature));
                        }
                        if (averaging && averagedWeightVectors != null) {
                            for (String feature : zVectorPredicted.keySet()) {
                                //averagedWeightVectors.get(prediction.getLabel()).put(feature, averagedWeightVectors.get(prediction.getLabel()).get(feature) - alpha * updatesLeft* zVectorPredicted.get(feature));
                                averagedWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, - alpha * updatesLeft* zVectorPredicted.get(feature), - alpha * updatesLeft* zVectorPredicted.get(feature));
                            }
                            for (String feature : zVectorMinCorrect.keySet()) {
                                //averagedWeightVectors.get(minCorrectLabel).put(feature, averagedWeightVectors.get(minCorrectLabel).get(feature) + alpha * updatesLeft* zVectorMinCorrect.get(feature));
                                averagedWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, alpha * updatesLeft* zVectorMinCorrect.get(feature), alpha * updatesLeft* zVectorMinCorrect.get(feature));
                            }
                        }
                        // update the diagonal covariance
                        for (String feature : instance.getFeatureVector().keySet()) {
                            // for the predicted
                            /*if (this.currentVarianceVectors.get(prediction.getLabel()).containsKey(feature)) {
                                this.currentVarianceVectors.get(prediction.getLabel()).put(feature, this.currentVarianceVectors.get(prediction.getLabel()).get(feature) - beta * Math.pow(zVectorPredicted.get(feature),2));
                            } else {
                                // Never updated this covariance before, add 1
                                this.currentVarianceVectors.get(prediction.getLabel()).put(feature, 1 - beta * Math.pow(zVectorPredicted.get(feature),2));
                            }*/
                            this.currentVarianceVectors.get(prediction.getLabel()).adjustOrPutValue(feature, - beta * Math.pow(zVectorPredicted.get(feature),2), 1 - beta * Math.pow(zVectorPredicted.get(feature),2));
                            // for the minCorrect
                            /*if (this.currentVarianceVectors.get(minCorrectLabel).containsKey(feature)) {
                                this.currentVarianceVectors.get(minCorrectLabel).put(feature, this.currentVarianceVectors.get(minCorrectLabel).get(feature) - beta * Math.pow(zVectorMinCorrect.get(feature),2));
                            } else {
                                // Never updated this covariance before, add 1
                                this.currentVarianceVectors.get(minCorrectLabel).put(feature, 1 - beta * Math.pow(zVectorMinCorrect.get(feature),2));
                            }*/
                            this.currentVarianceVectors.get(minCorrectLabel).adjustOrPutValue(feature, - beta * Math.pow(zVectorMinCorrect.get(feature),2), 1 - beta * Math.pow(zVectorMinCorrect.get(feature),2));
                        }
                    } else {
                        // the squared norm is twice the square of the features since they are the same per class 
                        Double norm = 2.0 * dotProduct(instance.getFeatureVector(), instance.getFeatureVector());
                        Double factor = loss / (norm + 1.0 / (2.0 * param));
                         
                        for (String feature : instance.getFeatureVector().keySet()) {
                            this.currentWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, - factor * instance.getFeatureVector().get(feature), - factor * instance.getFeatureVector().get(feature));
                        }
                        for (String feature : instance.getFeatureVector().keySet()) {
                            this.currentWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, factor * instance.getFeatureVector().get(feature), factor * instance.getFeatureVector().get(feature));
                        }
                        if (averaging && averagedWeightVectors != null) {
                            for (String feature : instance.getFeatureVector().keySet()) {
                                averagedWeightVectors.get(prediction.getLabel()).adjustOrPutValue(feature, - factor * updatesLeft * instance.getFeatureVector().get(feature), - factor * updatesLeft * instance.getFeatureVector().get(feature));
                            }
                            for (String feature : instance.getFeatureVector().keySet()) {
                                averagedWeightVectors.get(minCorrectLabel).adjustOrPutValue(feature, factor * updatesLeft * instance.getFeatureVector().get(feature), factor * updatesLeft * instance.getFeatureVector().get(feature));
                            }
                        }
                    }
                }
                if (averaging) {
		    updatesLeft -= 1;
                }
            }
            System.out.println("Training error rate in round " + r + " : " + (double)errorsInRound/(double)instances.size());
        }
	    
	if (averaging && averagedWeightVectors != null) {
            for (String label : this.currentWeightVectors.keySet()) {
                for (String feature : averagedWeightVectors.get(label).keySet()) {
                    //this.currentWeightVectors.get(label).put(feature, this.currentWeightVectors.get(label).get(feature) + 1.0/((double)(rounds * instances.size())) * averagedWeightVectors.get(label).get(feature));
                    this.currentWeightVectors.get(label).put(feature, averagedWeightVectors.get(label).get(feature) * (1.0/((double)(rounds * instances.size()))));
                }
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

        Double finalTrainingErrorRate = (double)finalTrainingErrors/(double)instances.size();
        System.out.println("Final training error rate=" + finalTrainingErrorRate.toString());
        System.out.println("Final training cost=" + finalTrainingCost.toString());

        return finalTrainingCost;
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
        for (int i  = 0; i < noWeightVectors; i++) {
            this.probWeightVectors.add(new THashMap<String, TObjectDoubleHashMap<String>>());
            for (String label : this.currentWeightVectors.keySet()) {
                this.probWeightVectors.get(i).put(label, new TObjectDoubleHashMap<String>());
            }
        }

        RandomGaussian gaussian = new RandomGaussian();
        for (String label : this.currentWeightVectors.keySet()) {
            // We are ignoring features that never got their weight set 
            for (String feature : this.currentWeightVectors.get(label).keySet()) {
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
                
        System.out.println("done");
        this.probabilities = true;
    }
    
    // train by optimizing the c parametr
    public static JAROW trainOpt(ArrayList<Instance> instances) {
        Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
        return trainOpt(instances, 10, params, 0.2, true, false);
    }
    public static JAROW trainOpt(ArrayList<Instance> instances, int rounds, Double[] paramValues, Double heldout, boolean adapt, boolean optimizeProbs) {
        System.out.println("Training with " + instances.size() + " instances");

        // this value will be kept if nothing seems to work better
        double bestParam = 1;
        Double lowestCost = Double.POSITIVE_INFINITY;
        JAROW bestClassifier = null;
        ArrayList<Instance> trainingInstances = new ArrayList(instances.subList(0, (int) Math.round(instances.size() * (1 - heldout))));
        int to = ((int) Math.round(instances.size() * (1 - heldout))) + 1;
        if (to >= instances.size()) {
            to = instances.size() - 1;
        }
        ArrayList<Instance> testingInstances = new ArrayList(instances.subList(to, instances.size()));
        for (Double param : paramValues) {
            System.out.println("Training with param=" + param + " on " + trainingInstances.size() + " instances");
            // Keep the weight vectors produced in each round
            JAROW classifier = new JAROW();
            classifier.train(trainingInstances, true, true, rounds, param, true);
            System.out.println("testing on " + testingInstances.size() + " instances");
            // Test on the dev for the weight vector produced in each round
            Double devCost = classifier.batchPredict(testingInstances);
            System.out.println("Dev cost:" + devCost + " avg cost per instance " + devCost/(double)testingInstances.size());

            if (devCost < lowestCost) {
                bestParam = param;
                lowestCost = devCost;
                bestClassifier = classifier;
            }
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
                Double scale = 1.0 - (double)i/(double)steps;
                System.out.println("scale= " +  scale);
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
        finalClassifier.train(instances, true, true, rounds, bestParam, true);
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
    
    public double dotProduct(TObjectDoubleHashMap<String> a1, TObjectDoubleHashMap<String> a2) {        
        double product = 0.0;
        for (String label : a1.keySet()) {
            if (a2.contains(label)) {
                product += a1.get(label) * a2.get(label);
            }
        }
        return product;
    }
}