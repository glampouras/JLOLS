package jarow;

// cost-sensitive multiclass classification with AROW

import java.util.HashMap;
import java.util.HashSet;

// the instances consist of a HashMap of labels to costs and feature vectors (Huang-style)

public class Instance {
    public static HashSet<Instance> removeHapaxLegomena(HashSet<Instance> instances) {
        System.out.println("Counting features");
        
        HashMap<String, Integer> feature2counts = new HashMap<>();
        
        for (Instance instance : instances) {
            for (String element : instance.getFeatureVector().keySet()) {
                feature2counts.put(element, feature2counts.get(element) + 1);
            }
        }

        System.out.println("Removing hapax legomena");
        HashSet<Instance> newInstances = new HashSet<>();
        for (Instance instance : instances) {
            HashMap<String, Double> newFeatureVector = new HashMap<>();
            for (String element : instance.getFeatureVector().keySet()) {
                // if this feature was encountered more than once
                if (feature2counts.get(element) > 1) {
                    newFeatureVector.put(element, instance.getFeatureVector().get(element));
                }
            }
            newInstances.add(new Instance(newFeatureVector, instance.getCosts()));
        }
        return newInstances;
    }
    
    private HashMap<String, Double> featureVector = null;
    private HashMap<String, Double> costs = null;
    
    private Double minCost = Double.POSITIVE_INFINITY;
    private Double maxCost = Double.NEGATIVE_INFINITY;
    
    private HashSet<String> worstLabels;
    private HashSet<String> correctLabels;
    
    public Instance(HashMap<String, Double> featureVector) {
        this(featureVector, null);
    }
    
    public Instance(HashMap<String, Double> featureVector, HashMap<String, Double> costs) {
        this.featureVector = featureVector;
        
        // we assume that the label with the lowest cost has a cost of zero and the rest increment on that
        // find out which are the correct answer, assuming it has a cost of zero        
        this.costs = costs;
                
        if (this.costs != null) {
            this.worstLabels = new HashSet<>();
            this.correctLabels = new HashSet<>();
            
            for (String label : this.costs.keySet()) {
                Double cost = this.costs.get(label);
                
                int compare = Double.compare(cost, this.minCost);
                if (compare < 0) {
                    this.minCost = cost;
                    
                    this.correctLabels = new HashSet<>();
                    this.correctLabels.add(label);
                } else if (compare == 0) {
                    this.correctLabels.add(label);
                }
                compare = Double.compare(cost, this.maxCost);
                if (compare > 0) {
                    this.maxCost = cost;
                    
                    this.worstLabels = new HashSet<>();
                    this.worstLabels.add(label);
                } else if (compare == 0) {
                    this.worstLabels.add(label);
                }
            }

            int compare = Double.compare(this.minCost, 0.0);
            if (compare > 0) {                
                for (String label : this.costs.keySet()) {
                    this.costs.put(label, this.costs.get(label) - this.minCost);
                }
                this.maxCost -= this.minCost;
            }
        }
    }
    
    @Override
    public String toString() {
        String retString = "";
        
        boolean isFirst = true;
        for (String label : this.costs.keySet()) {
            if (!isFirst) {
                retString += ",";
            } else {
                isFirst = !isFirst;
            }
            retString += label + ":" + this.costs.get(label).toString();
        }

        retString += "\t";
        
        isFirst = true;
        for (String feature : this.featureVector.keySet()) {
            if (!isFirst) {
                retString += " ";
            } else {
                isFirst = !isFirst;
            }
            retString += feature + ":" + this.featureVector.get(feature).toString();
        }
        return retString;
    }

    public HashMap<String, Double> getFeatureVector() {
        return featureVector;
    }

    public HashMap<String, Double> getCosts() {
        return costs;
    }

    public Double getMaxCost() {
        return maxCost;
    }

    public HashSet<String> getCorrectLabels() {
        return correctLabels;
    }
}
