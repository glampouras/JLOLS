package uk.ac.ucl.jdagger.jarow;

// cost-sensitive multiclass classification with AROW

import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.set.hash.THashSet;
import java.util.ArrayList;


// the instances consist of a HashMap of labels to costs and feature vectors (Huang-style)

public class Instance {
    public static ArrayList<Instance> removeHapaxLegomena(ArrayList<Instance> instances) {
        System.out.println("Counting features");
        
        TObjectIntHashMap<String> feature2counts = new TObjectIntHashMap<>();
        
        for (Instance instance : instances) {
            for (String element : instance.getFeatureVector().keySet()) {
                feature2counts.adjustOrPutValue(element, 1, 1);
            }
        }

        System.out.println("Removing hapax legomena");
        ArrayList<Instance> newInstances = new ArrayList<>();
        for (Instance instance : instances) {
            TObjectDoubleHashMap<String> newFeatureVector = new TObjectDoubleHashMap<>();
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
    
    private TObjectDoubleHashMap<String> featureVector = null;
    private TObjectDoubleHashMap<String> costs = null;
    
    private Double minCost = Double.POSITIVE_INFINITY;
    private Double maxCost = Double.NEGATIVE_INFINITY;
    
    private THashSet<String> worstLabels;
    private THashSet<String> correctLabels;
    
    public Instance(TObjectDoubleHashMap<String> featureVector) {
        this(featureVector, null);
    }
    
    public Instance(TObjectDoubleHashMap<String> featureVector, TObjectDoubleHashMap<String> costs) {
        this.featureVector = featureVector;
        
        // we assume that the label with the lowest cost has a cost of zero and the rest increment on that
        // find out which are the correct answer, assuming it has a cost of zero        
        this.costs = costs;
                
        if (this.costs != null) {
            this.worstLabels = new THashSet<>();
            this.correctLabels = new THashSet<>();
            
            for (String label : this.costs.keySet()) {
                Double cost = this.costs.get(label);
                
                int compare = Double.compare(cost, this.minCost);
                if (compare < 0) {
                    this.minCost = cost;
                    
                    this.correctLabels = new THashSet<>();
                    this.correctLabels.add(label);
                } else if (compare == 0) {
                    this.correctLabels.add(label);
                }
                compare = Double.compare(cost, this.maxCost);
                if (compare > 0) {
                    this.maxCost = cost;
                    
                    this.worstLabels = new THashSet<>();
                    this.worstLabels.add(label);
                } else if (compare == 0) {
                    this.worstLabels.add(label);
                }
            }

            int compare = Double.compare(this.minCost, 0.0);
            if (compare > 0) {                
                for (String label : this.costs.keySet()) {                    
                    this.costs.adjustValue(label, - this.minCost);
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
            retString += label + ":" + this.costs.get(label);
        }

        retString += "\t";
        
        isFirst = true;
        for (String feature : this.featureVector.keySet()) {
            if (!isFirst) {
                retString += " ";
            } else {
                isFirst = !isFirst;
            }
            retString += feature + ":" + this.featureVector.get(feature);
        }
        return retString;
    }

    public TObjectDoubleHashMap<String> getFeatureVector() {
        return featureVector;
    }

    public TObjectDoubleHashMap<String> getCosts() {
        return costs;
    }

    public Double getMaxCost() {
        return maxCost;
    }

    public THashSet<String> getCorrectLabels() {
        return correctLabels;
    }
}
