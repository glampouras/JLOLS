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

import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.set.hash.THashSet;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

// the instances consist of a HashMap of labels to costs and feature vectors (Huang-style)
/**
 * @author Gerasimos Lampouras
 */
public class Instance implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     *
     * @param instances
     * @return
     */
    public static ArrayList<Instance> removeHapaxLegomena(ArrayList<Instance> instances) {
        System.out.println("Counting features");

        TObjectIntHashMap<String> generalFeature2counts = new TObjectIntHashMap<>();
        TObjectIntHashMap<String> valueSpecificFeature2counts = new TObjectIntHashMap<>();

        for (Instance instance : instances) {
            for (String element : instance.getGeneralFeatureVector().keySet()) {
                generalFeature2counts.adjustOrPutValue(element, 1, 1);
            }
            for (String value : instance.getValueSpecificFeatureVector().keySet()) {
                for (String element : instance.getValueSpecificFeatureVector().keySet()) {
                    valueSpecificFeature2counts.adjustOrPutValue(value + "=" + element, 1, 1);
                }
            }
        }

        System.out.println("Removing hapax legomena");
        ArrayList<Instance> newInstances = new ArrayList<>();
        for (Instance instance : instances) {
            TObjectDoubleHashMap<String> newFeatureVector = new TObjectDoubleHashMap<>();
            for (String element : instance.getGeneralFeatureVector().keySet()) {
                // if this feature was encountered more than once
                if (generalFeature2counts.get(element) > 1) {
                    newFeatureVector.put(element, instance.getGeneralFeatureVector().get(element));
                }
            }
            HashMap<String, TObjectDoubleHashMap<String>> newValueSpecificFeatureVector = null;
            if (instance.getValueSpecificFeatureVector() != null) {
                newValueSpecificFeatureVector = new HashMap<>();
                for (String value : instance.getValueSpecificFeatureVector().keySet()) {
                    for (String element : instance.getValueSpecificFeatureVector().keySet()) {
                        // if this feature was encountered more than once
                        if (generalFeature2counts.get(value + "=" + element) > 1) {
                            if (!newValueSpecificFeatureVector.containsKey(value)) {
                                newValueSpecificFeatureVector.put(value, new TObjectDoubleHashMap<String>());
                            }
                            newValueSpecificFeatureVector.get(value).put(element, instance.getValueSpecificFeatureVector().get(value).get(element));
                        }
                    }
                }
            }
            newInstances.add(new Instance(newFeatureVector, newValueSpecificFeatureVector, instance.getCosts()));
        }
        return newInstances;
    }

    private TObjectDoubleHashMap<String> generalFeatureVector = null;
    HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatureVector = null;
    private TObjectDoubleHashMap<String> costs = null;

    private Double minCost = Double.POSITIVE_INFINITY;
    private Double maxCost = Double.NEGATIVE_INFINITY;

    private THashSet<String> worstLabels;
    private THashSet<String> correctLabels;

    /**
     *
     * @param generalFeatureVector
     * @param valueSpecificFeatureVector
     */
    public Instance(TObjectDoubleHashMap<String> generalFeatureVector, HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatureVector) {
        this(generalFeatureVector, valueSpecificFeatureVector, null);
    }

    /**
     *
     * @param generalFeatureVector
     * @param valueSpecificFeatureVector
     * @param costs
     */
    public Instance(TObjectDoubleHashMap<String> generalFeatureVector, HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatureVector, TObjectDoubleHashMap<String> costs) {
        this.generalFeatureVector = generalFeatureVector;
        this.valueSpecificFeatureVector = valueSpecificFeatureVector;

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
                    this.costs.adjustValue(label, -this.minCost);
                }
                this.maxCost -= this.minCost;
            }
        }
    }

    /**
     *
     * @return
     */
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
        for (String feature : this.generalFeatureVector.keySet()) {
            if (!isFirst) {
                retString += " ";
            } else {
                isFirst = !isFirst;
            }
            retString += feature + ":" + this.generalFeatureVector.get(feature);
        }

        retString += "\t";

        isFirst = true;
        if (this.valueSpecificFeatureVector != null) {
            for (String feature : this.valueSpecificFeatureVector.keySet()) {
                if (!isFirst) {
                    retString += " ";
                } else {
                    isFirst = !isFirst;
                }
                retString += feature + ":" + this.valueSpecificFeatureVector.get(feature);
            }
        }
        return retString;
    }

    /**
     *
     * @return
     */
    public TObjectDoubleHashMap<String> getGeneralFeatureVector() {
        return generalFeatureVector;
    }

    /**
     *
     * @return
     */
    public HashMap<String, TObjectDoubleHashMap<String>> getValueSpecificFeatureVector() {
        return valueSpecificFeatureVector;
    }

    /**
     *
     * @param label
     * @return
     */
    public TObjectDoubleHashMap<String> getFeatureVector(String label) {
        TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
        featureVector.putAll(getGeneralFeatureVector());
        if (getValueSpecificFeatureVector() != null
                && getValueSpecificFeatureVector().containsKey(label)) {
            featureVector.putAll(getValueSpecificFeatureVector().get(label));
        }

        return featureVector;
    }

    /**
     *
     * @return
     */
    public TObjectDoubleHashMap<String> getCosts() {
        return costs;
    }

    /**
     *
     * @return
     */
    public Double getMaxCost() {
        return maxCost;
    }

    /**
     *
     * @return
     */
    public THashSet<String> getCorrectLabels() {
        return correctLabels;
    }
}
