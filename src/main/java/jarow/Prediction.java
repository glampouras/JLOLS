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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.logging.Logger;

/**
 *
 * @author Gerasimos Lampouras
 */
public class Prediction {
    
    private TObjectDoubleHashMap<String> label2score;
    private Double score;
    private String label;
    private ArrayList<ArrayList<Object>> featureValueWeights;
    private TObjectDoubleHashMap<String> label2prob;
    private Double entropy;
    
    private HashSet<String> mostInfluencialFeatures;
    
    /**
     *
     */
    public Prediction() {        
        this.label2score = new TObjectDoubleHashMap<>();
        this.score = Double.NEGATIVE_INFINITY;
        this.label = null;
        this.featureValueWeights = new ArrayList<>();
        this.label2prob = new TObjectDoubleHashMap<>();
        this.entropy = 0.0;
    }

    /**
     *
     * @return
     */
    public TObjectDoubleHashMap<String> getLabel2Score() {
        return label2score;
    }

    /**
     *
     * @param label2score
     */
    public void setLabel2score(TObjectDoubleHashMap<String> label2score) {
        this.label2score = label2score;
    }

    /**
     *
     * @return
     */
    public HashSet<String> getMostInfluencialFeatures() {
        return mostInfluencialFeatures;
    }

    /**
     *
     * @param mostInfluencialFeatures
     */
    public void setMostInfluencialFeatures(HashSet<String> mostInfluencialFeatures) {
        this.mostInfluencialFeatures = mostInfluencialFeatures;
    }

    /**
     *
     * @return
     */
    public Double getScore() {
        return score;
    }

    /**
     *
     * @param score
     */
    public void setScore(Double score) {
        this.score = score;
    }

    /**
     *
     * @return
     */
    public String getLabel() {
        return label;
    }

    /**
     *
     * @param label
     */
    public void setLabel(String label) {
        this.label = label;
    }

    /**
     *
     * @return
     */
    public ArrayList<ArrayList<Object>> getFeatureValueWeights() {
        return featureValueWeights;
    }

    /**
     *
     * @param featureValueWeights
     */
    public void setFeatureValueWeights(ArrayList<ArrayList<Object>> featureValueWeights) {
        this.featureValueWeights = featureValueWeights;
    }

    /**
     *
     * @return
     */
    public TObjectDoubleHashMap<String> getLabel2prob() {
        return label2prob;
    }

    /**
     *
     * @param label2prob
     */
    public void setLabel2prob(TObjectDoubleHashMap<String> label2prob) {
        this.label2prob = label2prob;
    }

    /**
     *
     * @return
     */
    public Double getEntropy() {
        return entropy;
    }

    /**
     *
     * @param entropy
     */
    public void setEntropy(Double entropy) {
        this.entropy = entropy;
    }
    private static final Logger LOG = Logger.getLogger(Prediction.class.getName());
}
