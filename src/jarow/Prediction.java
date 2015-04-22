package jarow;

import java.util.ArrayList;
import java.util.HashMap;

public class Prediction {
    
    private HashMap<String, Double> label2score;
    private Double score;
    private String label;
    private ArrayList<ArrayList<Object>> featureValueWeights;
    private HashMap<String, Double> label2prob;
    private Double entropy;
    
    public Prediction() {        
        this.label2score = new HashMap<>();
        this.score = Double.NEGATIVE_INFINITY;
        this.label = null;
        this.featureValueWeights = new ArrayList<>();
        this.label2prob = new HashMap<>();
        this.entropy = 0.0;
    }

    public HashMap<String, Double> getLabel2score() {
        return label2score;
    }

    public void setLabel2score(HashMap<String, Double> label2score) {
        this.label2score = label2score;
    }

    public Double getScore() {
        return score;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public ArrayList<ArrayList<Object>> getFeatureValueWeights() {
        return featureValueWeights;
    }

    public void setFeatureValueWeights(ArrayList<ArrayList<Object>> featureValueWeights) {
        this.featureValueWeights = featureValueWeights;
    }

    public HashMap<String, Double> getLabel2prob() {
        return label2prob;
    }

    public void setLabel2prob(HashMap<String, Double> label2prob) {
        this.label2prob = label2prob;
    }

    public Double getEntropy() {
        return entropy;
    }

    public void setEntropy(Double entropy) {
        this.entropy = entropy;
    }
}
