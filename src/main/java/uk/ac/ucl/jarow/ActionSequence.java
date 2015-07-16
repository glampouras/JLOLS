package uk.ac.ucl.jarow;

import java.util.ArrayList;
import similarity_measures.Levenshtein;

public class ActionSequence {
    private ArrayList<Action> sequence;
    private double cost = 0.0;
    
    public ActionSequence() {
        sequence = new ArrayList<>();
    }
    
    public ActionSequence(ArrayList<Action> sequence, double cost) {
        this.sequence = new ArrayList<>();
        for (Action a : sequence) {
            this.sequence.add(new Action(a.getDecision()));
        }
        this.cost = cost;
    }
    
    public ActionSequence(ActionSequence as) {
        this.sequence = new ArrayList<>();
        for (Action a : as.getSequence()) {
            this.sequence.add(new Action(a.getDecision()));
        }
        this.cost = as.getCost();
    }
    
    public ActionSequence(ArrayList<Action> sequence, ArrayList<String> unbiasedRefList) {
        this.sequence = new ArrayList<>(sequence);
        this.cost = calculateDistance(getSequenceToString(), unbiasedRefList);
    }    

    public ArrayList<Action> getSequence() {
        return sequence;
    }
    
    public void modifyAndShortenSequence(int index, String decision) {        
        this.sequence.get(index).setDecision(decision);
        this.sequence = new ArrayList(this.sequence.subList(0, index + 1));
    }

    public double getCost() {
        return cost;
    }
    
    public void recalculateCost(ArrayList<String> unbiasedRefList) {
        this.cost = calculateDistance(getSequenceToString(), unbiasedRefList);
    }
    
    final public String getSequenceToString() {
        String s = "";
        for (Action act : sequence) {
            if (!act.getDecision().equals(RoboCup.TOKEN_END)) {
                s += act.getDecision() + " ";
            }
        }
        return s.trim();
    }

    @Override
    public int hashCode() {
        int hash = this.getSequenceToString().hashCode();
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final ActionSequence other = (ActionSequence) obj;
        if (!this.getSequenceToString().equals(other.getSequenceToString())) {
            return false;
        }
        return true;
    }
    
    public static double calculateDistance(String sequenceString, ArrayList<String> unbiasedRefList) {
        double maxScore = 0.0;
        for (String ref : unbiasedRefList) {
            double score = Levenshtein.getDistance(sequenceString, ref);
            if (score > maxScore) {
                maxScore = score;
            }
        }
        return maxScore;
    }
}