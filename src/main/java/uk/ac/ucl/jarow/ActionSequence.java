package uk.ac.ucl.jarow;

import java.util.ArrayList;
import java.util.Objects;
import similarity_measures.Levenshtein;

public class ActionSequence implements Comparable{
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
    
    public ActionSequence(ArrayList<Action> sequence, ActionSequence ref) {
        this.sequence = new ArrayList<>(sequence);
        this.cost = Levenshtein.getNormDistance(getSequenceToString(), ref.getSequenceToString());
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
    
    public void recalculateCost(ActionSequence ref) {
        this.cost = Levenshtein.getNormDistance(getSequenceToString(), ref.getSequenceToString());
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
        if (!this.getSequence().toString().equals(other.getSequence().toString())) {
            return false;
        }
        return true;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }
    
    public static double calculateDistance(String sequenceString, ArrayList<String> unbiasedRefList) {
        double minScore = 1000000000000.0;
        String minRef = "";
        for (String ref : unbiasedRefList) {
            double score = Levenshtein.getNormDistance(sequenceString, ref);
            if (score < minScore) {
                minRef = ref;
                minScore = score;
            }
        }
        System.out.println("MINREF " + minRef + " > " + minScore);
        return minScore;
    }

    public int compareTo(Object o) {
        if (o == null) {
            return -1;
        }
        if (!getClass().equals(o.getClass())) {
            return -1;
        }
        final ActionSequence other = (ActionSequence) o;
        if (this.getSequence().size() < other.getSequence().size()) {
            return -1;
        }
        if (this.getSequence().size() > other.getSequence().size()) {
            return 1;
        }
        return 0;
    }
}