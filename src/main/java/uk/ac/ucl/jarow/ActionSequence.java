package uk.ac.ucl.jarow;

import edu.stanford.nlp.mt.metrics.BLEUMetric;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;
import similarity_measures.Levenshtein;

public class ActionSequence implements Comparable {

    private ArrayList<Action> sequence;
    private double cost = 0.0;

    public ActionSequence() {
        sequence = new ArrayList<>();
    }

    public ActionSequence(ArrayList<Action> sequence, double cost) {
        this.sequence = new ArrayList<>();
        for (Action a : sequence) {
            this.sequence.add(new Action(a.getWord(), a.getAttribute()));
        }
        this.cost = cost;
    }

    public ActionSequence(ActionSequence as) {
        this.sequence = new ArrayList<>();
        for (Action a : as.getSequence()) {
            this.sequence.add(new Action(a.getWord(), a.getAttribute()));
        }
        this.cost = as.getCost();
    }

    public ActionSequence(ArrayList<Action> sequence, ActionSequence ref) {
        this.sequence = new ArrayList<>(sequence);
        //this.cost = Levenshtein.getNormDistance(getWordSequenceToString(), ref.getWordSequenceToString());
        ArrayList<String> cleanRefs = new ArrayList<String>();
        cleanRefs.add(ref.getWordSequenceToString().toLowerCase().replaceAll("\\p{Punct}|\\d","").replaceAll("  "," ").trim());
        this.cost = getBLEU(getWordSequenceToString().toLowerCase().replaceAll("\\p{Punct}|\\d","").replaceAll("  "," ").trim(), cleanRefs);
    }

    public ActionSequence(ArrayList<Action> sequence, HashSet<ActionSequence> refs) {
        this.sequence = new ArrayList<>(sequence);

        double min = Double.POSITIVE_INFINITY;
        /*for (ActionSequence ref : refs) {
            //double c = Levenshtein.getNormDistance(getWordSequenceToString(), ref.getWordSequenceToString());
            double c = getBLEU(getWordSequenceToString(), ref.getWordSequenceToString());
            if (c < min) {
                min = c;
            }
        System.out.println(getWordSequenceToString());
        System.out.println(ref.getWordSequenceToString());
        System.out.println("= " + c);
        }*/
        
        ArrayList<String> cleanRefs = new ArrayList<String>();
        for (ActionSequence ref : refs) {
            cleanRefs.add(ref.getWordSequenceToString().toLowerCase().replaceAll("\\p{Punct}|\\d","").replaceAll("  "," ").trim());
        }
        min = getBLEU(getWordSequenceToString().toLowerCase().replaceAll("\\p{Punct}|\\d","").replaceAll("  "," ").trim(), cleanRefs);

        this.cost = min;
    }

    public ArrayList<Action> getSequence() {
        return sequence;
    }

    public void modifyAndShortenSequence(int index, String decision) {
        this.sequence.get(index).setWord(decision);
        this.sequence = new ArrayList(this.sequence.subList(0, index + 1));
    }

    public double getCost() {
        return cost;
    }

    public void recalculateCost(ActionSequence ref) {
        //this.cost = Levenshtein.getNormDistance(getWordSequenceToString(), ref.getWordSequenceToString());
        ArrayList<String> cleanRefs = new ArrayList<String>();
        cleanRefs.add(ref.getWordSequenceToString().toLowerCase().replaceAll("\\p{Punct}|\\d","").replaceAll("  "," ").trim());
        this.cost = getBLEU(getWordSequenceToString().toLowerCase().replaceAll("\\p{Punct}|\\d","").replaceAll("  "," ").trim(), cleanRefs);
    }

    public void recalculateCost(HashSet<ActionSequence> refs) {
        double min = Double.POSITIVE_INFINITY;
        /*for (ActionSequence ref : refs) {
            //double c = Levenshtein.getNormDistance(getWordSequenceToString().toLowerCase(), ref.getWordSequenceToString().toLowerCase());
            double c = getBLEU(getWordSequenceToString(), ref.getWordSequenceToString());
            if (c < min) {
                min = c;
            }
        }*/
        ArrayList<String> cleanRefs = new ArrayList<String>();
        for (ActionSequence ref : refs) {
            cleanRefs.add(ref.getWordSequenceToString().toLowerCase().replaceAll("\\p{Punct}|\\d","").replaceAll("  "," ").trim());
        }
        min = getBLEU(getWordSequenceToString().toLowerCase().replaceAll("\\p{Punct}|\\d","").trim().replaceAll("  "," "), cleanRefs);

        this.cost = min;
    }

    final public String getWordSequenceToString() {
        String w = "";
        for (Action act : sequence) {
            if (!act.getWord().equals(RoboCup.TOKEN_END)) {
                w += act.getWord() + " ";
            }
        }
        return w.trim();
    }

    final public String getAttributeSequenceToString() {
        String a = "";
        for (Action act : sequence) {
            if (!act.getWord().equals(RoboCup.TOKEN_END)) {
                a += act.getAttribute() + " ";
            }
        }
        return a.trim();
    }

    @Override
    public int hashCode() {
        int hash = this.getWordSequenceToString().hashCode();
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
        /*for (String ref : unbiasedRefList) {
            //double score = Levenshtein.getNormDistance(sequenceString, ref);
            double score = getBLEU(sequenceString, ref);
            if (score < minScore) {
                minRef = ref;
                minScore = score;
            }
        }*/
        ArrayList<String> cleanRefs = new ArrayList<String>();
        for (String ref : unbiasedRefList) {
            cleanRefs.add(ref.toLowerCase().replaceAll("\\p{Punct}|\\d","").replaceAll("  "," ").trim());
        }
        minScore = getBLEU(sequenceString.toLowerCase().replaceAll("\\p{Punct}|\\d","").replaceAll("  "," ").trim(), cleanRefs);
        
        //System.out.println("MINREF " + minRef + " > " + minScore);
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

    public static double getBLEU(String s1, ArrayList<String> s2s) {        
        return 1.0 - BLEUMetric.computeLocalSmoothScore(s1, s2s, 4);
    }
        
    public static int getHammingDistance(String s1, String s2) {
        String[] tokens1 = s1.replaceAll("\\p{Punct}|\\d","").split(" ");
        String[] tokens2 = s2.replaceAll("\\p{Punct}|\\d","").split(" ");

        ArrayList<String> tokens1List = new ArrayList<String>();
        for (int i = 0; i < tokens1.length; i++) {
            if (!tokens1[i].trim().isEmpty()) {
                tokens1List.add(tokens1[i].trim().toLowerCase());
            }
        }
        ArrayList<String> tokens2List = new ArrayList<String>();
        for (int j = 0; j < tokens2.length; j++) {
            if (!tokens2[j].trim().isEmpty()) {
                tokens2List.add(tokens2[j].trim().toLowerCase());
            }
        }

        HashMap<Integer, HashSet<ArrayList<Integer>>> matches = new HashMap<>();
        for (int i = 0; i < tokens1List.size(); i++) {
            for (int j = 0; j < tokens2List.size(); j++) {
                if (tokens1List.get(i).equals(tokens2List.get(j))) {
                    ArrayList<Integer> match = new ArrayList<>();
                    match.add(i);
                    match.add(j);

                    int distance = Math.abs(i - j);
                    if (!matches.containsKey(distance)) {
                        matches.put(distance, new HashSet<ArrayList<Integer>>());
                    }
                    matches.get(distance).add(match);
                }
            }
        }
        ArrayList<Integer> values = new ArrayList<>(matches.keySet());
        Collections.sort(values);

        HashSet<Integer> usedIs = new HashSet<>();
        HashSet<Integer> usedJs = new HashSet<>();
        Integer totalDistance = 0;
        for (Integer value : values) {
            for (ArrayList<Integer> match : matches.get(value)) {
                if (!usedIs.contains(match.get(0)) && !usedJs.contains(match.get(1))) {
                    usedIs.add(match.get(0));
                    usedJs.add(match.get(1));

                    totalDistance += value;
                }
            }
        }

        for (int i = 0; i < tokens1List.size(); i++) {
            if (!usedIs.contains(i)) {
                totalDistance += i + 1;
            }
        }
        for (int j = 0; j < tokens2List.size(); j++) {
            if (!usedJs.contains(j)) {
                totalDistance += j + 1;
            }
        }
        return totalDistance;
    }
}
