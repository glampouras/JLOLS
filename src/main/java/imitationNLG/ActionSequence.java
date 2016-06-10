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
package imitationNLG;

import edu.stanford.nlp.mt.metrics.BLEUMetric;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

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

    public ActionSequence(ArrayList<Action> sequence, double cost, boolean cleanEnds) {
        this.sequence = new ArrayList<>();
        if (cleanEnds) {
            for (Action a : sequence) {
                if (!a.getWord().equals(Bagel.TOKEN_START)
                        && !a.getWord().equals(Bagel.TOKEN_END)) {
                    this.sequence.add(new Action(a.getWord(), a.getAttribute()));
                }
            }
        } else {
            for (Action a : sequence) {
                this.sequence.add(new Action(a.getWord(), a.getAttribute()));
            }
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
        cleanRefs.add(ref.getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim());
        this.cost = getBLEU(getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim(), cleanRefs);
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
            cleanRefs.add(ref.getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim());
        }
        min = getBLEU(getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim(), cleanRefs);

        this.cost = min;
    }

    public ArrayList<Action> getSequence() {
        return sequence;
    }

    public void modifyAndShortenSequence(int index, Action decision) {
        this.sequence.set(index, decision);
        this.sequence = new ArrayList(this.sequence.subList(0, index + 1));
    }

    public double getCost() {
        return cost;
    }

    public void recalculateCost(ActionSequence ref) {
        //this.cost = Levenshtein.getNormDistance(getWordSequenceToString(), ref.getWordSequenceToString());
        ArrayList<String> cleanRefs = new ArrayList<String>();
        cleanRefs.add(ref.getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim());
        this.cost = getBLEU(getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim(), cleanRefs);
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
            cleanRefs.add(ref.getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim());
        }
        min = getBLEU(getWordSequenceToNoPunctString().toLowerCase().replaceAll("\\p{Punct}|\\d", "").trim().replaceAll("  ", " "), cleanRefs);

        this.cost = min;
    }

    final public String getWordSequenceToString() {
        String w = "";
        for (Action act : sequence) {
            if (!act.getWord().equals(Bagel.TOKEN_START)
                    && !act.getWord().equals(RoboCup.TOKEN_END)) {
                w += act.getWord() + " ";
            }
        }
        return w.trim();
    }

    final public String getWordSequenceToNoPunctString() {
        String w = "";
        for (Action act : sequence) {
            if (!act.getWord().equals(Bagel.TOKEN_START)
                    && !act.getWord().equals(RoboCup.TOKEN_END)
                    && !act.getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                w += act.getWord() + " ";
            }
        }
        return w.trim();
    }

    final public int getNoEndLength() {
        int l = 0;
        for (Action act : sequence) {
            if (!act.getWord().equals(Bagel.TOKEN_START)
                    && !act.getWord().equals(RoboCup.TOKEN_END)) {
                l++;
            }
        }
        return l;
    }

    final public int getNoPunctLength() {
        int l = 0;
        for (Action act : sequence) {
            if (!act.getWord().equals(Bagel.TOKEN_START)
                    && !act.getWord().equals(Bagel.TOKEN_END)
                    && !act.getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                l++;
            }
        }
        return l;
    }

    final static public int getNoPunctLength(ArrayList<Action> list) {
        int l = 0;
        for (Action act : list) {
            if (!act.getWord().equals(Bagel.TOKEN_START)
                    && !act.getWord().equals(RoboCup.TOKEN_END)
                    && !act.getAttribute().equals(Bagel.TOKEN_PUNCT)) {
                l++;
            }
        }
        return l;
    }

    final public String getAttributeSequenceToString() {
        String a = "";
        for (Action act : sequence) {
            if (!act.getWord().equals(Bagel.TOKEN_START)
                    && !act.getWord().equals(RoboCup.TOKEN_END)) {
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
            cleanRefs.add(ref.toLowerCase().replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim());
        }
        minScore = getBLEU(sequenceString.toLowerCase().replaceAll("\\p{Punct}|\\d", "").replaceAll("  ", " ").trim(), cleanRefs);

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

    public static int getHammingDistance(String s1, ArrayList<String> s2s) {
        int min = Integer.MAX_VALUE;
        for (String s2 : s2s) {
            int dis = getHammingDistance(s1, s2);
            if (dis < min) {
                min = dis;
            }
        }
        return min;
    }

    public static int getHammingDistance(String s1, String s2) {
        String[] tokens1 = s1.split(" ");
        String[] tokens2 = s2.split(" ");

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

        ArrayList<String> tokens1ListBigrams = new ArrayList<String>();
        for (int i = 0; i < tokens1.length - 1; i++) {
            if (!tokens1[i].trim().isEmpty()
                    && !tokens1[i + 1].trim().isEmpty()) {
                tokens1ListBigrams.add(tokens1[i].trim().toLowerCase() + " " + tokens1[i + 1].trim().toLowerCase());
            }
        }
        ArrayList<String> tokens2ListBigrams = new ArrayList<String>();
        for (int j = 0; j < tokens2.length - 1; j++) {
            if (!tokens2[j].trim().isEmpty()
                    && !tokens2[j + 1].trim().isEmpty()) {
                tokens2ListBigrams.add(tokens2[j].trim().toLowerCase() + " " + tokens2[j + 1].trim().toLowerCase());
            }
        }

        HashMap<Integer, HashSet<ArrayList<Integer>>> matchesBigrams = new HashMap<>();
        for (int i = 0; i < tokens1ListBigrams.size(); i++) {
            for (int j = 0; j < tokens2ListBigrams.size(); j++) {
                if (tokens1ListBigrams.get(i).equals(tokens2ListBigrams.get(j))) {
                    ArrayList<Integer> match = new ArrayList<>();
                    match.add(i);
                    match.add(j);

                    int distance = Math.abs(i - j);
                    if (!matchesBigrams.containsKey(distance)) {
                        matchesBigrams.put(distance, new HashSet<ArrayList<Integer>>());
                    }
                    matchesBigrams.get(distance).add(match);
                }
            }
        }
        ArrayList<Integer> valuesBigrams = new ArrayList<>(matchesBigrams.keySet());
        Collections.sort(valuesBigrams);

        HashSet<Integer> usedIsBigrams = new HashSet<>();
        HashSet<Integer> usedJsBigrams = new HashSet<>();
        for (Integer value : valuesBigrams) {
            for (ArrayList<Integer> match : matchesBigrams.get(value)) {
                if (!usedIsBigrams.contains(match.get(0)) && !usedJsBigrams.contains(match.get(1))) {
                    usedIsBigrams.add(match.get(0));
                    usedJsBigrams.add(match.get(1));

                    totalDistance += value;
                }
            }
        }

        for (int i = 0; i < tokens1ListBigrams.size(); i++) {
            if (!usedIsBigrams.contains(i)) {
                totalDistance += i + 1;
            }
        }
        for (int j = 0; j < tokens2ListBigrams.size(); j++) {
            if (!usedJsBigrams.contains(j)) {
                totalDistance += j + 1;
            }
        }

        ArrayList<String> tokens1List3grams = new ArrayList<String>();
        for (int i = 0; i < tokens1.length - 2; i++) {
            if (!tokens1[i].trim().isEmpty()
                    && !tokens1[i + 1].trim().isEmpty()
                    && !tokens1[i + 2].trim().isEmpty()) {
                tokens1List3grams.add(tokens1[i].trim().toLowerCase() + " " + tokens1[i + 1].trim().toLowerCase() + " " + tokens1[i + 2].trim().toLowerCase());
            }
        }
        ArrayList<String> tokens2List3grams = new ArrayList<String>();
        for (int j = 0; j < tokens2.length - 2; j++) {
            if (!tokens2[j].trim().isEmpty()
                    && !tokens2[j + 1].trim().isEmpty()
                    && !tokens2[j + 2].trim().isEmpty()) {
                tokens2List3grams.add(tokens2[j].trim().toLowerCase() + " " + tokens2[j + 1].trim().toLowerCase() + " " + tokens2[j + 2].trim().toLowerCase());
            }
        }

        HashMap<Integer, HashSet<ArrayList<Integer>>> matches3grams = new HashMap<>();
        for (int i = 0; i < tokens1List3grams.size(); i++) {
            for (int j = 0; j < tokens2List3grams.size(); j++) {
                if (tokens1List3grams.get(i).equals(tokens2List3grams.get(j))) {
                    ArrayList<Integer> match = new ArrayList<>();
                    match.add(i);
                    match.add(j);

                    int distance = Math.abs(i - j);
                    if (!matches3grams.containsKey(distance)) {
                        matches3grams.put(distance, new HashSet<ArrayList<Integer>>());
                    }
                    matches3grams.get(distance).add(match);
                }
            }
        }
        ArrayList<Integer> values3grams = new ArrayList<>(matches3grams.keySet());
        Collections.sort(values3grams);

        HashSet<Integer> usedIs3grams = new HashSet<>();
        HashSet<Integer> usedJs3grams = new HashSet<>();
        for (Integer value : values3grams) {
            for (ArrayList<Integer> match : matches3grams.get(value)) {
                if (!usedIs3grams.contains(match.get(0)) && !usedJs3grams.contains(match.get(1))) {
                    usedIs3grams.add(match.get(0));
                    usedJs3grams.add(match.get(1));

                    totalDistance += value;
                }
            }
        }

        for (int i = 0; i < tokens1List3grams.size(); i++) {
            if (!usedIs3grams.contains(i)) {
                totalDistance += i + 1;
            }
        }
        for (int j = 0; j < tokens2List3grams.size(); j++) {
            if (!usedJs3grams.contains(j)) {
                totalDistance += j + 1;
            }
        }

        return totalDistance;
    }

    public static int getHammingDistanceRoboCup(String s1, String s2) {
        String[] tokens1 = s1.replaceAll("\\p{Punct}|\\d", "").split(" ");
        String[] tokens2 = s2.replaceAll("\\p{Punct}|\\d", "").split(" ");

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

        ArrayList<String> tokens1ListBigrams = new ArrayList<String>();
        for (int i = 0; i < tokens1.length - 1; i++) {
            if (!tokens1[i].trim().isEmpty()
                    && !tokens1[i + 1].trim().isEmpty()) {
                tokens1ListBigrams.add(tokens1[i].trim().toLowerCase() + " " + tokens1[i + 1].trim().toLowerCase());
            }
        }
        ArrayList<String> tokens2ListBigrams = new ArrayList<String>();
        for (int j = 0; j < tokens2.length - 1; j++) {
            if (!tokens2[j].trim().isEmpty()
                    && !tokens2[j + 1].trim().isEmpty()) {
                tokens2ListBigrams.add(tokens2[j].trim().toLowerCase() + " " + tokens2[j + 1].trim().toLowerCase());
            }
        }

        HashMap<Integer, HashSet<ArrayList<Integer>>> matchesBigrams = new HashMap<>();
        for (int i = 0; i < tokens1ListBigrams.size(); i++) {
            for (int j = 0; j < tokens2ListBigrams.size(); j++) {
                if (tokens1ListBigrams.get(i).equals(tokens2ListBigrams.get(j))) {
                    ArrayList<Integer> match = new ArrayList<>();
                    match.add(i);
                    match.add(j);

                    int distance = Math.abs(i - j);
                    if (!matchesBigrams.containsKey(distance)) {
                        matchesBigrams.put(distance, new HashSet<ArrayList<Integer>>());
                    }
                    matchesBigrams.get(distance).add(match);
                }
            }
        }
        ArrayList<Integer> valuesBigrams = new ArrayList<>(matchesBigrams.keySet());
        Collections.sort(valuesBigrams);

        HashSet<Integer> usedIsBigrams = new HashSet<>();
        HashSet<Integer> usedJsBigrams = new HashSet<>();
        for (Integer value : valuesBigrams) {
            for (ArrayList<Integer> match : matchesBigrams.get(value)) {
                if (!usedIsBigrams.contains(match.get(0)) && !usedJsBigrams.contains(match.get(1))) {
                    usedIsBigrams.add(match.get(0));
                    usedJsBigrams.add(match.get(1));

                    totalDistance += value;
                }
            }
        }

        for (int i = 0; i < tokens1ListBigrams.size(); i++) {
            if (!usedIsBigrams.contains(i)) {
                totalDistance += i + 1;
            }
        }
        for (int j = 0; j < tokens2ListBigrams.size(); j++) {
            if (!usedJsBigrams.contains(j)) {
                totalDistance += j + 1;
            }
        }

        ArrayList<String> tokens1List3grams = new ArrayList<String>();
        for (int i = 0; i < tokens1.length - 2; i++) {
            if (!tokens1[i].trim().isEmpty()
                    && !tokens1[i + 1].trim().isEmpty()
                    && !tokens1[i + 2].trim().isEmpty()) {
                tokens1List3grams.add(tokens1[i].trim().toLowerCase() + " " + tokens1[i + 1].trim().toLowerCase() + " " + tokens1[i + 2].trim().toLowerCase());
            }
        }
        ArrayList<String> tokens2List3grams = new ArrayList<String>();
        for (int j = 0; j < tokens2.length - 2; j++) {
            if (!tokens2[j].trim().isEmpty()
                    && !tokens2[j + 1].trim().isEmpty()
                    && !tokens2[j + 2].trim().isEmpty()) {
                tokens2List3grams.add(tokens2[j].trim().toLowerCase() + " " + tokens2[j + 1].trim().toLowerCase() + " " + tokens2[j + 2].trim().toLowerCase());
            }
        }

        HashMap<Integer, HashSet<ArrayList<Integer>>> matches3grams = new HashMap<>();
        for (int i = 0; i < tokens1List3grams.size(); i++) {
            for (int j = 0; j < tokens2List3grams.size(); j++) {
                if (tokens1List3grams.get(i).equals(tokens2List3grams.get(j))) {
                    ArrayList<Integer> match = new ArrayList<>();
                    match.add(i);
                    match.add(j);

                    int distance = Math.abs(i - j);
                    if (!matches3grams.containsKey(distance)) {
                        matches3grams.put(distance, new HashSet<ArrayList<Integer>>());
                    }
                    matches3grams.get(distance).add(match);
                }
            }
        }
        ArrayList<Integer> values3grams = new ArrayList<>(matches3grams.keySet());
        Collections.sort(values3grams);

        HashSet<Integer> usedIs3grams = new HashSet<>();
        HashSet<Integer> usedJs3grams = new HashSet<>();
        for (Integer value : values3grams) {
            for (ArrayList<Integer> match : matches3grams.get(value)) {
                if (!usedIs3grams.contains(match.get(0)) && !usedJs3grams.contains(match.get(1))) {
                    usedIs3grams.add(match.get(0));
                    usedJs3grams.add(match.get(1));

                    totalDistance += value;
                }
            }
        }

        for (int i = 0; i < tokens1List3grams.size(); i++) {
            if (!usedIs3grams.contains(i)) {
                totalDistance += i + 1;
            }
        }
        for (int j = 0; j < tokens2List3grams.size(); j++) {
            if (!usedJs3grams.contains(j)) {
                totalDistance += j + 1;
            }
        }

        return totalDistance;
    }

    public String toString() {
        return "ActionSequence{" + "sequence=" + sequence + ", cost=" + cost + '}';
    }
}
