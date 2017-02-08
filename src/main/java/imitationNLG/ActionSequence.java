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
import similarity_measures.Rouge;

/**
 *
 * @author Gerasimos Lampouras
 */
public class ActionSequence implements Comparable<ActionSequence> {
    private ArrayList<Action> sequence;

    /**
     *
     */
    public ActionSequence() {
        sequence = new ArrayList<>();
    }

    /**
     *
     * @param sequence
     */
    public ActionSequence(ArrayList<Action> sequence) {
        this.sequence = new ArrayList<>();
        sequence.stream().forEach((a) -> {
            this.sequence.add(new Action(a));
        });
    }

    /**
     *
     * @param sequence
     * @param cleanEnds
     */
    public ActionSequence(ArrayList<Action> sequence, boolean cleanEnds) {
        this.sequence = new ArrayList<>();
        if (cleanEnds) {
            sequence.stream().filter((a) -> (!a.getWord().equals(Action.TOKEN_START)
                    && !a.getWord().equals(Action.TOKEN_END))).forEach((a) -> {
                        this.sequence.add(new Action(a));
            });
        } else {
            sequence.stream().forEach((a) -> {
                this.sequence.add(new Action(a));
            });
        }
    }

    /**
     *
     * @param as
     */
    public ActionSequence(ActionSequence as) {
        this.sequence = new ArrayList<>();
        as.getSequence().stream().forEach((a) -> {
            this.sequence.add(new Action(a));
        });
    }

    /**
     *
     * @return
     */
    public ArrayList<Action> getSequence() {
        return sequence;
    }

    /**
     *
     * @param index
     * @param decision
     */
    public void modifyAndShortenSequence(int index, Action decision) {
        decision.setAttrValueTracking(this.sequence.get(index));
        
        this.sequence.set(index, decision);
        this.sequence = new ArrayList<Action>(this.sequence.subList(0, index + 1));
    }

    /**
     *
     * @return
     */
    final public String getWordSequenceToString() {
        String w = "";
        w = sequence.stream().filter((act) -> (!act.getWord().equals(Action.TOKEN_START)
                && !act.getWord().equals(Action.TOKEN_END))).map((act) -> act.getWord() + " ").reduce(w, String::concat);
        return w.trim();
    }

    /**
     *
     * @return
     */
    final public String getWordSequenceToNoPunctString() {
        String w = "";
        w = sequence.stream().filter((act) -> (!act.getWord().equals(Action.TOKEN_START)
                && !act.getWord().equals(Action.TOKEN_END)
                && !act.getAttribute().equals(Action.TOKEN_PUNCT))).map((act) -> act.getWord() + " ").reduce(w, String::concat);
        return w.trim();
    }

    /**
     *
     * @return
     */
    final public String getAttrSequenceToString() {
        String w = "";
        w = sequence.stream().map((act) -> act.getAttribute() + " ").reduce(w, String::concat);
        return w.trim();
    }

    /**
     *
     * @return
     */
    final public int getNoEndLength() {
        int l = 0;
        l = sequence.stream().filter((act) -> (!act.getWord().equals(Action.TOKEN_START)
                && !act.getWord().equals(Action.TOKEN_END))).map((_item) -> 1).reduce(l, Integer::sum);
        return l;
    }

    /**
     *
     * @return
     */
    final public int getNoPunctLength() {
        int l = 0;
        l = sequence.stream().filter((act) -> (!act.getWord().equals(Action.TOKEN_START)
                && !act.getWord().equals(Action.TOKEN_END)
                && !act.getAttribute().equals(Action.TOKEN_PUNCT))).map((_item) -> 1).reduce(l, Integer::sum);
        return l;
    }

    /**
     *
     * @return
     */
    final public ArrayList<String> getAttributeSequence() {
        ArrayList<String> seq = new ArrayList<>();
        sequence.stream().forEach((act) -> {
            seq.add(act.getAttribute());
        });
        return seq;
    }
    
    /**
     *
     * @param index
     * @return
     */
    final public ArrayList<String> getAttributeSubSequence(int index) {
        ArrayList<String> seq = new ArrayList<String>();
        sequence.subList(0, index).stream().forEach((act) -> {
            seq.add(act.getAttribute());
        });
        return seq;
    }
    
    /**
     *
     * @return
     */
    final public String getAttributeSequenceToString() {
        String a = "";
        a = sequence.stream().filter((act) -> (!act.getWord().equals(Action.TOKEN_START)
                && !act.getWord().equals(Action.TOKEN_END))).map((act) -> act.getAttribute() + " ").reduce(a, String::concat);
        return a.trim();
    }

    /**
     *
     * @return
     */
    @Override
    public int hashCode() {
        int hash = this.getWordSequenceToString().hashCode();
        return hash;
    }

    /**
     *
     * @param obj
     * @return
     */
    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final ActionSequence other = (ActionSequence) obj;
        return this.getSequence().toString().equals(other.getSequence().toString());
    }

    /**
     *
     * @param o
     * @return
     */
    @Override
    public int compareTo(ActionSequence o) {
        if (o == null) {
            return -1;
        }
        if (!getClass().equals(o.getClass())) {
            return -1;
        }
        final ActionSequence other = o;
        if (this.getSequence().size() < other.getSequence().size()) {
            return -1;
        }
        if (this.getSequence().size() > other.getSequence().size()) {
            return 1;
        }
        return 0;
    }

    /**
     *
     */
    public static String metric = "B";

    /**
     *
     * @param s1
     * @param s2s
     * @param coverageError
     * @return
     */
    public static double getCostMetric(String s1, ArrayList<String> s2s, Double coverageError) {
        switch (metric) {
            case "B":
                return getBLEU(s1, s2s);
            case "R":
                return getROUGE(s1, s2s);
            case "BC":
                if (coverageError == -1.0) {
                    return getBLEU(s1, s2s);
                }
                return (getBLEU(s1, s2s) + coverageError) / 2.0;
            case "RC":
                if (coverageError == -1.0) {
                    return getROUGE(s1, s2s);
                }
                return (getROUGE(s1, s2s) + coverageError) / 2.0;
            case "BRC":
                if (coverageError == -1.0) {
                    return (getBLEU(s1, s2s) + getROUGE(s1, s2s)) / 2.0;
                }
                return (getBLEU(s1, s2s) + getROUGE(s1, s2s) + coverageError) / 3.0;
            case "BR":
                return (getBLEU(s1, s2s) + getROUGE(s1, s2s)) / 2.0;
            default:
                break;
        }
        return getBLEU(s1, s2s);
    }

    /**
     *
     * @param s1
     * @param s2s
     * @return
     */
    public static double getBLEU(String s1, ArrayList<String> s2s) {
        return 1.0 - BLEUMetric.computeLocalSmoothScore(s1, s2s, 4);
    }

    /**
     *
     * @param s1
     * @param s2s
     * @return
     */
    public static double getROUGE(String s1, ArrayList<String> s2s) {
        double maxRouge = 0.0;
        for (String s2 : s2s) {
            double rouge = Rouge.ROUGE_N(s1, s2, 4);
            if (rouge > maxRouge) {
                maxRouge = rouge;
            }
        }
        return 1.0 - maxRouge;
    }

    /**
     *
     * @return
     */
    @Override
    public String toString() {
        return "ActionSequence{" + "sequence=" + sequence + '}';
    }
}