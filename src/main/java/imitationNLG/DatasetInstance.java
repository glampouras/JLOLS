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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 *
 * @author Gerasimos Lampouras
 */
public class DatasetInstance implements Serializable, Comparable<DatasetInstance> {

    private MeaningRepresentation MR;
    ArrayList<String> trainMentionedValueSequence;
    ArrayList<String> trainMentionedAttributeSequence;
    ArrayList<Action> trainRealization;
    ArrayList<Action> trainAttrRealization;
    HashMap<ArrayList<Action>, ArrayList<String>> evalMentionedValueSequences;
    HashMap<ArrayList<Action>, ArrayList<String>> evalMentionedAttributeSequences;
    HashSet<ArrayList<Action>> evalRealizations;
    HashSet<ArrayList<Action>> evalAttrRealizations;
    HashMap<ArrayList<Action>, String> alignedSubRealization;
    HashMap<ArrayList<Action>, String> realizationCorrectAction;
    HashSet<String> realizationAltCorrectActions;

    String trainReference = "";
    HashSet<String> evalReferences = new HashSet<String>();

    /**
     *
     * @param MR
     * @param mentionedValueSequence
     * @param mentionedAttributeSequence
     * @param realization
     */
    public DatasetInstance(MeaningRepresentation MR, ArrayList<String> mentionedValueSequence, ArrayList<String> mentionedAttributeSequence, ArrayList<Action> realization) {
        this.MR = MR;

        this.trainMentionedValueSequence = mentionedValueSequence;
        this.trainMentionedAttributeSequence = mentionedAttributeSequence;
        this.trainRealization = realization;

        this.evalMentionedValueSequences = new HashMap<>();
        this.evalMentionedValueSequences.put(realization, mentionedValueSequence);

        this.evalMentionedAttributeSequences = new HashMap<>();
        this.evalMentionedAttributeSequences.put(realization, mentionedAttributeSequence);

        this.evalRealizations = new HashSet<>();
        this.evalRealizations.add(realization);

        this.alignedSubRealization = new HashMap<>();
        this.realizationCorrectAction = new HashMap<>();

        this.realizationAltCorrectActions = new HashSet<>();
    }

    /**
     *
     * @param di
     */
    public DatasetInstance(DatasetInstance di) {
        this.MR = di.getMeaningRepresentation();

        this.trainMentionedValueSequence = di.trainMentionedValueSequence;
        this.trainMentionedAttributeSequence = di.trainMentionedAttributeSequence;
        this.trainRealization = di.getTrainRealization();

        this.evalMentionedValueSequences = new HashMap<>();
        for (ArrayList<Action> realization : di.getEvalMentionedValueSequences().keySet()) {
            this.evalMentionedValueSequences.put(new ArrayList<Action>(realization), new ArrayList<String>(di.getEvalMentionedValueSequences().get(realization)));
        }

        this.evalMentionedAttributeSequences = new HashMap<>();
        for (ArrayList<Action> realization : di.getEvalMentionedAttributeSequences().keySet()) {
            this.evalMentionedAttributeSequences.put(new ArrayList<Action>(realization), new ArrayList<String>(di.getEvalMentionedAttributeSequences().get(realization)));
        }

        this.evalRealizations = new HashSet<>();
        for (ArrayList<Action> realization : di.getEvalRealizations()) {
            this.evalRealizations.add(new ArrayList<Action>(realization));
        }

        this.alignedSubRealization = new HashMap<>();
        this.realizationCorrectAction = new HashMap<>();

        this.realizationAltCorrectActions = new HashSet<String>();

        this.trainReference = di.getTrainReference();
        this.evalReferences.addAll(di.getEvalReferences());
    }

    /**
     *
     * @param mentionedValueSequences
     * @param mentionedAttributeSequences
     * @param realizations
     */
    public void mergeDatasetInstance(HashMap<ArrayList<Action>, ArrayList<String>> mentionedValueSequences, HashMap<ArrayList<Action>, ArrayList<String>> mentionedAttributeSequences, HashSet<ArrayList<Action>> realizations) {
        this.evalRealizations.addAll(realizations);
        this.evalMentionedValueSequences.putAll(mentionedValueSequences);
        this.evalMentionedAttributeSequences.putAll(mentionedAttributeSequences);
    }

    /**
     *
     * @param mentionedValueSequence
     * @param mentionedAttributeSequence
     * @param realization
     */
    public void mergeDatasetInstance(ArrayList<String> mentionedValueSequence, ArrayList<String> mentionedAttributeSequence, ArrayList<Action> realization) {
        this.evalRealizations.add(realization);
        this.evalMentionedValueSequences.put(realization, mentionedValueSequence);
        this.evalMentionedAttributeSequences.put(realization, mentionedAttributeSequence);
    }

    /**
     *
     * @return
     */
    public MeaningRepresentation getMeaningRepresentation() {
        return MR;
    }

    /**
     *
     * @return
     */
    public HashMap<ArrayList<Action>, ArrayList<String>> getEvalMentionedValueSequences() {
        return evalMentionedValueSequences;
    }

    /**
     *
     * @return
     */
    public HashMap<ArrayList<Action>, ArrayList<String>> getEvalMentionedAttributeSequences() {
        return evalMentionedAttributeSequences;
    }

    /**
     *
     * @return
     */
    public HashSet<ArrayList<Action>> getEvalRealizations() {
        return evalRealizations;
    }

    /**
     *
     * @return
     */
    public HashSet<ArrayList<Action>> getEvalAttrRealizations() {
        if (this.evalAttrRealizations == null
                || (this.evalAttrRealizations.isEmpty() && !this.evalAttrRealizations.isEmpty())) {
            this.evalAttrRealizations = new HashSet<>();
            for (ArrayList<Action> evalRealization : evalRealizations) {
                ArrayList<Action> evalAttrRealization = new ArrayList<>();
                String previousAttr = "";
                for (Action act : evalRealization) {
                    if (!act.getAttribute().equals(previousAttr)) {
                        if (!act.getAttribute().equals(Action.TOKEN_END)) {
                            evalAttrRealization.add(new Action(Action.TOKEN_START, act.getAttribute()));
                        } else {
                            evalAttrRealization.add(new Action(Action.TOKEN_END, act.getAttribute()));
                        }
                    }
                    if (!act.getAttribute().equals(Action.TOKEN_PUNCT)) {
                        previousAttr = act.getAttribute();
                    }
                }
                this.evalAttrRealizations.add(evalAttrRealization);
            }
        }
        return this.evalAttrRealizations;
    }

    /**
     *
     * @return
     */
    public ArrayList<String> getAlignedSubRealizations() {
        ArrayList<String> subReals = new ArrayList<>();
        for (ArrayList<Action> real : this.alignedSubRealization.keySet()) {
            subReals.add(this.alignedSubRealization.get(real));
        }
        return subReals;
    }

    /**
     *
     * @param realization
     * @return
     */
    public String getRealizationCorrectAction(ArrayList<Action> realization) {
        return this.realizationCorrectAction.get(realization);
    }

    /**
     *
     * @return
     */
    public HashSet<String> getRealizationAltCorrectActions() {
        return realizationAltCorrectActions;
    }

    /**
     *
     * @param evalRealizations
     */
    public void setEvalRealizations(HashSet<ArrayList<Action>> evalRealizations) {
        this.evalRealizations = evalRealizations;
    }

    /**
     *
     * @return
     */
    public ArrayList<Action> getTrainRealization() {
        return trainRealization;
    }

    /**
     *
     * @return
     */
    public HashSet<String> getEvalReferences() {
        return evalReferences;
    }

    /**
     *
     * @param evalReferences
     */
    public void setEvalReferences(HashSet<String> evalReferences) {
        this.evalReferences = evalReferences;
    }

    /**
     *
     * @return
     */
    public String getTrainReference() {
        return trainReference;
    }

    /**
     *
     * @param trainReference
     */
    public void setTrainReference(String trainReference) {
        this.trainReference = trainReference;
    }

    /**
     *
     * @return
     */
    public ArrayList<Action> getTrainAttrRealization() {
        if (this.trainAttrRealization == null
                || (this.trainAttrRealization.isEmpty() && !this.trainRealization.isEmpty())) {
            this.trainAttrRealization = new ArrayList<>();
            String previousAttr = "";
            for (Action act : trainRealization) {
                if (!act.getAttribute().equals(previousAttr)) {
                    if (!act.getAttribute().equals(Action.TOKEN_END)) {
                        this.trainAttrRealization.add(new Action(Action.TOKEN_START, act.getAttribute()));
                    } else {
                        this.trainAttrRealization.add(new Action(Action.TOKEN_END, act.getAttribute()));
                    }
                }
                if (!act.getAttribute().equals(Action.TOKEN_PUNCT)) {
                    previousAttr = act.getAttribute();
                }
            }
        }
        return this.trainAttrRealization;
    }

    /**
     *
     * @param trainRealization
     */
    public void setTrainRealization(ArrayList<Action> trainRealization) {
        this.trainRealization = trainRealization;
        this.trainAttrRealization = new ArrayList<>();
        String previousAttr = "";
        for (Action act : trainRealization) {
            if (!act.getAttribute().equals(previousAttr)) {
                if (!act.getAttribute().equals(Action.TOKEN_END)) {
                    this.trainAttrRealization.add(new Action(Action.TOKEN_START, act.getAttribute()));
                } else {
                    this.trainAttrRealization.add(new Action(Action.TOKEN_END, act.getAttribute()));
                }
            }
            previousAttr = act.getAttribute();
        }
    }

    /**
     *
     * @param o
     * @return
     */
    public int compareTo(DatasetInstance o) {
        final int BEFORE = -1;
        final int EQUAL = 0;
        final int AFTER = 1;

        //this optimization is usually worthwhile, and can
        //always be added
        if (this == o) {
            return EQUAL;
        }

        //primitive numbers follow this form
        if (this.getMeaningRepresentation().getAttributes().values().size() < o.getMeaningRepresentation().getAttributes().values().size()) {
            return BEFORE;
        }
        if (this.getMeaningRepresentation().getAttributes().values().size() > o.getMeaningRepresentation().getAttributes().values().size()) {
            return AFTER;
        }

        //booleans follow this form
        if (this.getTrainRealization().size() < o.getTrainRealization().size()) {
            return BEFORE;
        }
        if (this.getTrainRealization().size() > o.getTrainRealization().size()) {
            return AFTER;
        }

        return EQUAL;
    }
}
