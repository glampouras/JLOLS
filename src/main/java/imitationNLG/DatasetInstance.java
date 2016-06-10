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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class DatasetInstance {

    private MeaningRepresentation MR;
    ArrayList<String> trainMentionedValueSequence;
    ArrayList<String> trainMentionedAttributeSequence;
    ArrayList<Action> trainRealization;
    HashMap<ArrayList<Action>, ArrayList<String>> evalMentionedValueSequences;
    HashMap<ArrayList<Action>, ArrayList<String>> evalMentionedAttributeSequences;
    HashSet<ArrayList<Action>> evalRealizations;
    HashMap<ArrayList<Action>, String> alignedSubRealization;
    HashMap<ArrayList<Action>, String> realizationCorrectAction;
    HashSet<String> realizationAltCorrectActions;

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

        this.realizationAltCorrectActions = new HashSet<String>();
    }

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
    }

    public void mergeDatasetInstance(HashMap<ArrayList<Action>, ArrayList<String>> mentionedValueSequences, HashMap<ArrayList<Action>, ArrayList<String>> mentionedAttributeSequences, HashSet<ArrayList<Action>> realizations) {
        this.evalRealizations.addAll(realizations);
        this.evalMentionedValueSequences.putAll(mentionedValueSequences);
        this.evalMentionedAttributeSequences.putAll(mentionedAttributeSequences);
    }

    public void mergeDatasetInstance(ArrayList<String> mentionedValueSequence, ArrayList<String> mentionedAttributeSequence, ArrayList<Action> realization) {
        this.evalRealizations.add(realization);
        this.evalMentionedValueSequences.put(realization, mentionedValueSequence);
        this.evalMentionedAttributeSequences.put(realization, mentionedAttributeSequence);
    }

    public MeaningRepresentation getMeaningRepresentation() {
        return MR;
    }

    public HashMap<ArrayList<Action>, ArrayList<String>> getEvalMentionedValueSequences() {
        return evalMentionedValueSequences;
    }

    public HashMap<ArrayList<Action>, ArrayList<String>> getEvalMentionedAttributeSequences() {
        return evalMentionedAttributeSequences;
    }

    public HashSet<ArrayList<Action>> getEvalRealizations() {
        return evalRealizations;
    }

    public void setRealizations(HashSet<ArrayList<Action>> realizations) {
        this.evalRealizations = new HashSet<>();
        this.evalRealizations.addAll(realizations);
    }

    public void setAlignedSubRealization(ArrayList<Action> realization, String alignedSubRealization) {
        this.alignedSubRealization.put(realization, alignedSubRealization);
    }

    public String getAlignedSubRealization(ArrayList<Action> realization) {
        return this.alignedSubRealization.get(realization);
    }

    public ArrayList<String> getAlignedSubRealizations() {
        ArrayList<String> subReals = new ArrayList<>();
        for (ArrayList<Action> real : this.alignedSubRealization.keySet()) {
            subReals.add(this.alignedSubRealization.get(real));
        }
        return subReals;
    }

    public void setRealizationCorrectAction(ArrayList<Action> realization, String realizationCorrectAction) {
        this.realizationCorrectAction.put(realization, realizationCorrectAction);
    }

    public String getRealizationCorrectAction(ArrayList<Action> realization) {
        return this.realizationCorrectAction.get(realization);
    }

    public HashSet<String> getRealizationAltCorrectActions() {
        return realizationAltCorrectActions;
    }

    public void resetRealizationAltCorrectActions() {
        this.realizationAltCorrectActions = new HashSet<String>();
    }

    public void addRealizationAltCorrectActions(String altCorrectActions) {
        this.realizationAltCorrectActions.add(altCorrectActions);
    }

    public void setEvalRealizations(HashSet<ArrayList<Action>> evalRealizations) {
        this.evalRealizations = evalRealizations;
    }

    public ArrayList<Action> getTrainRealization() {
        return trainRealization;
    }

    public void setTrainRealization(ArrayList<Action> trainRealization) {
        this.trainRealization = trainRealization;
    }
}
