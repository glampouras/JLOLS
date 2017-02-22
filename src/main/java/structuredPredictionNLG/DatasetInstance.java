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
package structuredPredictionNLG;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;

/**
 * Internal representation of an instance of the dataset
 * @author Gerasimos Lampouras
 * @organization University of Sheffield
 */
public class DatasetInstance implements Serializable, Comparable<DatasetInstance> {
    private static final long serialVersionUID = 1L;
    
    private MeaningRepresentation MR;
    // A reference for the word actions of the DatasetInstance; this is constructed using the reference directly corresponding to this instance in the dataset
    private ArrayList<Action> directReferenceSequence;    
    // A reference for the content actions of the DatasetInstance; this is constructed using the reference directly corresponding to this instance in the dataset
    private ArrayList<Action> directAttrSequence;
    // Realized string of the word actions in the direct reference 
    private String directReference = "";
    
    // References to be used during evaluation of this DatasetInstance
    private HashSet<String> evaluationReferences = new HashSet<String>();

    /**
     * Main constructor
     * @param MR The MeaningRepresentation corresponding to this instance of the dataset.
     * @param directReferenceSequence The ActionSequence of the direct reference in the dataset.
     * @param directReference Realized string of the word actions in the direct reference.
     */
    public DatasetInstance(MeaningRepresentation MR, ArrayList<Action> directReferenceSequence, String directReference) {
        this.MR = MR;

        this.directReferenceSequence = directReferenceSequence;
        this.directReference = directReference;

        this.evaluationReferences = new HashSet<>();
        this.evaluationReferences.add(directReference);
    }

    /**
     * Clone constructor.
     * @param di DatasetInstance whose values will be used to instantiate this object.
     */
    public DatasetInstance(DatasetInstance di) {
        this.MR = di.getMeaningRepresentation();

        this.directReferenceSequence = di.getDirectReferenceSequence();

        this.directReference = di.getDirectReference();
        this.evaluationReferences.addAll(di.getEvaluationReferences());
    }

    /**
     * Returns the meaning representation of this DatasetInstance.
     * @return The meaning representation of this DatasetInstance.
     */
    public MeaningRepresentation getMeaningRepresentation() {
        return MR;
    }

    /**
     * Returns the ActionSequence of the direct reference of this DatasetInstance.
     * @return The ActionSequence of the direct reference of this DatasetInstance.
     */
    public ArrayList<Action> getDirectReferenceSequence() {
        return directReferenceSequence;
    }

    /**
     * Returns (and constructs when first called) a sequence of content actions based on the direct referenec of this DatasetInstance.
     * @return A sequence of content actions.
     */
    public ArrayList<Action> getDirectReferenceAttrValueSequence() {
        if (this.directAttrSequence == null
                || (this.directAttrSequence.isEmpty() && !this.directReferenceSequence.isEmpty())) {
            this.directAttrSequence = new ArrayList<>();
            String previousAttr = "";
            for (Action act : directReferenceSequence) {
                if (!act.getAttribute().equals(previousAttr)) {
                    if (!act.getAttribute().equals(Action.TOKEN_END)) {
                        this.directAttrSequence.add(new Action(Action.TOKEN_START, act.getAttribute()));
                    } else {
                        this.directAttrSequence.add(new Action(Action.TOKEN_END, act.getAttribute()));
                    }
                }
                if (!act.getAttribute().equals(Action.TOKEN_PUNCT)) {
                    previousAttr = act.getAttribute();
                }
            }
        }
        return this.directAttrSequence;
    }

    /**
     * Sets the word action sequence (and also constructs the corresponding content action sequence) to be used as direct reference sequence for the DatasetInstance.
     * @param directReferenceSequence The word action sequence to be set.
     */
    public void setDirectReferenceSequence(ArrayList<Action> directReferenceSequence) {
        this.directReferenceSequence = directReferenceSequence;
        this.directAttrSequence = new ArrayList<>();
        String previousAttr = "";
        for (Action act : directReferenceSequence) {
            if (!act.getAttribute().equals(previousAttr)) {
                if (!act.getAttribute().equals(Action.TOKEN_END)) {
                    this.directAttrSequence.add(new Action(Action.TOKEN_START, act.getAttribute()));
                } else {
                    this.directAttrSequence.add(new Action(Action.TOKEN_END, act.getAttribute()));
                }
            }
            previousAttr = act.getAttribute();
        }
    }

    /**
     * Returns the direct references of this DatasetInstance.
     * @return The direct references of this DatasetInstance.
     */
    public String getDirectReference() {
        return directReference;
    }

    /**
     * Returns the references used for evaluation of this DatasetInstance.
     * @return The references used for evaluation of this DatasetInstance.
     */
    public HashSet<String> getEvaluationReferences() {
        return evaluationReferences;
    }

    /**
     * Compares this DatasetInstance to another, based on their MeaningRepresentation and realizations.
     * @param o The reference object with which to compare.
     * @return -1 if this DatasetInstance's length has less attribute/value pairs or a realization of lesser length, 0 if they are equal, and 1 otherwise.
     */
    @Override
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
        if (this.getMeaningRepresentation().getAttributeValues().values().size() < o.getMeaningRepresentation().getAttributeValues().values().size()) {
            return BEFORE;
        }
        if (this.getMeaningRepresentation().getAttributeValues().values().size() > o.getMeaningRepresentation().getAttributeValues().values().size()) {
            return AFTER;
        }

        //booleans follow this form
        if (this.getDirectReferenceSequence().size() < o.getDirectReferenceSequence().size()) {
            return BEFORE;
        }
        if (this.getDirectReferenceSequence().size() > o.getDirectReferenceSequence().size()) {
            return AFTER;
        }

        return EQUAL;
    }
}
