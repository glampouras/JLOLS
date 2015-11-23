/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uk.ac.ucl.jarow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 *
 * @author localadmin
 */
public class DatasetInstance {
    private MeaningRepresentation MR;
    
    HashSet<ArrayList<String>> mentionedValueSequences;
    HashSet<ArrayList<String>> mentionedAttributeSequences;
    
    HashSet<ArrayList<Action>> realizations;

    public DatasetInstance(MeaningRepresentation MR, ArrayList<String> mentionedValueSequence, ArrayList<String> mentionedAttributeSequence, ArrayList<Action> realization) {
        this.MR = MR;
        this.mentionedValueSequences = new HashSet<>();
        this.mentionedValueSequences.add(mentionedValueSequence);
        
        this.mentionedAttributeSequences = new HashSet<>();
        this.mentionedAttributeSequences.add(mentionedAttributeSequence);
        
        this.realizations = new HashSet<>();
        this.realizations.add(realization);
    }   
    
    public DatasetInstance(MeaningRepresentation MR, ArrayList<String> mentionedValueSequence, ArrayList<String> mentionedAttributeSequence, HashSet<ArrayList<Action>> realizations) {
        this.MR = MR;
        this.mentionedValueSequences = new HashSet<>();
        this.mentionedValueSequences.add(mentionedValueSequence);
        
        this.mentionedAttributeSequences = new HashSet<>();
        this.mentionedAttributeSequences.add(mentionedAttributeSequence);
        
        this.realizations = new HashSet<>();
        this.realizations.addAll(realizations);
    }       
    
    public void mergeDatasetInstance(DatasetInstance DI) {
        this.mentionedValueSequences.addAll(DI.getMentionedValueSequences());
        this.mentionedAttributeSequences.addAll(DI.getMentionedAttributeSequences());
        this.realizations.addAll(DI.getRealizations());
    }
    
    public void mergeDatasetInstance(ArrayList<String> mentionedValueSequence, ArrayList<String> mentionedAttributeSequence, ArrayList<Action> realization) {
        this.mentionedValueSequences.add(mentionedValueSequence);
        this.mentionedAttributeSequences.add(mentionedAttributeSequence);
        this.realizations.add(realization);
    }
    
    public void mergeDatasetInstance(ArrayList<String> mentionedValueSequence, ArrayList<String> mentionedAttributeSequence, HashSet<ArrayList<Action>> realizations) {
        this.mentionedValueSequences.add(mentionedValueSequence);
        this.mentionedAttributeSequences.add(mentionedAttributeSequence);
        this.realizations.addAll(realizations);
    }

    public MeaningRepresentation getMeaningRepresentation() {
        return MR;
    }

    public HashSet<ArrayList<String>> getMentionedValueSequences() {
        return mentionedValueSequences;
    }

    public HashSet<ArrayList<String>> getMentionedAttributeSequences() {
        return mentionedAttributeSequences;
    }

    public HashSet<ArrayList<Action>> getRealizations() {
        return realizations;
    }
}
