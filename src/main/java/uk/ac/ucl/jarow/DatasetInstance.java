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
    
    HashMap<ArrayList<Action>, ArrayList<String>> mentionedValueSequences;
    HashMap<ArrayList<Action>, ArrayList<String>> mentionedAttributeSequences;
    
    HashSet<ArrayList<Action>> realizations;

    public DatasetInstance(MeaningRepresentation MR, ArrayList<String> mentionedValueSequence, ArrayList<String> mentionedAttributeSequence, ArrayList<Action> realization) {
        this.MR = MR;
        this.mentionedValueSequences = new HashMap<>();
        this.mentionedValueSequences.put(realization, mentionedValueSequence);
        
        this.mentionedAttributeSequences = new HashMap<>();
        this.mentionedAttributeSequences.put(realization, mentionedAttributeSequence);
        
        this.realizations = new HashSet<>();
        this.realizations.add(realization);
    }   
        
    public void mergeDatasetInstance(ArrayList<String> mentionedValueSequence, ArrayList<String> mentionedAttributeSequence, ArrayList<Action> realization) {
        this.realizations.add(realization);
        this.mentionedValueSequences.put(realization, mentionedValueSequence);
        this.mentionedAttributeSequences.put(realization, mentionedAttributeSequence);
    }

    public MeaningRepresentation getMeaningRepresentation() {
        return MR;
    }

    public HashMap<ArrayList<Action>, ArrayList<String>> getMentionedValueSequences() {
        return mentionedValueSequences;
    }

    public HashMap<ArrayList<Action>, ArrayList<String>> getMentionedAttributeSequences() {
        return mentionedAttributeSequences;
    }

    public HashSet<ArrayList<Action>> getRealizations() {
        return realizations;
    }
}
