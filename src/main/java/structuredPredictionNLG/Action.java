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
import java.util.Objects;

/**
 * Action implements a single action of the NLG action sequences.
 * The same structure is used for both content and word actions.
 * @author Gerasimos Lampouras
 * @organization University of Sheffield
 */
public class Action implements Serializable {

    /**
     * Default action identifier for the start of an ActionSequence.
     * When combined with a specific attribute, it denotes the start of that attribute 's subsequence.
     */
    final public static String TOKEN_START = "@start@";

    /**
     * Default action identifier for the end of an ActionSequence.
     * When combined with a specific @attribute, it denotes the end of that attribute 's subsequence.
     */
    final public static String TOKEN_END = "@end@";

    /**
     * Default action identifier for punctuation occurrences in an ActionSequence.
     */
    final public static String TOKEN_PUNCT = "@punct@";

    /**
     * Default action identifier for variable occurrences in an ActionSequence.
     * This is usually combined with an attribute identifier and an index integer to distinguish between variables.
     */
    final public static String TOKEN_X = "@x@";

    private static final long serialVersionUID = 1L;
    
    // Main components of an action
    // In practice, attribute may be set as "attribute|value"
    private String word;
    private String attribute;

    /* Collections used for tracking attribute/value pairs when generating the word sequence.
       The first two loggers track attribute/value pairs according to the content sequence (e.g. which attribute/value pairs we have already started and stopped generating words for).
       The latter two loggers track attribute/value pairs according to the word sequence (e.g. which attribute/value pairs have actually been expressed as words in the word sequence).
       The difference exists because we may have stopped generating words for an attribute/value pairs in the content sequence, without actually expressing that attribute/value pair as text.
    */
    HashSet<String> attrValuesBeforeThisTimestep_InContentSequence;
    HashSet<String> attrValuesAfterThisTimestep_InContentSequence;
    ArrayList<String> attrValuesBeforeThisTimestep_InWordSequence;
    ArrayList<String> attrValuesAfterThisTimestep_InWordSequence;
    // This variable logs whether the attribute/value pair we are currently generating words for has been expressed in the current word subsequence or not.
    boolean isValueMentionedAtThisTimestep;
    
    /**
     * Main constructor.
     * Each action consists of a word and corresponding attribute to which the word aligns to. 
     * @param word In addition to any String word appearing in the data, a word can also be any of the default Action identifiers.
     * @param attribute In addition to any String attribute appearing in the data, an attribute can also be a TOKEN_START or TOKEN_END default Action identifiers, denoting the start and end of an ActionSequence respectively.
     */
    public Action(String word, String attribute) {
        this.word = word;
        this.attribute = attribute;
    }

    /**
     * Clone constructor.
     * @param a Action whose values will be used to instantiate this object.
     */
    public Action(Action a) {
        this.word = a.getWord();
        this.attribute = a.getAttribute();
        
        if (a.getAttrValuesBeforeThisTimestep_InContentSequence() != null) {
            this.attrValuesBeforeThisTimestep_InContentSequence = new HashSet<>(a.getAttrValuesBeforeThisTimestep_InContentSequence());
        }
        if (a.getAttrValuesAfterThisTimestep_InContentSequence() != null) {
            this.attrValuesAfterThisTimestep_InContentSequence = new HashSet<>(a.getAttrValuesAfterThisTimestep_InContentSequence());
        }
        if (a.getAttrValuesBeforeThisTimestep_InWordSequence() != null) {
            this.attrValuesBeforeThisTimestep_InWordSequence = new ArrayList<>(a.getAttrValuesBeforeThisTimestep_InWordSequence());
        }
        if (a.getAttrValuesAfterThisTimestep_InWordSequence() != null) {
            this.attrValuesAfterThisTimestep_InWordSequence = new ArrayList<>(a.getAttrValuesAfterThisTimestep_InWordSequence());
        }
        this.isValueMentionedAtThisTimestep = a.isValueMentionedAtThisTimestep;
    }
    
    /**
     * This is used to get a normalized lowercased String descriptor of the Action, to use as a potential label for the classifiers.
     * @return Lowercased String descriptor of the Action.
     */
    public String getAction() {
        return word.toLowerCase().trim();
    }

    /**
     * Returns the value of the word component.
     * @return The value of the word component.
     */
    public String getWord() {
        return word;
    }

    /**
     * Sets the value of the word component.
     * @param word The value of the word component to be set.
     */
    public void setWord(String word) {
        this.word = word;
    }

    /**
     * Returns the value of the attribute component.
     * @return The value of the attribute component.
     */
    public String getAttribute() {
        return attribute;
    }

    /**
     * Sets the value of the attribute component.
     * @param attribute The value of the attribute component to be set.
     */
    public void setAttribute(String attribute) {
        this.attribute = attribute;
    }

    /**
     * Used to log which attribute/value pairs are still yet to be mentioned at this time-step at the content level of the ActionSequence.
     * @return A set of all attribute/value pairs that are still yet to be mentioned at this time-step at the content level of the ActionSequence.
     */
    public HashSet<String> getAttrValuesBeforeThisTimestep_InContentSequence() {
        return attrValuesBeforeThisTimestep_InContentSequence;
    }

    /**
     * Used to log which attribute/value pairs are already mentioned at this time-step at the content level of the ActionSequence.
     * @return A set of all attribute/value pairs that are already mentioned at this time-step at the content level of the ActionSequence.
     */
    public HashSet<String> getAttrValuesAfterThisTimestep_InContentSequence() {
        return attrValuesAfterThisTimestep_InContentSequence;
    }
    
    /**
     * Used to log which attribute/value pairs are still yet to be mentioned at this time-step at the word level of the ActionSequence.
     * @return A set of all attribute/value pairs that are still yet to be mentioned at this time-step at the word level of the ActionSequence.
     */
    public ArrayList<String> getAttrValuesBeforeThisTimestep_InWordSequence() {
        return attrValuesBeforeThisTimestep_InWordSequence;
    }
    
    /**
     * Used to log which attribute/value pairs are already mentioned at this time-step at the word level of the ActionSequence.
     * @return A set of all attribute/value pairs that are already mentioned at this time-step at the word level of the ActionSequence.
     */
    public ArrayList<String> getAttrValuesAfterThisTimestep_InWordSequence() {
        return attrValuesAfterThisTimestep_InWordSequence;
    }

    /**
     * Used to log whether the value of the attribute/value currently generated is already mentioned at this time-step of the ActionSequence.
     * @return True if the value of the attribute/value currently generated is already mentioned at this time-step of the ActionSequence; False otherwise.
     */
    public boolean isValueMentionedAtThisTimestep() {
        return isValueMentionedAtThisTimestep;
    }
    
    /**
     * Quality-of-life method to set the content level loggers at the same time.
     * @param attrValuesBeforeThisTimestep_InContentSequence Before this time-step content-level logger to be set.
     * @param attrValuesAfterThisTimestep_InContentSequence After this time-step content-level logger to be set.
     */
    public void setAttrValueTracking(HashSet<String> attrValuesBeforeThisTimestep_InContentSequence, HashSet<String> attrValuesAfterThisTimestep_InContentSequence) {
        this.attrValuesBeforeThisTimestep_InContentSequence = new HashSet<>(attrValuesBeforeThisTimestep_InContentSequence);
        this.attrValuesAfterThisTimestep_InContentSequence = new HashSet<>(attrValuesAfterThisTimestep_InContentSequence);
    }
    
    /**
     * Quality-of-life method to set all the loggers at the same time.
     * @param attrValuesBeforeThisTimestep_InContentSequence Before this time-step content-level logger to be set.
     * @param attrValuesAfterThisTimestep_InContentSequence After this time-step content-level logger to be set.
     * @param attrValuesBeforeThisTimestep_InWordSequence Before this time-step word-level logger to be set.
     * @param attrValuesAfterThisTimestep_InWordSequence After this time-step word-level logger to be set.
     * @param isValueMentionedAtThisTimestep Logger concerning value of attribute/value pair currently generated.
     */
    public void setAttrValueTracking(HashSet<String> attrValuesBeforeThisTimestep_InContentSequence, HashSet<String> attrValuesAfterThisTimestep_InContentSequence, ArrayList<String> attrValuesBeforeThisTimestep_InWordSequence, ArrayList<String> attrValuesAfterThisTimestep_InWordSequence, boolean isValueMentionedAtThisTimestep) {
        this.attrValuesBeforeThisTimestep_InContentSequence = new HashSet<>(attrValuesBeforeThisTimestep_InContentSequence);
        this.attrValuesAfterThisTimestep_InContentSequence = new HashSet<>(attrValuesAfterThisTimestep_InContentSequence);
        
        this.attrValuesBeforeThisTimestep_InWordSequence = new ArrayList<>(attrValuesBeforeThisTimestep_InWordSequence);
        this.attrValuesAfterThisTimestep_InWordSequence = new ArrayList<>(attrValuesAfterThisTimestep_InWordSequence);
            
        this.isValueMentionedAtThisTimestep = isValueMentionedAtThisTimestep;
    }
    
    /**
     * Sets all the loggers as copies of those in another Action.
     * @param a The Action whose loggers to copy.
     */
    public void setAttrValueTracking(Action a) {
        if (a.getAttrValuesBeforeThisTimestep_InContentSequence() != null) {
            this.attrValuesBeforeThisTimestep_InContentSequence = new HashSet<>(a.getAttrValuesBeforeThisTimestep_InContentSequence());
        }
        if (a.getAttrValuesAfterThisTimestep_InContentSequence() != null) {
            this.attrValuesAfterThisTimestep_InContentSequence = new HashSet<>(a.getAttrValuesAfterThisTimestep_InContentSequence());
        }
        if (a.getAttrValuesBeforeThisTimestep_InWordSequence() != null) {
            this.attrValuesBeforeThisTimestep_InWordSequence = new ArrayList<>(a.getAttrValuesBeforeThisTimestep_InWordSequence());
        }
        if (a.getAttrValuesAfterThisTimestep_InWordSequence() != null) {
            this.attrValuesAfterThisTimestep_InWordSequence = new ArrayList<>(a.getAttrValuesAfterThisTimestep_InWordSequence());
        }
        this.isValueMentionedAtThisTimestep = a.isValueMentionedAtThisTimestep;
    }

    /**
     * Returns the following string representation of the Action: A{word, attribute}. 
     * @return A string representation of the Action. 
     */
    @Override
    public String toString() {
        return "A{" + word + "," + attribute + '}';
    }

    /**
     * Returns a hash code value for this Action.
     * @return A hash code value for this Action.
     */
    @Override
    public int hashCode() {
        int hash = 5;
        hash = 29 * hash + Objects.hashCode(this.word);
        return hash;
    }

    /**
     * Indicates whether some other Action is "equal to" this one. 
     * @param obj The reference object with which to compare.
     * @return true if this action is the same as the obj argument; false otherwise.
     */
    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Action other = (Action) obj;
        return Objects.equals(this.word, other.word);
    }
}
