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
import java.util.HashSet;
import java.util.Objects;

/**
 * Action implements a single action of the NLG action sequences.
 * The same structure is used for both content and word actions.
 * @author Gerasimos Lampouras
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
    private String word;
    private String attribute;

    // Collections used for logging of attribute/value pairs throughout the ActionSequence, on the content and word sequence levels
    HashSet<String> attrValuesBeforeThisPointInContentSequence;
    HashSet<String> attrValuesAfterThisPointInContentSequence;
    ArrayList<String> attrValuesBeforeThisPointInWordSequence;
    ArrayList<String> attrValuesAfterThisPointInWordSequence;
    boolean isValueMentionedAtThisPoint;
    
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
        
        if (a.getAttrValuesBeforeThisPointInContentSequence() != null) {
            this.attrValuesBeforeThisPointInContentSequence = new HashSet<>(a.getAttrValuesBeforeThisPointInContentSequence());
        }
        if (a.getAttrValuesAfterThisPointInContentSequence() != null) {
            this.attrValuesAfterThisPointInContentSequence = new HashSet<>(a.getAttrValuesAfterThisPointInContentSequence());
        }
        if (a.getAttrValuesBeforeThisPointInWordSequence() != null) {
            this.attrValuesBeforeThisPointInWordSequence = new ArrayList<>(a.getAttrValuesBeforeThisPointInWordSequence());
        }
        if (a.getAttrValuesAfterThisPointInWordSequence() != null) {
            this.attrValuesAfterThisPointInWordSequence = new ArrayList<>(a.getAttrValuesAfterThisPointInWordSequence());
        }
        this.isValueMentionedAtThisPoint = a.isValueMentionedAtThisPoint;
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
     * Used to log which attribute/value pairs are still yet to be mentioned at this point at the content level of the ActionSequence.
     * @return A set of all attribute/value pairs that are still yet to be mentioned at this point at the content level of the ActionSequence.
     */
    public HashSet<String> getAttrValuesBeforeThisPointInContentSequence() {
        return attrValuesBeforeThisPointInContentSequence;
    }

    /**
     * Used to log which attribute/value pairs are already mentioned at this point at the content level of the ActionSequence.
     * @return A set of all attribute/value pairs that are already mentioned at this point at the content level of the ActionSequence.
     */
    public HashSet<String> getAttrValuesAfterThisPointInContentSequence() {
        return attrValuesAfterThisPointInContentSequence;
    }
    
    /**
     * Used to log which attribute/value pairs are still yet to be mentioned at this point at the word level of the ActionSequence.
     * @return A set of all attribute/value pairs that are still yet to be mentioned at this point at the word level of the ActionSequence.
     */
    public ArrayList<String> getAttrValuesBeforeThisPointInWordSequence() {
        return attrValuesBeforeThisPointInWordSequence;
    }
    
    /**
     * Used to log which attribute/value pairs are already mentioned at this point at the word level of the ActionSequence.
     * @return A set of all attribute/value pairs that are already mentioned at this point at the word level of the ActionSequence.
     */
    public ArrayList<String> getAttrValuesAfterThisPointInWordSequence() {
        return attrValuesAfterThisPointInWordSequence;
    }

    /**
     * Used to log whether the value of the attribute/value currently generated is already mentioned at this point of the ActionSequence.
     * @return True if the value of the attribute/value currently generated is already mentioned at this point of the ActionSequence; False otherwise.
     */
    public boolean isValueMentionedAtThisPoint() {
        return isValueMentionedAtThisPoint;
    }
    
    /**
     * Quality-of-life method to set the content level loggers at the same time.
     * @param attrValuesBeforeThisPointInContentSequence Before this point content-level logger to be set.
     * @param attrValuesAfterThisPointInContentSequence After this point content-level logger to be set.
     */
    public void setAttrValueTracking(HashSet<String> attrValuesBeforeThisPointInContentSequence, HashSet<String> attrValuesAfterThisPointInContentSequence) {
        this.attrValuesBeforeThisPointInContentSequence = new HashSet<>(attrValuesBeforeThisPointInContentSequence);
        this.attrValuesAfterThisPointInContentSequence = new HashSet<>(attrValuesAfterThisPointInContentSequence);
    }
    
    /**
     * Quality-of-life method to set all the loggers at the same time.
     * @param attrValuesBeforeThisPointInContentSequence Before this point content-level logger to be set.
     * @param attrValuesAfterThisPointInContentSequence After this point content-level logger to be set.
     * @param attrValuesBeforeThisPointInWordSequence Before this point word-level logger to be set.
     * @param attrValuesAfterThisPointInWordSequence After this point word-level logger to be set.
     * @param isValueMentionedAtThisPoint Logger concerning value of attribute/value pair currently generated.
     */
    public void setAttrValueTracking(HashSet<String> attrValuesBeforeThisPointInContentSequence, HashSet<String> attrValuesAfterThisPointInContentSequence, ArrayList<String> attrValuesBeforeThisPointInWordSequence, ArrayList<String> attrValuesAfterThisPointInWordSequence, boolean isValueMentionedAtThisPoint) {
        this.attrValuesBeforeThisPointInContentSequence = new HashSet<>(attrValuesBeforeThisPointInContentSequence);
        this.attrValuesAfterThisPointInContentSequence = new HashSet<>(attrValuesAfterThisPointInContentSequence);
        
        this.attrValuesBeforeThisPointInWordSequence = new ArrayList<>(attrValuesBeforeThisPointInWordSequence);
        this.attrValuesAfterThisPointInWordSequence = new ArrayList<>(attrValuesAfterThisPointInWordSequence);
            
        this.isValueMentionedAtThisPoint = isValueMentionedAtThisPoint;
    }
    
    /**
     * Sets all the loggers as copies of those in another Action.
     * @param a The Action whose loggers to copy.
     */
    public void setAttrValueTracking(Action a) {
        if (a.getAttrValuesBeforeThisPointInContentSequence() != null) {
            this.attrValuesBeforeThisPointInContentSequence = new HashSet<>(a.getAttrValuesBeforeThisPointInContentSequence());
        }
        if (a.getAttrValuesAfterThisPointInContentSequence() != null) {
            this.attrValuesAfterThisPointInContentSequence = new HashSet<>(a.getAttrValuesAfterThisPointInContentSequence());
        }
        if (a.getAttrValuesBeforeThisPointInWordSequence() != null) {
            this.attrValuesBeforeThisPointInWordSequence = new ArrayList<>(a.getAttrValuesBeforeThisPointInWordSequence());
        }
        if (a.getAttrValuesAfterThisPointInWordSequence() != null) {
            this.attrValuesAfterThisPointInWordSequence = new ArrayList<>(a.getAttrValuesAfterThisPointInWordSequence());
        }
        this.isValueMentionedAtThisPoint = a.isValueMentionedAtThisPoint;
    }

    /**
     * Returns the following string representation of the Action: A{word, attribute}. 
     * @return a string representation of the Action. 
     */
    @Override
    public String toString() {
        return "A{" + word + "," + attribute + '}';
    }

    /**
     *
     * @return a hash code value for this Action.
     */
    @Override
    public int hashCode() {
        int hash = 5;
        hash = 29 * hash + Objects.hashCode(this.word);
        return hash;
    }

    /**
     * Indicates whether some other Action is "equal to" this one. 
     * @param obj the reference object with which to compare.
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
