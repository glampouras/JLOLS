
package structuredPredictionNLG;

import static structuredPredictionNLG.Action.TOKEN_END;
import static structuredPredictionNLG.Action.TOKEN_PUNCT;
import static structuredPredictionNLG.Action.TOKEN_START;
import java.util.ArrayList;


/*
 * Each ActionSequence consists of an ArrayList of Actions.
 * The ActionSequence typically begins with a series of content actions,
 * and followed by subsets of word actions, each corresponding to one of the preceding content actions.
 * @author Gerasimos Lampouras
 * @organization University of Sheffield
 */
public class ActionSequence implements Comparable<ActionSequence> {
    private ArrayList<Action> sequence;

    /**
     * Empty constructor.
     */
    public ActionSequence() {
        sequence = new ArrayList<Action>();
    }

    /**
     * Main constructor.
     * @param sequence ArrayList whose values will be used to instantiate this object.
     */
    public ActionSequence(ArrayList<Action> sequence) {
        this.sequence = new ArrayList<>();
        sequence.stream().forEach((a) -> {
            this.sequence.add(new Action(a));
        });
    }

    /**
     * Clone constructor. 
     * @param as ActionSequence whose values will be used to instantiate this object.
     */
    public ActionSequence(ActionSequence as) {
        this.sequence = new ArrayList<>();
        as.getSequence().stream().forEach((a) -> {
            this.sequence.add(new Action(a));
        });
    }

    /**
     * Alternative clone constructor, where Action.TOKEN_START and Action.TOKEN_END actions may be ommited.
     * @param sequence ActionSequence whose values will be used to instantiate this object.
     * @param cleanEnds Whether or not Action.TOKEN_START and Action.TOKEN_END actions should be ommited.
     */
    public ActionSequence(ArrayList<Action> sequence, boolean cleanEnds) {
        this.sequence = new ArrayList<>();
        if (cleanEnds) {
            sequence.stream().filter((a) -> (!a.getWord().equals(TOKEN_START)
                    && !a.getWord().equals(TOKEN_END))).forEach((a) -> {
                        this.sequence.add(new Action(a));
            });
        } else {
            sequence.stream().forEach((a) -> {
                this.sequence.add(new Action(a));
            });
        }
    }

    /**
     * Returns the ArrayList consisting the ActionSequence.
     * @return The ArrayList consisting the ActionSequence
     */
    public ArrayList<Action> getSequence() {
        return sequence;
    }

    /**
     * Replace the action at the indexed cell of the ActionSequence, and shorten the sequence up to and including the index.
     * Initially, this method is used to replace the action at a timestep of a roll-in sequence with an alternative action.
     * Afterwards, it shortens the sequence so that the rest of it (after the index) can be recalculated by performing a roll-out.
     * @param index The index of the sequence to be modified.
     * @param decision The Action that should replace the indexed action of the ActionSequence.
     */
    public void modifyAndShortenSequence(int index, Action decision) {
        decision.setAttrValueTracking(this.sequence.get(index));
        
        this.sequence.set(index, decision);
        this.sequence = new ArrayList<Action>(this.sequence.subList(0, index + 1));
    }

    /**
     * Returns a string representation of the word actions in the ActionSequence.
     * @return A string representation of the word actions in the ActionSequence.
     */
    public String getWordSequenceToString() {
        String w = "";
        w = sequence.stream().filter((act) -> (!act.getWord().equals(TOKEN_START)
                && !act.getWord().equals(TOKEN_END))).map((act) -> act.getWord() + " ").reduce(w, String::concat);
        return w.trim();
    }

    /**
     * Returns a string representation of the word actions in the ActionSequence, while ommiting all puntuation.
     * @return A string representation of the word actions in the ActionSequence, without any puntuation.
     */
    public String getWordSequenceToString_NoPunct() {
        String w = "";
        w = sequence.stream().filter((act) -> (!act.getWord().equals(TOKEN_START)
                && !act.getWord().equals(TOKEN_END)
                && !act.getAttribute().equals(TOKEN_PUNCT))).map((act) -> act.getWord() + " ").reduce(w, String::concat);
        return w.trim();
    }

    /**
     * Returns a string representation of the content actions in the ActionSequence.
     * @return A string representation of the content actions in the ActionSequence.
     */
    public String getAttrSequenceToString() {
        String w = "";
        w = sequence.stream().map((act) -> act.getAttribute() + " ").reduce(w, String::concat);
        return w.trim();
    }

    /**
     * Returns the length of the sequence when not accounting for Action.TOKEN_START and Action.TOKEN_END actions.
     * @return The length of the sequence when not accounting for Action.TOKEN_START and Action.TOKEN_END actions.
     */
    public int getLength_NoBorderTokens() {
        int l = 0;
        l = sequence.stream().filter((act) -> (!act.getWord().equals(TOKEN_START)
                && !act.getWord().equals(TOKEN_END))).map((_item) -> 1).reduce(l, Integer::sum);
        return l;
    }

    /**
     * Returns the length of the sequence when not accounting for Action.TOKEN_START and Action.TOKEN_END actions, and punctuation.
     * @return The length of the sequence when not accounting for Action.TOKEN_START and Action.TOKEN_END actions, and punctuation.
     */
    public int getLength_NoBorderTokens_NoPunct() {
        int l = 0;
        l = sequence.stream().filter((act) -> (!act.getWord().equals(TOKEN_START)
                && !act.getWord().equals(TOKEN_END)
                && !act.getAttribute().equals(TOKEN_PUNCT))).map((_item) -> 1).reduce(l, Integer::sum);
        return l;
    }

    /**
     * Returns a subsequence consisting only of the content actions in the ActionSequence.
     * @return A subsequence consisting only of the content actions in the ActionSequence.
     */
    public ArrayList<String> getAttributeSequence() {
        ArrayList<String> seq = new ArrayList<>();
        sequence.stream().forEach((act) -> {
            seq.add(act.getAttribute());
        });
        return seq;
    }
    
    /**
     * Returns a subsequence consisting only of the content actions in the ActionSequence, up to a specified index.
     * @param index Endpoint (exclusive) of the content subsequence.
     * @return A subsequence consisting only of the content actions in the ActionSequence, up to a specified index.
     */
    public ArrayList<String> getAttributeSubSequence(int index) {
        ArrayList<String> seq = new ArrayList<String>();
        sequence.subList(0, index).stream().forEach((act) -> {
            seq.add(act.getAttribute());
        });
        return seq;
    }

    /**
     * Returns a hash code value for this Action.
     * @return A hash code value for this Action.
     */
    @Override
    public int hashCode() {
        return this.getWordSequenceToString().hashCode();
    }

    /**
     * Indicates whether some other ActionSequence is "equal to" this one. 
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
        final ActionSequence other = (ActionSequence) obj;
        return this.getSequence().toString().equals(other.getSequence().toString());
    }

    /**
     * Compares this ActionSequence to another, based on their respective lengths.
     * @param o The reference object with which to compare.
     * @return -1 if this ActionSequence's length is smaller, 0 if the lengths are equal, and 1 otherwise.
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
     * Returns a string representation of the ActionSequence. 
     * @return A string representation of the ActionSequence
     */
    @Override
    public String toString() {
        return "ActionSequence{" + "sequence=" + sequence + '}';
    }
}