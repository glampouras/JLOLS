package uk.ac.ucl.jarow;

import java.util.Objects;

public class Action {
    private String word;
    private String attribute;
        
    public Action(String word, String attribute) {
        this.word = word;
        this.attribute = attribute;
    }
    
    public Action(Action a) {
        this.word = a.getWord();
        this.attribute = a.getAttribute();
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public String getAttribute() {
        return attribute;
    }

    public void setAttribute(String attribute) {
        this.attribute = attribute;
    }

    @Override
    public String toString() {
        return "A{" + word + "," + attribute + '}';
    }

    @Override
    public int hashCode() {
        int hash = 5;
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
        final Action other = (Action) obj;
        if (!Objects.equals(this.word, other.word)) {
            return false;
        }
        return true;
    }
}
