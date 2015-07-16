package uk.ac.ucl.jarow;

import java.util.ArrayList;
import java.util.Objects;

public class MeaningRepresentation {
    private String predicate;
    private ArrayList<String> arguments;

    public MeaningRepresentation(String predicate, ArrayList<String> arguments) {
        this.predicate = predicate;
        this.arguments = arguments;
    }

    public String getPredicate() {
        return predicate;
    }

    public ArrayList<String> getArguments() {
        return arguments;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 61 * hash + Objects.hashCode(this.predicate);
        hash = 61 * hash + Objects.hashCode(this.arguments);
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
        final MeaningRepresentation other = (MeaningRepresentation) obj;
        if (!Objects.equals(this.predicate, other.predicate)) {
            return false;
        }
        if (this.arguments.size() != other.arguments.size()) {
            return false;
        }
        for (int i = 0; i < this.arguments.size(); i++) {
            if (!this.arguments.get(i).equals(other.arguments.get(i))) {
                return false;
            }
        }
        return true;
    }
}
