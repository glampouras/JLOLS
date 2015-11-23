package uk.ac.ucl.jarow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;
import static uk.ac.ucl.jarow.Bagel.abstractDatasetInstances;

public class MeaningRepresentation {
    private String predicate;
    private HashMap<String, HashSet<String>> arguments;

    public MeaningRepresentation(String predicate, HashMap<String, HashSet<String>> arguments) {
        this.predicate = predicate;
        this.arguments = arguments;
    }

    public String getPredicate() {
        return predicate;
    }

    public HashMap<String, HashSet<String>> getAttributes() {
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
        if (!this.arguments.equals(other.arguments)) {
            return false;
        }
        return true;
    }
}
