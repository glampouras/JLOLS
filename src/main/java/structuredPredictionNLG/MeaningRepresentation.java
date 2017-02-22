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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;


/**
 * Internal representation of a Meaning Representation
 * @author Gerasimos Lampouras
 * @organization University of Sheffield
 */
public class MeaningRepresentation implements Serializable {
    private static final long serialVersionUID = 1L;
    
    //  A MeaningRepresentation consists of a predicate, and a map of attributes to sets of values
    private final String predicate;
    private final HashMap<String, HashSet<String>> attributeValues;
    /*  
     * This variable stores the string describing the meaning representation in the dataset.
     * Used mostly for tracking "unique" instances of an MR in the dataset, when lexicalization and attribute order are considered.
    */
    private String MRstr = "";
    
    /*
     * This variable maps the variable values (e.g. @x@attr), to the corresponding lexicalized string values.
     * It is populated during the initial delexicalization of the MR, and used after generation for post-processing re-lexicalization of the variables.
    */
    private HashMap<String, String> delexicalizationMap = new HashMap<>();    

    /**
     * Main constructor
     * @param predicate The predicate of the meaning representation.
     * @param attributeValues A map of the meaning representation's attributes to their values.
     * @param MRstr A string describing the meaning representation in the dataset.
     */
    public MeaningRepresentation(String predicate, HashMap<String, HashSet<String>> attributeValues, String MRstr) {
        this.predicate = predicate;
        this.attributeValues = attributeValues;
        this.MRstr = MRstr;
    }
    
    /**
     * Secondary constructor
     * @param predicate The predicate of the meaning representation.
     * @param attributeValues A map of the meaning representation's attributes to their values.
     * @param MRstr A string describing the meaning representation in the dataset.
     * @param delexicalizationMap The map between variable values and corresponding lexicalized string values, to be set.
     */
    public MeaningRepresentation(String predicate, HashMap<String, HashSet<String>> attributeValues, String MRstr, HashMap<String, String> delexicalizationMap) {
        this.predicate = predicate;
        this.attributeValues = attributeValues;
        this.MRstr = MRstr;
        this.delexicalizationMap = delexicalizationMap;
    }

    /**
     * Returns the value of the predicate.
     * @return The value of the predicate.
     */
    public String getPredicate() {
        return predicate;
    }

    /**
     * Returns the map of attributes to values.
     * @return The map of attributes to values.
     */
    public HashMap<String, HashSet<String>> getAttributeValues() {
        return attributeValues;
    }

    /**
     * A string representation of the MeaningRepresantation, used primarily to compare MeaningRepresantation objects.
     * We store the value, so we do not have to reconstruct it.
     */
    private String abstractMR = "";

    /**
     * Returns (and constructs when first called) a string representation of the MeaningRepresantation.
     * @return A string representation of the MeaningRepresantation.
     */
    public String getAbstractMR() {
        if (abstractMR.isEmpty()) {
            abstractMR = predicate + ":";
            ArrayList<String> attrs = new ArrayList<>(attributeValues.keySet());
            Collections.sort(attrs);
            HashMap<String, Integer> xCounts = new HashMap<>();
            attrs.forEach((attr) -> {
                xCounts.put(attr, 0);
            });
            attrs.stream().map((attr) -> {
                abstractMR += attr + "={";
                return attr;
            }).map((attr) -> {
                ArrayList<String> values = new ArrayList<>(attributeValues.get(attr));
                Collections.sort(values);
                values.forEach((value) -> {
                    if (attr.equals("name")
                            || attr.equals("type")
                            || attr.equals("pricerange")
                            || attr.equals("price")
                            || attr.equals("phone")
                            || attr.equals("address")
                            || attr.equals("postcode")
                            || attr.equals("area")
                            || attr.equals("near")
                            || attr.equals("food")
                            || attr.equals("goodformeal")
                            || attr.equals("count")) {
                        abstractMR += Action.TOKEN_X + attr + "_" + xCounts.get(attr) + ",";
                        xCounts.put(attr, xCounts.get(attr) + 1);
                    } else {
                        abstractMR += value + ",";
                    }
                });            
                return attr;
            }).forEachOrdered((_item) -> {
                abstractMR += "}";
            });
        }
        return abstractMR;
    }

    /**
     * Returns the value of the string describing the meaning representation in the dataset.
     * @return The value of the string describing the meaning representation in the dataset.
     */
    public String getMRstr() {
        return MRstr;
    }

    /**
     * Returns a hash code value for this Action.
     * @return A hash code value for this Action.
     */
    @Override
    public int hashCode() {
        int hash = 7;
        hash = 61 * hash + Objects.hashCode(this.predicate);
        hash = 61 * hash + Objects.hashCode(this.attributeValues);
        return hash;
    }

    /**
     * Indicates whether some other MeaningRepresenation is "equal to" this one. 
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
        final MeaningRepresentation other = (MeaningRepresentation) obj;
        if (!Objects.equals(this.predicate, other.predicate)) {
            return false;
        }
        return this.attributeValues.equals(other.attributeValues);
    }

    /**
     * Returns the map between the variable values and corresponding lexicalized string values.
     * @return The maps between the variable values and corresponding lexicalized string values.
     */
    public HashMap<String, String> getDelexicalizationMap() {
        return delexicalizationMap;
    }
}
