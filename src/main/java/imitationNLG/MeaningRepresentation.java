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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;

/**
 *
 * @author Gerasimos Lampouras
 */
public class MeaningRepresentation implements Serializable {
    private String predicate;
    private HashMap<String, HashSet<String>> arguments;
    private HashMap<String, String> delexMap = new HashMap<>();
    
    String MRstr = "";

    /**
     *
     * @param predicate
     * @param arguments
     * @param MRstr
     */
    public MeaningRepresentation(String predicate, HashMap<String, HashSet<String>> arguments, String MRstr) {
        this.predicate = predicate;
        this.arguments = arguments;
        this.MRstr = MRstr;
    }

    /**
     *
     * @return
     */
    public String getPredicate() {
        return predicate;
    }

    /**
     *
     * @return
     */
    public HashMap<String, HashSet<String>> getAttributes() {
        return arguments;
    }

    String abstractMR = "";

    /**
     *
     * @return
     */
    public String getAbstractMR() {
        if (abstractMR.isEmpty()) {
            abstractMR = predicate + ":";
            ArrayList<String> attrs = new ArrayList<>(arguments.keySet());
            Collections.sort(attrs);
            HashMap<String, Integer> xCounts = new HashMap<>();
            for (String attr : attrs) {
                xCounts.put(attr, 0);
            }
            for (String attr : attrs) {
                abstractMR += attr + "={";

                ArrayList<String> values = new ArrayList<>(arguments.get(attr));
                Collections.sort(values);
                for (String value : values) {
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
                }            
                abstractMR += "}";
            }
        }
        return abstractMR;
    }

    /**
     *
     * @return
     */
    public String getMRstr() {
        return MRstr;
    }

    /**
     *
     * @return
     */
    @Override
    public int hashCode() {
        int hash = 7;
        hash = 61 * hash + Objects.hashCode(this.predicate);
        hash = 61 * hash + Objects.hashCode(this.arguments);
        return hash;
    }

    /**
     *
     * @param obj
     * @return
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
        if (!this.arguments.equals(other.arguments)) {
            return false;
        }
        return true;
    }

    /**
     *
     * @return
     */
    public HashMap<String, String> getDelexMap() {
        return delexMap;
    }

    /**
     *
     * @param delexMap
     */
    public void setDelexMap(HashMap<String, String> delexMap) {
        this.delexMap = delexMap;
    }
}
