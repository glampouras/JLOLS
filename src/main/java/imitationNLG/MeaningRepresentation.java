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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;

public class MeaningRepresentation {
    private String predicate;
    private HashMap<String, HashSet<String>> arguments;
    private HashMap<String, String> delexMap = new HashMap<>();
    
    String MRstr = "";

    public MeaningRepresentation(String predicate, HashMap<String, HashSet<String>> arguments, String MRstr) {
        this.predicate = predicate;
        this.arguments = arguments;
        this.MRstr = MRstr;
    }

    public String getPredicate() {
        return predicate;
    }

    public HashMap<String, HashSet<String>> getAttributes() {
        return arguments;
    }

    public String getMRstr() {
        return MRstr;
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

    public HashMap<String, String> getDelexMap() {
        return delexMap;
    }

    public void setDelexMap(HashMap<String, String> delexMap) {
        this.delexMap = delexMap;
    }
}
