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

import java.util.HashSet;
import java.util.Objects;

public class Action {

    private String word;
    private String attribute;

    HashSet<String> attrValuesStillToBeMentionedAtThisPoint;

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

    public HashSet<String> getAttrValuesStillToBeMentionedAtThisPoint() {
        return attrValuesStillToBeMentionedAtThisPoint;
    }

    public void setAttrValuesStillToBeMentionedAtThisPoint(HashSet<String> attrValuesStillToBeMentionedAtThisPoint) {
        this.attrValuesStillToBeMentionedAtThisPoint = new HashSet<>(attrValuesStillToBeMentionedAtThisPoint);
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
