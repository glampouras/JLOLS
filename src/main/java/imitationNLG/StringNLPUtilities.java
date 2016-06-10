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

import java.util.ArrayList;

public class StringNLPUtilities {

    public static final String PREPOSITION_EN_ABOARD = "aboard";
    public static final String PREPOSITION_EN_ABOUT = "about";
    public static final String PREPOSITION_EN_ABOVE = "above";
    public static final String PREPOSITION_EN_ACROSS = "across";
    public static final String PREPOSITION_EN_AFTER = "after";
    public static final String PREPOSITION_EN_AGAINST = "against";
    public static final String PREPOSITION_EN_ALONG = "along";
    public static final String PREPOSITION_EN_AMID = "amid";
    public static final String PREPOSITION_EN_AMONG = "among";
    public static final String PREPOSITION_EN_ANTI = "anti";
    public static final String PREPOSITION_EN_AROUND = "around";
    public static final String PREPOSITION_EN_AS = "as";
    public static final String PREPOSITION_EN_AT = "at";
    public static final String PREPOSITION_EN_BEFORE = "before";
    public static final String PREPOSITION_EN_BEHIND = "behind";
    public static final String PREPOSITION_EN_BELOW = "below";
    public static final String PREPOSITION_EN_BENEATH = "beneath";
    public static final String PREPOSITION_EN_BESIDE = "beside";
    public static final String PREPOSITION_EN_BETWEEN = "between";
    public static final String PREPOSITION_EN_BEYOND = "beyond";
    public static final String PREPOSITION_EN_BUT = "but";
    public static final String PREPOSITION_EN_BY = "by";
    public static final String PREPOSITION_EN_DESPITE = "despite";
    public static final String PREPOSITION_EN_DOWN = "down";
    public static final String PREPOSITION_EN_DURING = "during";
    public static final String PREPOSITION_EN_EXCEPT = "except";
    public static final String PREPOSITION_EN_FOR = "for";
    public static final String PREPOSITION_EN_FROM = "from";
    public static final String PREPOSITION_EN_IN = "in";
    public static final String PREPOSITION_EN_INSIDE = "inside";
    public static final String PREPOSITION_EN_INTO = "into";
    public static final String PREPOSITION_EN_LIKE = "like";
    public static final String PREPOSITION_EN_MINUS = "minus";
    public static final String PREPOSITION_EN_NEAR = "near";
    public static final String PREPOSITION_EN_OF = "of";
    public static final String PREPOSITION_EN_OFF = "off";
    public static final String PREPOSITION_EN_ON = "on";
    public static final String PREPOSITION_EN_ONTO = "onto";
    public static final String PREPOSITION_EN_OPPOSITE = "opposite";
    public static final String PREPOSITION_EN_OUTSIDE = "outside";
    public static final String PREPOSITION_EN_OVER = "over";
    public static final String PREPOSITION_EN_PAST = "past";
    public static final String PREPOSITION_EN_PER = "per";
    public static final String PREPOSITION_EN_PLUS = "plus";
    public static final String PREPOSITION_EN_ROUND = "round";
    public static final String PREPOSITION_EN_SAVE = "save";
    public static final String PREPOSITION_EN_SINCE = "since";
    public static final String PREPOSITION_EN_THAN = "than";
    public static final String PREPOSITION_EN_THROUGH = "through";
    public static final String PREPOSITION_EN_TO = "to";
    public static final String PREPOSITION_EN_TOWARD = "toward";
    public static final String PREPOSITION_EN_TOWARDS = "towards";
    public static final String PREPOSITION_EN_UNDER = "under";
    public static final String PREPOSITION_EN_UNLIKE = "unlike";
    public static final String PREPOSITION_EN_UNTIL = "until";
    public static final String PREPOSITION_EN_UP = "up";
    public static final String PREPOSITION_EN_UPON = "upon";
    public static final String PREPOSITION_EN_VIA = "via";
    public static final String PREPOSITION_EN_WITH = "with";
    public static final String PREPOSITION_EN_WITHIN = "within";
    public static final String PREPOSITION_EN_WITHOUT = "without";

    public static ArrayList<String> prepositionList = null;

    public static ArrayList<String> getEnglishPrepositionList() {
        if (prepositionList == null) {
            prepositionList = new ArrayList<String>();

            prepositionList.add(PREPOSITION_EN_ABOARD);
            prepositionList.add(PREPOSITION_EN_ABOUT);
            prepositionList.add(PREPOSITION_EN_ABOVE);
            prepositionList.add(PREPOSITION_EN_ACROSS);
            prepositionList.add(PREPOSITION_EN_AFTER);
            prepositionList.add(PREPOSITION_EN_AGAINST);
            prepositionList.add(PREPOSITION_EN_ALONG);
            prepositionList.add(PREPOSITION_EN_AMID);
            prepositionList.add(PREPOSITION_EN_AMONG);
            prepositionList.add(PREPOSITION_EN_ANTI);
            prepositionList.add(PREPOSITION_EN_AROUND);
            prepositionList.add(PREPOSITION_EN_AS);
            prepositionList.add(PREPOSITION_EN_AT);
            prepositionList.add(PREPOSITION_EN_BEFORE);
            prepositionList.add(PREPOSITION_EN_BEHIND);
            prepositionList.add(PREPOSITION_EN_BELOW);
            prepositionList.add(PREPOSITION_EN_BENEATH);
            prepositionList.add(PREPOSITION_EN_BESIDE);
            prepositionList.add(PREPOSITION_EN_BETWEEN);
            prepositionList.add(PREPOSITION_EN_BEYOND);
            prepositionList.add(PREPOSITION_EN_BUT);
            prepositionList.add(PREPOSITION_EN_BY);
            prepositionList.add(PREPOSITION_EN_DESPITE);
            prepositionList.add(PREPOSITION_EN_DOWN);
            prepositionList.add(PREPOSITION_EN_DURING);
            prepositionList.add(PREPOSITION_EN_EXCEPT);
            prepositionList.add(PREPOSITION_EN_FOR);
            prepositionList.add(PREPOSITION_EN_FROM);
            prepositionList.add(PREPOSITION_EN_IN);
            prepositionList.add(PREPOSITION_EN_INSIDE);
            prepositionList.add(PREPOSITION_EN_INTO);
            prepositionList.add(PREPOSITION_EN_LIKE);
            prepositionList.add(PREPOSITION_EN_MINUS);
            prepositionList.add(PREPOSITION_EN_NEAR);
            prepositionList.add(PREPOSITION_EN_OF);
            prepositionList.add(PREPOSITION_EN_OFF);
            prepositionList.add(PREPOSITION_EN_ON);
            prepositionList.add(PREPOSITION_EN_ONTO);
            prepositionList.add(PREPOSITION_EN_OPPOSITE);
            prepositionList.add(PREPOSITION_EN_OUTSIDE);
            prepositionList.add(PREPOSITION_EN_OVER);
            prepositionList.add(PREPOSITION_EN_PAST);
            prepositionList.add(PREPOSITION_EN_PER);
            prepositionList.add(PREPOSITION_EN_PLUS);
            prepositionList.add(PREPOSITION_EN_ROUND);
            prepositionList.add(PREPOSITION_EN_SAVE);
            prepositionList.add(PREPOSITION_EN_SINCE);
            prepositionList.add(PREPOSITION_EN_THAN);
            prepositionList.add(PREPOSITION_EN_THROUGH);
            prepositionList.add(PREPOSITION_EN_TO);
            prepositionList.add(PREPOSITION_EN_TOWARD);
            prepositionList.add(PREPOSITION_EN_TOWARDS);
            prepositionList.add(PREPOSITION_EN_UNDER);
            prepositionList.add(PREPOSITION_EN_UNLIKE);
            prepositionList.add(PREPOSITION_EN_UNTIL);
            prepositionList.add(PREPOSITION_EN_UP);
            prepositionList.add(PREPOSITION_EN_UPON);
            prepositionList.add(PREPOSITION_EN_VIA);
            prepositionList.add(PREPOSITION_EN_WITH);
            prepositionList.add(PREPOSITION_EN_WITHIN);
            prepositionList.add(PREPOSITION_EN_WITHOUT);
        }

        return prepositionList;
    }

    public static boolean isArticle(String s) {
        if (s.equals("the") || s.equals("a") || s.equals("an")) {
            return true;
        }
        return false;
    }

    public static boolean isPreposition(String s) {
        if (getEnglishPrepositionList().contains(s)) {
            return true;
        }
        return false;
    }
}
