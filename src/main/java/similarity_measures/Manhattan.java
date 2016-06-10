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
package similarity_measures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Manhattan {

    public static double getSimilarity(String s1, String s2, boolean greek) {
        if (greek) {
            s1 = StringManipulation.normalizeGreek(s1);
            s2 = StringManipulation.normalizeGreek(s2);
        }
        s1 = s1.toUpperCase().replaceAll("\\s", " ");
        s2 = s2.toUpperCase().replaceAll("\\s", " ");

        ArrayList<String> s1Tokens = StringManipulation.getTokensList(s1);
        ArrayList<String> s2Tokens = StringManipulation.getTokensList(s2);

        int maxPossibleTokens = s1Tokens.size() + s2Tokens.size();

        HashSet<String> allTokens = new HashSet<String>();
        allTokens.addAll(s1Tokens);
        allTokens.addAll(s2Tokens);

        HashMap<String, Integer> x = StringManipulation.getTokensMap(s1Tokens);
        HashMap<String, Integer> y = StringManipulation.getTokensMap(s2Tokens);

        int totalDistance = 0;

        for (String token : allTokens) {
            if (!x.containsKey(token)) {
                totalDistance += y.get(token);
            } else if (!y.containsKey(token)) {
                totalDistance += x.get(token);
            } else {
                int distance = x.get(token) - y.get(token);
                totalDistance += (distance > 0) ? distance : -distance;
            }
        }
        double similarity = 1.0 - ((double) totalDistance / (double) maxPossibleTokens);
        return similarity;
    }

    public static double getNGramSimilarity(String s1, String s2, int n, boolean greek) {
        if (greek) {
            s1 = StringManipulation.normalizeGreek(s1);
            s2 = StringManipulation.normalizeGreek(s2);
        }
        s1 = s1.toUpperCase().replaceAll("\\s", " ");
        s2 = s2.toUpperCase().replaceAll("\\s", " ");

        ArrayList<String> s1Tokens = StringManipulation.getNGrams(s1, n);
        ArrayList<String> s2Tokens = StringManipulation.getNGrams(s2, n);

        int maxPossibleTokens = s1Tokens.size() + s2Tokens.size();

        HashSet<String> allTokens = new HashSet<String>();
        allTokens.addAll(s1Tokens);
        allTokens.addAll(s2Tokens);

        HashMap<String, Integer> x = StringManipulation.getTokensMap(s1Tokens);
        HashMap<String, Integer> y = StringManipulation.getTokensMap(s2Tokens);

        int totalDistance = 0;

        for (String token : allTokens) {
            if (!x.containsKey(token)) {
                totalDistance += y.get(token);
            } else if (!y.containsKey(token)) {
                totalDistance += x.get(token);
            } else {
                int distance = x.get(token) - y.get(token);
                totalDistance += (distance > 0) ? distance : -distance;
            }
        }
        double similarity = 1.0 - ((double) totalDistance / (double) maxPossibleTokens);
        return similarity;
    }

    //Used in reordered summary words
    public static double getSimilarity(ArrayList<String> ar1, ArrayList<String> ar2) {
        int maxPossibleTokens = ar1.size() + ar2.size();

        HashSet<String> allTokens = new HashSet<String>();
        allTokens.addAll(ar1);
        allTokens.addAll(ar2);

        int totalDistance = 0;

        HashMap<String, Integer> x = StringManipulation.getTokensMap(ar1);
        HashMap<String, Integer> y = StringManipulation.getTokensMap(ar2);

        String sentence;
        for (int i = 0; i < ar1.size(); i++) {
            sentence = ar1.get(i);
            if (!ar2.contains(sentence)) {
                totalDistance += i;
            } else {
                for (int j = 0; j < ar2.size(); j++) {
                    if (ar2.get(j).equals(sentence)) {
                        int distance = i - j;
                        totalDistance += (distance > 0) ? distance : -distance;
                    }
                }
            }
        }
        double similarity = 1.0 - ((double) totalDistance / (double) maxPossibleTokens);
        return similarity;
    }

    //Used in reordered summary words
    //Only calculates the number of sentences not in the same position
    public static double getSimpleSimilarity(ArrayList<String> ar1, ArrayList<String> ar2) {
        int totalDistance = 0;

        for (int i = 0; i < ar2.size(); i++) {
            if (!ar1.get(i).equals(ar2.get(i))) {
                totalDistance++;
            }
        }
        double similarity = 1.0 - ((double) totalDistance / (double) ar1.size());
        return similarity;
    }
}
