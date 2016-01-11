package similarity_measures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

public class Levenshtein {

    public static double getSimilarity(ArrayList<String> ar1, ArrayList<String> ar2) {

        int s1Length = ar1.size();
        int s2Length = ar2.size();

        if (s1Length == 0) {
            return 0.0;
        }
        if (s2Length == 0) {
            return 0.0;
        }

        int[][] d = new int[s1Length + 1][s2Length + 1];

        for (int i = 0; i <= s1Length; i++) {
            d[i][0] = i;
        }
        for (int j = 0; j <= s2Length; j++) {
            d[0][j] = j;
        }

        String token1 = "";
        String token2 = "";
        int cost = 0;
        for (int i = 1; i <= s1Length; i++) {
            token1 = ar1.get(i - 1);
            for (int j = 1; j <= s2Length; j++) {
                token2 = ar2.get(j - 1);
                cost = (token1.equals(token2)) ? 0 : 1;
                d[i][j] = MathFunctions.minOf3(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
            }
        }

        int maxLength = (s1Length > s2Length) ? s1Length : s2Length;
        double similarity = 1.0 - ((double) d[s1Length][s2Length] / (double) maxLength);
        return similarity;
    }

    public static double getSimilarity(String s1, String s2, boolean perChar, boolean greek) {
        if (greek) {
            s1 = StringManipulation.normalizeGreek(s1);
            s2 = StringManipulation.normalizeGreek(s2);
        }
        s1 = s1.toUpperCase().replaceAll("\\s", " ");
        s2 = s2.toUpperCase().replaceAll("\\s", " ");

        if (perChar) {
            int s1Length = s1.length();
            int s2Length = s2.length();

            if (s1Length == 0) {
                return 0.0;
            }
            if (s2Length == 0) {
                return 0.0;
            }

            int[][] d = new int[s1Length + 1][s2Length + 1];

            for (int i = 0; i <= s1Length; i++) {
                d[i][0] = i;
            }
            for (int j = 0; j <= s2Length; j++) {
                d[0][j] = j;
            }

            char ch1 = ' ';
            char ch2 = ' ';
            int cost = 0;
            for (int i = 1; i <= s1Length; i++) {
                ch1 = s1.charAt(i - 1);
                for (int j = 1; j <= s2Length; j++) {
                    ch2 = s2.charAt(j - 1);
                    cost = (ch1 == ch2) ? 0 : 1;
                    d[i][j] = MathFunctions.minOf3(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
                }
            }

            int maxLength = (s1Length > s2Length) ? s1Length : s2Length;
            double similarity = 1.0 - ((double) d[s1Length][s2Length] / (double) maxLength);
            return similarity;
        } else {
            ArrayList<String> s1Tokens = StringManipulation.getTokensList(s1);
            ArrayList<String> s2Tokens = StringManipulation.getTokensList(s2);

            int s1Length = s1Tokens.size();
            int s2Length = s2Tokens.size();

            if (s1Length == 0) {
                return 0.0;
            }
            if (s2Length == 0) {
                return 0.0;
            }

            int[][] d = new int[s1Length + 1][s2Length + 1];

            for (int i = 0; i <= s1Length; i++) {
                d[i][0] = i;
            }
            for (int j = 0; j <= s2Length; j++) {
                d[0][j] = j;
            }

            String token1 = "";
            String token2 = "";
            int cost = 0;
            for (int i = 1; i <= s1Length; i++) {
                token1 = s1Tokens.get(i - 1);
                for (int j = 1; j <= s2Length; j++) {
                    token2 = s2Tokens.get(j - 1);
                    cost = (token1.equals(token2)) ? 0 : 1;
                    d[i][j] = MathFunctions.minOf3(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
                }
            }

            int maxLength = (s1Length > s2Length) ? s1Length : s2Length;
            double similarity = 1.0 - ((double) d[s1Length][s2Length] / (double) maxLength);
            return similarity;
        }
    }

    public static double getNormDistance(String s1, String s2, int refSize) {
        s1 = s1.toUpperCase().replaceAll("\\s", " ");
        s2 = s2.toUpperCase().replaceAll("\\s", " ");

        ArrayList<String> s1Tokens = StringManipulation.getTokensList(s1);
        ArrayList<String> s2Tokens = StringManipulation.getTokensList(s2);

        int s1Length = s1Tokens.size();
        int s2Length = s2Tokens.size();

        if (s1Length == 0) {
            return 1.0;
        }
        if (s2Length == 0) {
            return 1.0;
        }

        return getNormDistance(s1Tokens, s2Tokens, (double) refSize * 2);
    }

    public static double getNormDistance(String s1, String s2) {
        s1 = s1.toUpperCase().replaceAll("\\s", " ");
        s2 = s2.toUpperCase().replaceAll("\\s", " ");

        ArrayList<String> s1Tokens = StringManipulation.getTokensList(s1);
        ArrayList<String> s2Tokens = StringManipulation.getTokensList(s2);

        int s1Length = s1Tokens.size();
        int s2Length = s2Tokens.size();

        if (s1Length == 0) {
            return 1.0;
        }
        if (s2Length == 0) {
            return 1.0;
        }

        return getNormDistance(s1Tokens, s2Tokens, (double) (1));
    }

    public static double getNormDistance(ArrayList<String> s1Tokens, ArrayList<String> s2Tokens, double normSizeDenum) {
        int s1Length = s1Tokens.size();
        int s2Length = s2Tokens.size();

        int[][] d = new int[s1Length + 1][s2Length + 1];

        for (int i = 0; i <= s1Length; i++) {
            d[i][0] = i;
        }
        for (int j = 0; j <= s2Length; j++) {
            d[0][j] = j;
        }

        String token1 = "";
        String token2 = "";
        int cost = 0;
        for (int i = 1; i <= s1Length; i++) {
            token1 = s1Tokens.get(i - 1);
            for (int j = 1; j <= s2Length; j++) {
                token2 = s2Tokens.get(j - 1);
                cost = (token1.equals(token2)) ? 0 : 2;
                d[i][j] = MathFunctions.minOf3(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
            }
        }
        return (double) (((double) d[s1Length][s2Length]) / (normSizeDenum));
    }

    public static int getDistance(String s1, String s2) {
        s1 = s1.toUpperCase().replaceAll("\\s", " ");
        s2 = s2.toUpperCase().replaceAll("\\s", " ");

        ArrayList<String> s1Tokens = StringManipulation.getTokensList(s1);
        ArrayList<String> s2Tokens = StringManipulation.getTokensList(s2);

        int s1Length = s1Tokens.size();
        int s2Length = s2Tokens.size();

        int[][] d = new int[s1Length + 1][s2Length + 1];

        for (int i = 0; i <= s1Length; i++) {
            d[i][0] = i;
        }
        for (int j = 0; j <= s2Length; j++) {
            d[0][j] = j;
        }

        String token1 = "";
        String token2 = "";
        int cost = 0;
        for (int i = 1; i <= s1Length; i++) {
            token1 = s1Tokens.get(i - 1);
            for (int j = 1; j <= s2Length; j++) {
                token2 = s2Tokens.get(j - 1);
                cost = (token1.equals(token2)) ? 0 : 1;
                d[i][j] = MathFunctions.minOf3(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
            }
        }

        return d[s1Length][s2Length];
    }

    //CAN ONLY ADD, NOT DELERE ARE SUBSTITUTE    
    public static double getIncreasingDistance(String s1, String s2) {
        s1 = s1.toUpperCase().replaceAll("\\s", " ");
        s2 = s2.toUpperCase().replaceAll("\\s", " ");

        ArrayList<String> s1Tokens = StringManipulation.getTokensList(s1);
        ArrayList<String> s2Tokens = StringManipulation.getTokensList(s2);

        int s1Length = s1Tokens.size();
        int s2Length = s2Tokens.size();

        if (s1Length == 0) {
            return 0.0;
        }
        if (s2Length == 0) {
            return 0.0;
        }

        int[][] d = new int[s1Length + 1][s2Length + 1];

        for (int i = 0; i <= s1Length; i++) {
            d[i][0] = i;
        }
        for (int j = 0; j <= s2Length; j++) {
            d[0][j] = j;
        }

        String token1 = "";
        String token2 = "";
        int cost = 0;
        for (int i = 1; i <= s1Length; i++) {
            token1 = s1Tokens.get(i - 1);
            for (int j = 1; j <= s2Length; j++) {
                token2 = s2Tokens.get(j - 1);
                cost = (token1.equals(token2)) ? 0 : 2 * (i + j - 1) / (s1Length + s2Length);
                d[i][j] = MathFunctions.minOf3(d[i - 1][j] + 1 * (i - 1 + j) / (s1Length + s2Length), d[i][j - 1] + 1 * (i + j - 1) / (s1Length + s2Length), d[i - 1][j - 1] + cost);
            }
        }

        return (double) ((double) d[s1Length][s2Length] / (double) (s1Length + s2Length));
    }

    public static double getSimilarity(String s1, String s2) {
        s1 = s1.toLowerCase().replaceAll("\\s", " ");
        s2 = s2.toLowerCase().replaceAll("\\s", " ");

        int s1Length = s1.length();
        int s2Length = s2.length();

        if (s1Length == 0) {
            return 0.0;
        }
        if (s2Length == 0) {
            return 0.0;
        }

        boolean pass = false;
        if (s1Length < 2 || s2Length < 2) {
            pass = true;
        }
        for (int i = 1; i <= s1Length - 1; i++) {
            if (!pass) {
                String sub1 = s1.substring(i - 1, i + 1);
                for (int j = 1; j <= s2Length - 1; j++) {
                    if (!pass) {
                        String sub2 = s2.substring(j - 1, j + 1);
                        if (sub1.equalsIgnoreCase(sub2)) {
                            pass = true;
                        }
                    }
                }
            }
        }

        if (pass) {
            int[][] d = new int[s1Length + 1][s2Length + 1];

            for (int i = 0; i <= s1Length; i++) {
                d[i][0] = i;
            }
            for (int j = 0; j <= s2Length; j++) {
                d[0][j] = j;
            }

            char ch1 = ' ';
            char ch2 = ' ';
            int cost = 0;
            for (int i = 1; i <= s1Length; i++) {
                ch1 = s1.charAt(i - 1);
                for (int j = 1; j <= s2Length; j++) {
                    ch2 = s2.charAt(j - 1);
                    cost = (ch1 == ch2) ? 0 : 1;
                    d[i][j] = MathFunctions.minOf3(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
                }
            }

            int maxLength = (s1Length > s2Length) ? s1Length : s2Length;
            double similarity = 1.0 - (d[s1Length][s2Length] / ((double) maxLength));
            return similarity;
        }
        return 0.0;
    }

    public static Object[] getMatchedSimilarity(ArrayList<String> s1Tokens, ArrayList<String> s2Tokens, ArrayList<Double> tokenImportance) {
        if (s1Tokens.isEmpty() || s2Tokens.isEmpty()) {
            Object[] results = new Object[4];
            results[0] = 0.0;
            results[1] = 0.0;
            results[2] = null;
            results[3] = -1;
            return results;
        }

        ArrayList<Double> matchImportance = new ArrayList<Double>();
        for (String token : s2Tokens) {
            matchImportance.add(1.0);
        }
        HashMap<int[], Double> matches = new HashMap<int[], Double>();
        int maxLength = (s1Tokens.size() > s2Tokens.size()) ? s1Tokens.size() : s2Tokens.size();

        HashSet<Integer> matchedIndeces1 = new HashSet<Integer>();
        HashSet<Integer> matchedIndeces2 = new HashSet<Integer>();
        for (int m = 0; m < maxLength; m++) {
            int[] maxMatch = null;
            double maxSimilarity = -1.0;
            for (int i = 0; i < s1Tokens.size(); i++) {
                if (!matchedIndeces1.contains(i)) {
                    String token1 = s1Tokens.get(i);
                    for (int j = 0; j < s2Tokens.size(); j++) {
                        if (!matchedIndeces2.contains(j)) {
                            String token2 = s2Tokens.get(j);

                            double similarity;
                            if (token1.equalsIgnoreCase("$num$") || token2.equalsIgnoreCase("$num$")) {
                                similarity = (token1.equalsIgnoreCase(token2)) ? 1.0 : 0.0;
                            } else {
                                similarity = getSimilarity(token1, token2);
                            }

                            if (similarity > maxSimilarity) {
                                maxSimilarity = similarity;

                                maxMatch = new int[2];
                                maxMatch[0] = i;
                                maxMatch[1] = j;
                            }
                        }
                    }
                }
            }
            if (maxSimilarity >= 0.0 && maxMatch != null) {
                matchedIndeces1.add(maxMatch[0]);
                matchedIndeces2.add(maxMatch[1]);
                matches.put(maxMatch, maxSimilarity);

                Double importance = 0.25;
                if (maxMatch[0] < tokenImportance.size()) {
                    importance = tokenImportance.get(maxMatch[0]);
                }

                matchImportance.set(maxMatch[1], importance);
            }
        }

        double totalSimilarity = 0.0;
        /*for (int[] match : matches.keySet()) {
         double weight = entityFrequencies.getEntityNameImportance().get(match[0]);
         Double similarity = matches.get(match);
            
         totalSimilarity += weight * similarity;
         }*/
        for (Double similarity : matches.values()) {
            totalSimilarity += similarity;
        }
        double finalSimilarity = totalSimilarity / (double) maxLength;

        double matchCrosses = 0.0;
        for (int[] match1 : matches.keySet()) {
            for (int[] match2 : matches.keySet()) {
                if (!Arrays.equals(match1, match2)) {
                    if ((match1[0] < match2[0] && match1[1] > match2[1])
                            || (match1[0] > match2[0] && match1[1] < match2[1])) {
                        matchCrosses += 1.0;
                    }
                }
            }
        }

        //System.out.println(s1Tokens + "<>" + s2Tokens + " = " + finalSimilarity);
        Object[] results = new Object[4];
        results[0] = finalSimilarity;
        results[1] = matchCrosses;
        results[2] = matchImportance;
        results[3] = -1;
        return results;
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

        int s1Length = s1Tokens.size();
        int s2Length = s2Tokens.size();

        if (s1Length == 0) {
            return 0.0;
        }
        if (s2Length == 0) {
            return 0.0;
        }

        int[][] d = new int[s1Length + 1][s2Length + 1];

        for (int i = 0; i <= s1Length; i++) {
            d[i][0] = i;
        }
        for (int j = 0; j <= s2Length; j++) {
            d[0][j] = j;
        }

        String token1 = "";
        String token2 = "";
        int cost = 0;
        for (int i = 1; i <= s1Length; i++) {
            token1 = s1Tokens.get(i - 1);
            for (int j = 1; j <= s2Length; j++) {
                token2 = s2Tokens.get(j - 1);
                cost = (token1.equals(token2)) ? 0 : 1;
                d[i][j] = MathFunctions.minOf3(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
            }
        }

        int maxLength = (s1Length > s2Length) ? s1Length : s2Length;
        double similarity = 1.0 - ((double) d[s1Length][s2Length] / (double) maxLength);
        return similarity;
    }

    public static double getNGramDistance(String s1, String s2, int n, boolean greek) {
        if (greek) {
            s1 = StringManipulation.normalizeGreek(s1);
            s2 = StringManipulation.normalizeGreek(s2);
        }
        s1 = s1.toUpperCase().replaceAll("\\s", " ");
        s2 = s2.toUpperCase().replaceAll("\\s", " ");

        ArrayList<String> s1Tokens = StringManipulation.getNGrams(s1, n);
        ArrayList<String> s2Tokens = StringManipulation.getNGrams(s2, n);

        int s1Length = s1Tokens.size();
        int s2Length = s2Tokens.size();

        if (s1Length == 0) {
            return 0.0;
        }
        if (s2Length == 0) {
            return 0.0;
        }

        int[][] d = new int[s1Length + 1][s2Length + 1];

        for (int i = 0; i <= s1Length; i++) {
            d[i][0] = i;
        }
        for (int j = 0; j <= s2Length; j++) {
            d[0][j] = j;
        }

        String token1 = "";
        String token2 = "";
        int cost = 0;
        for (int i = 1; i <= s1Length; i++) {
            token1 = s1Tokens.get(i - 1);
            for (int j = 1; j <= s2Length; j++) {
                token2 = s2Tokens.get(j - 1);
                cost = (token1.equals(token2)) ? 0 : 1;
                d[i][j] = MathFunctions.minOf3(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
            }
        }

        return (double) d[s1Length][s2Length];
    }

    public void test() {

    }
}
