package similarity_measures;

import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.WordTokenFactory;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;

public class StringManipulation {

    public static String normalizeGreek(String s1) {
        s1 = s1.replaceAll("�", "�");
        s1 = s1.replaceAll("�", "�");
        s1 = s1.toUpperCase();
        String normalized = s1.replaceAll("�", "�").replaceAll("�", "�").replaceAll("�", "�").replaceAll("�", "�").replaceAll("�", "�").replaceAll("�", "�").replaceAll("�", "�").replaceAll("�", "�").replaceAll("�", "�");
        return normalized;
    }

    public static ArrayList<String> getTokensList(String s1) {
        ArrayList<String> tokens = new ArrayList<String>();
        String[] s1Tokens = s1.split(" ");
        for (String s1Token : s1Tokens) {
            if (!s1Token.trim().isEmpty()) {
                tokens.add(s1Token);
            }
        }
        return tokens;
    }

    public static ArrayList<String> getTokensListAndReplaceNumbers(String s1) {
        ArrayList<String> tokens = new ArrayList<String>();

        //String[] s1Tokens = s1.split(" ");
        String options = "tokenizeNLs=false,americanize=false,ptb3Escaping=false,normalizeCurrency=false,normalizeFractions=false,normalizeParentheses=false,normalizeOtherBrackets=false,asciiQuotes=false,latexQuotes=false,unicodeQuotes=false,ptb3Ellipsis=false,unicodeEllipsis=false,ptb3Dashes=false,escapeForwardSlashAsterisk=false,untokenizable=noneDelete";
        PTBTokenizer tokenizer = new PTBTokenizer(new StringReader(s1), new WordTokenFactory(), options);

        boolean previousTokenIsNumber = false;
        for (Word label; tokenizer.hasNext();) {
            label = (Word) tokenizer.next();
        //for (String s1Token : s1Tokens) {
            if (!label.value().trim().isEmpty()) {
                if (label.value().replaceAll("\\p{Punct}", "").replaceAll("\\p{Space}", "").trim().matches("(?=[^A-Za-z]+$).*[0-9].*")) {
                    if (!previousTokenIsNumber) {
                        tokens.add("$num$");
                    }
                    previousTokenIsNumber = true;
                } else {
                    previousTokenIsNumber = false;
                    tokens.add(label.value().trim());
                }
            }
        }
        return tokens;
    }

    public static HashMap<String, Integer> getTokensMap(ArrayList<String> s1Tokens) {
        HashMap<String, Integer> tokens = new HashMap<String, Integer>();
        for (String s1Token : s1Tokens) {
            if (!tokens.containsKey(s1Token)) {
                tokens.put(s1Token, 1);
            } else {
                tokens.put(s1Token, tokens.get(s1Token) + 1);
            }
        }
        return tokens;
    }

    public static ArrayList<String> getNGrams(String s1, int n) {
        ArrayList<String> nGrams = new ArrayList<String>();
        StringBuffer nGram = new StringBuffer();
        for (int i = 0 - n + 1; i < s1.length(); i++) {
            nGram = new StringBuffer();
            for (int j = 0; j < n; j++) {
                if (i + j < 0) {
                    nGram.append("#");
                } else if (i + j >= s1.length()) {
                    nGram.append("%");
                } else {
                    nGram.append(s1.charAt(i + j));
                }
            }
            nGrams.add(nGram.toString());
        }
        /*for (String ngram : nGrams)
        {
        System.out.println(ngram);
        }*/
        return nGrams;
    }

    /*public static ArrayList<String> getNGrams(ArrayList<String> s1, int n)
    {
    ArrayList<String> nGrams = new ArrayList<String>();
    StringBuffer nGram = new StringBuffer();
    for(int i=0 - n + 1;i<s1.size(); i++)
    {
    nGram = new StringBuffer();
    for(int j=0;j<n;j++)
    {
    if(i+j<0)
    {
    nGram.append("#");
    }
    else if(i+j>=s1.size())
    {
    nGram.append("%");
    }
    else
    {
    nGram.append(s1.get(i+j));
    }
    }
    nGrams.add(nGram.toString());
    }
    for (String ngram : nGrams)
    {
    System.out.println(ngram);
    }
    return nGrams;
    }*/
    public static String getSoundex(String s1, int soundexLength, boolean greek) {
        if (greek) {
            return s1;
        }
        String tmpStr;
        String wordStr;
        char curChar;
        char lastChar;
        final int wsLen;
        final char firstLetter;

        //ensure soundexLen is in a valid range
        if (soundexLength > 10) {
            soundexLength = 10;
        }
        if (soundexLength < 4) {
            soundexLength = 4;
        }

        //check for empty input
        if (s1.length() == 0) {
            return ("");
        }

        //remove case
        s1 = s1.toUpperCase();

        /* Clean and tidy
         */
        wordStr = s1;
        wordStr = wordStr.replaceAll("[^A-Z]", " "); // rpl non-chars w space
        wordStr = wordStr.replaceAll("\\s+", "");   // remove spaces

        //check for empty input again the previous clean and tidy could of shrunk it to zero.
        if (wordStr.length() == 0) {
            return ("");
        }

        /* The above improvements
         * may change this first letter
         */
        firstLetter = wordStr.charAt(0);

        // uses the assumption that enough valid characters are in the first 4 times the soundex required length
        if (wordStr.length() > (6 * 4) + 1) {
            wordStr = "-" + wordStr.substring(1, 6 * 4);
        } else {
            wordStr = "-" + wordStr.substring(1);
        }
        // Begin Classic SoundEx
		/*
        1) B,P,F,V
        2) C,S,K,G,J,Q,X,Z
        3) D,T
        4) L
        5) M,N
        6) R
         */
        wordStr = wordStr.replaceAll("[AEIOUWH]", "0");
        wordStr = wordStr.replaceAll("[BPFV]", "1");
        wordStr = wordStr.replaceAll("[CSKGJQXZ]", "2");
        wordStr = wordStr.replaceAll("[DT]", "3");
        wordStr = wordStr.replaceAll("[L]", "4");
        wordStr = wordStr.replaceAll("[MN]", "5");
        wordStr = wordStr.replaceAll("[R]", "6");

        // Remove extra equal adjacent digits
        wsLen = wordStr.length();
        lastChar = '-';
        tmpStr = "-";     /* replacing skipped first character */
        for (int i = 1; i < wsLen; i++) {
            curChar = wordStr.charAt(i);
            if (curChar != lastChar) {
                tmpStr += curChar;
                lastChar = curChar;
            }
        }
        wordStr = tmpStr;
        wordStr = wordStr.substring(1);          /* Drop first letter code   */
        wordStr = wordStr.replaceAll("0", "");  /* remove zeros             */
        wordStr += "000000000000000000";              /* pad with zeros on right  */
        wordStr = firstLetter + "-" + wordStr;      /* Add first letter of word */
        wordStr = wordStr.substring(0, soundexLength); /* size to taste     */
        return (wordStr);
    }

    public static String getCommonCharacters(String s1, String s2, int maxSeparatingDistance) {
        StringBuffer commonCharactersBuffer = new StringBuffer();
        StringBuffer s2Buffer = new StringBuffer(s2);
        char ch = ' ';
        for (int i = 0; i < s1.length(); i++) {
            ch = s1.charAt(i);
            boolean found = false;

            for (int j = Math.max(0, i - maxSeparatingDistance); j < Math.min(i + maxSeparatingDistance, s2.length() - 1) && !found; j++) {
                if (s2Buffer.charAt(j) == ch) {
                    found = true;
                    commonCharactersBuffer.append(ch);
                    s2Buffer.setCharAt(j, (char) 0);
                }
            }
        }

        return commonCharactersBuffer.toString();
    }

    public static ArrayList<String> getCommonTokens(ArrayList<String> s1Tokens, ArrayList<String> s2Tokens, int maxSeparatingDistance) {
        ArrayList<String> commonTokens = new ArrayList<String>();
        ArrayList<String> s2TokensList = new ArrayList<String>(s2Tokens);
        String token = "";
        for (int i = 0; i < s1Tokens.size(); i++) {
            token = s1Tokens.get(i);
            boolean found = false;

            for (int j = Math.max(0, i - maxSeparatingDistance); j < Math.min(i + maxSeparatingDistance, s2Tokens.size() - 1) && !found; j++) {
                if (s2TokensList.get(j) != null) {
                    if (s2TokensList.get(j).equals(token)) {
                        found = true;
                        commonTokens.add(token);
                        s2TokensList.set(j, null);
                    }
                }
            }
        }

        return commonTokens;
    }

    public static int getCommonPrefixLength(String s1, String s2, int minCommonPrefixLength) {
        int commonPrefixLength = MathFunctions.minOf3(minCommonPrefixLength, s1.length(), s2.length());
        for (int i = 0; i < commonPrefixLength; i++) {
            if (s1.charAt(i) != s2.charAt(i)) {
                return i;
            }
        }
        return commonPrefixLength;
    }

    public static int getCommonPrefixLength(ArrayList<String> s1Tokens, ArrayList<String> s2Tokens, int minCommonPrefixLength) {
        int commonPrefixLength = MathFunctions.minOf3(minCommonPrefixLength, s1Tokens.size(), s2Tokens.size());
        for (int i = 0; i < commonPrefixLength; i++) {
            if (!s1Tokens.get(i).equals(s2Tokens.get(i))) {
                return i;
            }
        }
        return commonPrefixLength;
    }
}