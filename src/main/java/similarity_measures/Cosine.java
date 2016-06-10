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
import java.util.HashSet;

//M2
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.File;
import java.util.StringTokenizer;

import java.io.IOException;
import java.io.FileNotFoundException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class Cosine
{
    public static void main(String args[])
    {
        Cosine.initStopWords("Data\\smart_common_words.txt");

        String s1 = "It urged the  member states to show their '' necessary reactions to this preventive attitude of the European Parliament. '' U.S.A.";
        String s2 = "YOU ARE NOT GOING TO TELL ME WHAT TO DO  ";

        s1 = Cosine.cleanString(s1);
        s2 = Cosine.cleanString(s2);

        System.err.println(s1);
        //System.err.println(Cosine.getSimilarityMinusStopWords(s1, s2, false));
        System.err.println(Cosine.getTextLength(s1));

    }
    
        //M2
        static ArrayList<String> stopWords = null;
        
        //M2
        public static void initStopWords(String path)
        {
            File f = null;
            BufferedReader br = null;
            String str;

            try
            {
                f = new File(path);
            }
            catch (NullPointerException e)
            {
                System.err.println ("File not found.");
            }

            try
            {
                br = new BufferedReader(new InputStreamReader(new FileInputStream(f)));
            }
            catch ( FileNotFoundException e )
            {
                System.err.println("Error opening file!");
            }

            stopWords = new ArrayList();
            
            try
            {
                str = br.readLine();

                while (str!=null)
                {	
                    //System.out.println("|" + str.trim() + "|");
                    stopWords.add(str.trim());

                    str = br.readLine();
                }
            }
            catch (IOException e)
            {	
            }
            try
            {
                    br.close();
            }
            catch (IOException e)
            {	
                    System.err.println("Error closing file.");
            }
	}
        
        //M2
        public static String cleanString(String s)
        {
            char[] chars = s.toCharArray();
            
            for (int i = 1; i < chars.length - 2; i++)
            {
                if ((chars[i] == '.')&&(Character.isUpperCase(chars[i-1]))&&(Character.isUpperCase(chars[i+1]))&&(chars[i+2] == '.'))
                {
                    chars[i] = '-';
                } 
            }
            
            return new String(chars).toUpperCase().replaceAll("'S ", " ").replaceAll("-", "").replaceAll("\\p{Punct}", " ").replaceAll("\\s+", " ");
        }
        
        //M2
        public static int getTextLength(String text)
        {
            StringTokenizer st = new StringTokenizer(text);

            int counter = 0;
            while(st.hasMoreTokens())
            {
                st.nextToken();
                counter++;
            }
            
            //return st.countTokens();
            return counter;
        }
        
        //M2
        public static double getSimilarityMinusStopWords(String s1, String s2, boolean greek)
	{
            //Thread safe
            ArrayList<String> stopWordsCopy = new ArrayList(stopWords);
            //Thread safe
            
		if(greek)
		{
			s1 = StringManipulation.normalizeGreek(s1);
			s2 = StringManipulation.normalizeGreek(s2);
		}
		s1 = s1.toUpperCase().replaceAll("\\s", " ");
		s2 = s2.toUpperCase().replaceAll("\\s", " ");
                
		
		ArrayList<String> s1Tokens = StringManipulation.getTokensList(s1);
		ArrayList<String> s2Tokens = StringManipulation.getTokensList(s2);
		
                for (int i = 0; i < s1Tokens.size(); i++)
                {
                    String s = s1Tokens.get(i);
                    
                    for (String stop : stopWordsCopy)
                    {
                        if (s.equalsIgnoreCase(stop))
                        {
                            s1Tokens.remove(i);
                            i--;
                            break;
                        }
                    }
                }
                
                for (int i = 0; i < s2Tokens.size(); i++)
                {
                    String s = s2Tokens.get(i);
                    
                    for (String stop : stopWordsCopy)
                    {
                        if (s.equalsIgnoreCase(stop))
                        {
                            s2Tokens.remove(i);
                            i--;
                            break;
                        }
                    }
                }
                
                //System.out.println("!" + s2Tokens);
                
		HashSet<String> s1UniqueTokens = new HashSet<String>(s1Tokens);
		HashSet<String> s2UniqueTokens = new HashSet<String>(s2Tokens);
 		
		HashSet<String> allTokens = new HashSet<String>();
		allTokens.addAll(s1Tokens);
		allTokens.addAll(s2Tokens);
		

		int numOfS1Tokens = s1UniqueTokens.size();
		int numOfS2Tokens = s2UniqueTokens.size();
		int numOfCommonTokens = (numOfS1Tokens + numOfS2Tokens) - allTokens.size();
		
                if (numOfS1Tokens*numOfS2Tokens == 0)
                    return 0;
                
                double similarity = (double)numOfCommonTokens / (Math.sqrt((double)numOfS1Tokens) * Math.sqrt((double)numOfS2Tokens));
		return similarity;
	}
    
	public static BigDecimal getSimilarity(String s1, String s2, boolean greek) {
		if(greek)
		{
                    s1 = StringManipulation.normalizeGreek(s1);
                    s2 = StringManipulation.normalizeGreek(s2);
		}
		s1 = s1.toUpperCase().replaceAll("\\s", " ");
		s2 = s2.toUpperCase().replaceAll("\\s", " ");
                		
		ArrayList<String> s1Tokens = StringManipulation.getTokensList(s1);
		ArrayList<String> s2Tokens = StringManipulation.getTokensList(s2);
		
		HashSet<String> s1UniqueTokens = new HashSet<String>(s1Tokens);
		HashSet<String> s2UniqueTokens = new HashSet<String>(s2Tokens);
 		
		HashSet<String> allTokens = new HashSet<String>();
		allTokens.addAll(s1Tokens);
		allTokens.addAll(s2Tokens);
		
		int numOfS1Tokens = s1UniqueTokens.size();
		int numOfS2Tokens = s2UniqueTokens.size();
		int numOfCommonTokens = (numOfS1Tokens + numOfS2Tokens) - allTokens.size();
                
		double similarity = (double)numOfCommonTokens / (Math.sqrt((double)numOfS1Tokens) * Math.sqrt((double)numOfS2Tokens));
		return BigDecimal.valueOf(similarity);
	}
        
        public static double getSimilarity(ArrayList<String> s1Tokens, ArrayList<String> s2Tokens) {
            if (s1Tokens.isEmpty() || s2Tokens.isEmpty()) {
                return 0.0;
            }
            
            HashSet<String> uniqueTokens = new HashSet<String>(s1Tokens);
            uniqueTokens.addAll(s2Tokens);

            ArrayList<String> allTokens = new ArrayList<String>(uniqueTokens);
            ArrayList<Double> s1Vector = new ArrayList<Double>();
            ArrayList<Double> s2Vector = new ArrayList<Double>();                

            for (int i = 0; i < allTokens.size(); i++) {
                double tf = 0.0;
                for (int j = 0; j < s1Tokens.size(); j++) {
                    if (allTokens.get(i).equals(s1Tokens.get(j))) {
                        tf += 1.0;
                    }
                }
                s1Vector.add(tf);

                tf = 0.0;
                for (int j = 0; j < s2Tokens.size(); j++) {
                    if (allTokens.get(i).equals(s2Tokens.get(j))) {
                        tf += 1.0;
                    }
                }
                s2Vector.add(tf);
            }

            double dotProduct = 0.0;
            double s1Product = 0.0;
            double s2Product = 0.0;
            for (int i = 0; i < allTokens.size(); i++) {
                dotProduct += s1Vector.get(i) * s2Vector.get(i);

                s1Product += s1Vector.get(i) * s1Vector.get(i);
                s2Product += s2Vector.get(i) * s2Vector.get(i);
            }

            return dotProduct/(Math.sqrt(s1Product) * Math.sqrt(s2Product));
	}
	
	public static double getNGramSimilarity(String s1, String s2, int n, boolean greek)
	{
		if(greek)
		{
                    s1 = StringManipulation.normalizeGreek(s1);
                    s2 = StringManipulation.normalizeGreek(s2);
		}
		s1 = s1.toUpperCase().replaceAll("\\s", " ");
		s2 = s2.toUpperCase().replaceAll("\\s", " ");
		
		ArrayList<String> s1Tokens = StringManipulation.getNGrams(s1, n);
		ArrayList<String> s2Tokens = StringManipulation.getNGrams(s2, n);
		
		HashSet<String> s1UniqueTokens = new HashSet<String>(s1Tokens);
		HashSet<String> s2UniqueTokens = new HashSet<String>(s2Tokens);
 		
		HashSet<String> allTokens = new HashSet<String>();
		allTokens.addAll(s1Tokens);
		allTokens.addAll(s2Tokens);
		
		int numOfS1Tokens = s1UniqueTokens.size();
		int numOfS2Tokens = s2UniqueTokens.size();
		int numOfCommonTokens = (numOfS1Tokens + numOfS2Tokens) - allTokens.size();
		
		double similarity = (double)numOfCommonTokens / (Math.sqrt((double)numOfS1Tokens) * Math.sqrt((double)numOfS2Tokens));
		return similarity;
	}
}
