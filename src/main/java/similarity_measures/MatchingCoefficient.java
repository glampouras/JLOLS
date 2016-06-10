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

public class MatchingCoefficient
{
	public static double getSimilarity(String s1, String s2, boolean greek)
	{
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
		
		double similarity = (double)numOfCommonTokens / (double)Math.max(numOfS1Tokens, numOfS2Tokens);
		return similarity;
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
		
		double similarity = (double)numOfCommonTokens / (double)Math.max(numOfS1Tokens, numOfS2Tokens);
		return similarity;
	}
}
