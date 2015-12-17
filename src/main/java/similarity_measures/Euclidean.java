package similarity_measures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Euclidean
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
		
		int maxPossibleTokens = (s1Tokens.size() * s1Tokens.size()) + (s2Tokens.size() * s2Tokens.size());
		
		HashSet<String> allTokens = new HashSet<String>();
		allTokens.addAll(s1Tokens);
		allTokens.addAll(s2Tokens);
		
		HashMap<String, Integer> x = StringManipulation.getTokensMap(s1Tokens);
		HashMap<String, Integer> y = StringManipulation.getTokensMap(s2Tokens);
		
		int totalDistance = 0;
		
		for (String token : allTokens)
		{
			if(!x.containsKey(token))
			{
				totalDistance += y.get(token) * y.get(token);
			}
			else if(!y.containsKey(token))
			{
				totalDistance += x.get(token) * x.get(token);
			}
			else
			{
				int distance = x.get(token) - y.get(token);
				totalDistance += distance * distance;
			}
		}
		double similarity = 1.0 - (Math.sqrt((double)totalDistance) / Math.sqrt((double)maxPossibleTokens));
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
		
		int maxPossibleTokens = (s1Tokens.size() * s1Tokens.size()) + (s2Tokens.size() * s2Tokens.size());
		
		HashSet<String> allTokens = new HashSet<String>();
		allTokens.addAll(s1Tokens);
		allTokens.addAll(s2Tokens);
		
		HashMap<String, Integer> x = StringManipulation.getTokensMap(s1Tokens);
		HashMap<String, Integer> y = StringManipulation.getTokensMap(s2Tokens);
		
		int totalDistance = 0;
		
		for (String token : allTokens)
		{
			if(!x.containsKey(token))
			{
				totalDistance += y.get(token) * y.get(token);
			}
			else if(!y.containsKey(token))
			{
				totalDistance += x.get(token) * x.get(token);
			}
			else
			{
				int distance = x.get(token) - y.get(token);
				totalDistance += distance * distance;
			}
		}
		double similarity = 1.0 - (Math.sqrt((double)totalDistance) / Math.sqrt((double)maxPossibleTokens));
		return similarity;
	}
}
