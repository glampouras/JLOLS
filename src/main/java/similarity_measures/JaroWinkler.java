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

public class JaroWinkler
{
	public static double getSimilarity(String s1, String s2, boolean perChar, int minCommonPrefixLength, double p, boolean greek)
	{
		if(greek)
		{
			s1 = StringManipulation.normalizeGreek(s1);
			s2 = StringManipulation.normalizeGreek(s2);
		}
		s1 = s1.toUpperCase().replaceAll("\\s", " ");
		s2 = s2.toUpperCase().replaceAll("\\s", " ");
		
		double dj = Jaro.getSimilarity(s1, s2, perChar, greek);
		
		int commonPrefixLength = (perChar) ? StringManipulation.getCommonPrefixLength(s1, s2, minCommonPrefixLength)
										: StringManipulation.getCommonPrefixLength(StringManipulation.getTokensList(s1), StringManipulation.getTokensList(s2), minCommonPrefixLength);
		
		double similarity = (double)(dj + commonPrefixLength*p*(1 - dj));
		return similarity;
	}
	
	public static double getNGramSimilarity(String s1, String s2, int minCommonPrefixLength, double p, int n, boolean greek)
	{
		if(greek)
		{
			s1 = StringManipulation.normalizeGreek(s1);
			s2 = StringManipulation.normalizeGreek(s2);
		}
		s1 = s1.toUpperCase().replaceAll("\\s", " ");
		s2 = s2.toUpperCase().replaceAll("\\s", " ");
		
		double dj = Jaro.getNGramSimilarity(s1, s2, n, greek);
		
		int commonPrefixLength = StringManipulation.getCommonPrefixLength(StringManipulation.getNGrams(s1, n), StringManipulation.getNGrams(s2, n), minCommonPrefixLength);
		
		double similarity = (double)(dj + commonPrefixLength*p*(1 - dj));
		return similarity;
	}
}
