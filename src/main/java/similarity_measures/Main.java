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

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args)
	{
		String s1 = "����� ����� �������� �����˹�";
		String s2 = "����� ����� �������� ��������";
		boolean greek = true;
		boolean perChar = true;
		
		double similarity = Manhattan.getSimilarity(s1, s2, greek);
		System.out.println("Manhattan similarity: " + similarity);
		
		similarity = Levenshtein.getSimilarity(s1, s2, perChar, greek);
		System.out.println("Levenshtein similarity: " + similarity);
		
		//similarity = Cosine.getSimilarity(s1, s2, greek);
		System.out.println("Cosine similarity: " + similarity);
		
		similarity = Euclidean.getSimilarity(s1, s2, greek);
		System.out.println("Euclidean similarity: " + similarity);
		
		similarity = MatchingCoefficient.getSimilarity(s1, s2, greek);
		System.out.println("Matching Coefficient similarity: " + similarity);
		
		similarity = DiceCoefficient.getSimilarity(s1, s2, greek);
		System.out.println("Dice Coefficient similarity: " + similarity);
		
		similarity = JaccardCoefficient.getSimilarity(s1, s2, greek);
		System.out.println("Jaccard Coefficient similarity: " + similarity);
		
		similarity = Jaro.getSimilarity(s1, s2, perChar, greek);
		System.out.println("Jaro similarity: " + similarity);
		
		similarity = JaroWinkler.getSimilarity(s1, s2, perChar, 6, 0.1, greek);
		System.out.println("Jaro Winkler similarity: " + similarity);
	}

}
