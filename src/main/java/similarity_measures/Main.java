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
