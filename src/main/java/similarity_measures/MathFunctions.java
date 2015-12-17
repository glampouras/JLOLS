package similarity_measures;

public class MathFunctions
{
	public static int minOf3(int a, int b, int c)
	{
		int min = a < b ? a : b;
		min = (c < min) ? c : min;
		return min;
	}
}
