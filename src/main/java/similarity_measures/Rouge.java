
package similarity_measures;

import java.util.HashSet;

public class Rouge {

    /**
     *
     * @param X
     * @param Y
     * @param N
     * @return
     */
    public static double ROUGE_N(String X, String Y, int N) {
        HashSet<String> ngramsX = new HashSet<>();
        String tokensX[] = X.split(" ");
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < tokensX.length; i++) {
                if (i + n < tokensX.length) {
                    String ngram = "";
                    for (int j = i; j <= i + n; j++) {
                        ngram += tokensX[j] + " ";
                    }
                    ngram = ngram.trim();
                    ngramsX.add(ngram);
                }
            }
        }        
        
        HashSet<String> totalNGrams = new HashSet<>(ngramsX);
        HashSet<String> commonNGrams = new HashSet<>();
        String tokensY[] = Y.split(" ");
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < tokensY.length; i++) {
                if (i + n < tokensY.length) {
                    String ngram = "";
                    for (int j = i; j <= i + n; j++) {
                        ngram += tokensY[j] + " ";
                    }
                    ngram = ngram.trim();
                    totalNGrams.add(ngram);
                    if (ngramsX.contains(ngram)) {
                        commonNGrams.add(ngram);
                    }
                }
            }
        }
        return commonNGrams.size() / (double) totalNGrams.size();
    }

    private Rouge() {
    }
}
