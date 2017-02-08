package simpleLM;

import imitationNLG.Action;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

/**
 *
 * @author Gerasimos Lampouras
 */
public class SimpleLM implements Serializable {
    private static final long serialVersionUID = 1L;

    private HashMap<String, Integer> unigramCounts;
    private Integer unigramTotalCount;
    private HashMap<Integer, HashMap<ArrayList<String>, HashMap<String, Integer>>> ngramCounts;
    private HashMap<Integer, HashMap<ArrayList<String>, Integer>> ngramTotalCounts;
    private int order = 3;

    /**
     *
     * @param order
     */
    public SimpleLM(int order) {
        this.order = order;

        unigramCounts = new HashMap<String, Integer>();
        unigramTotalCount = 0;
        ngramCounts = new HashMap<Integer, HashMap<ArrayList<String>, HashMap<String, Integer>>>();
        ngramTotalCounts = new HashMap<Integer, HashMap<ArrayList<String>, Integer>>();
        for (int n = 1; n < order; n++) {
            ngramCounts.put(n, new HashMap<ArrayList<String>, HashMap<String, Integer>>());
            ngramTotalCounts.put(n, new HashMap<ArrayList<String>, Integer>());
        }
    }

    /**
     *
     * @param sequences
     */
    public void trainOnStrings(ArrayList<ArrayList<String>> sequences) {
        sequences.stream().forEach((ArrayList<String> seq) -> {
            seq.stream().filter((token) -> (!token.equals(Action.TOKEN_END)
                    && !token.equals("@@"))).map((token) -> {
                        if (!unigramCounts.containsKey(token)) {
                            unigramCounts.put(token, 1);
                        } else {
                            unigramCounts.put(token, unigramCounts.get(token) + 1);
                        }
                        return token;
                    }).forEach((_item) -> {
                        unigramTotalCount++;
                    });
        });
        sequences.stream().forEach((seq) -> {
            for (int n = 1; n < order; n++) {
                for (int i = n; i < seq.size(); i++) {
                    String token = seq.get(i);

                    ArrayList<String> context = new ArrayList<String>();
                    int window = i - n;
                    for (int c = i - 1; (c >= window && c >= 0); c--) {
                        if (!seq.get(c).equals(Action.TOKEN_END)) {
                            context.add(0, seq.get(c));
                        } else {
                            window--;
                        }
                    }
                    if (!context.isEmpty()
                            && !context.contains("@@")) {
                        if (!ngramCounts.get(n).containsKey(context)) {
                            ngramCounts.get(n).put(context, new HashMap<String, Integer>());
                        }
                        if (!ngramCounts.get(n).get(context).containsKey(token)) {
                            ngramCounts.get(n).get(context).put(token, 1);
                        } else {
                            ngramCounts.get(n).get(context).put(token, ngramCounts.get(n).get(context).get(token) + 1);
                        }

                        if (!ngramTotalCounts.get(n).containsKey(context)) {
                            ngramTotalCounts.get(n).put(context, 1);
                        } else {
                            ngramTotalCounts.get(n).put(context, ngramTotalCounts.get(n).get(context) + 1);
                        }
                    }
                }
            }
        });
    }

    /**
     *
     * @param seq
     * @return
     */
    public double getProbability(ArrayList<String> seq) {
        double prob = 1.0;
        for (int i = 0; i < seq.size(); i++) {
            String token = seq.get(i);
            int o = i;
            if (o >= order) {
                o = order - 1;
            }
            while (o >= 0) {
                if (o == 0) {
                    if (!unigramCounts.containsKey(token)) {
                        if (token.equals("@@")) {
                            prob *= 1.0;
                        } else {
                            prob *= 0.0;
                        }
                    } else {
                        prob *= unigramCounts.get(token).doubleValue() / unigramTotalCount.doubleValue();
                    }
                    o = -1;
                } else {
                    ArrayList<String> context = new ArrayList<String>();
                    for (int c = i - o; c < i; c++) {
                        context.add(seq.get(c));
                    }
                    if (!ngramCounts.get(o).containsKey(context)) {
                        o--;
                    } else if (!ngramCounts.get(o).get(context).containsKey(token)) {
                        o--;
                    } else {
                        prob *= ngramCounts.get(o).get(context).get(token).doubleValue() / ngramTotalCounts.get(o).get(context).doubleValue();
                        o = -1;
                    }
                }
            }
        }
        return prob;
    }
}
