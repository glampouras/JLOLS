/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imitationLearning;

import static edu.stanford.nlp.mt.metrics.BLEUMetric.computeLocalSmoothScore;
import java.util.ArrayList;
import java.util.logging.Logger;
import static java.util.logging.Logger.getLogger;
import static similarity_measures.Rouge.ROUGE_N;

/**
 *
 * @author Black Fox
 */
public class LossFunction {
    /**
     *
     */
    public static String metric = "B";
    /**
     *
     * @param s1
     * @param s2s
     * @param coverageError
     * @return
     */
    public static double getCostMetric(String s1, ArrayList<String> s2s, Double coverageError) {
        switch (metric) {
            case "B":
                return getBLEU(s1, s2s);
            case "R":
                return getROUGE(s1, s2s);
            case "BC":
                if (coverageError == -1.0) {
                    return getBLEU(s1, s2s);
                }
                return (getBLEU(s1, s2s) + coverageError) / 2.0;
            case "RC":
                if (coverageError == -1.0) {
                    return getROUGE(s1, s2s);
                }
                return (getROUGE(s1, s2s) + coverageError) / 2.0;
            case "BRC":
                if (coverageError == -1.0) {
                    return (getBLEU(s1, s2s) + getROUGE(s1, s2s)) / 2.0;
                }
                return (getBLEU(s1, s2s) + getROUGE(s1, s2s) + coverageError) / 3.0;
            case "BR":
                return (getBLEU(s1, s2s) + getROUGE(s1, s2s)) / 2.0;
            default:
                break;
        }
        return getBLEU(s1, s2s);
    }

    /**
     *
     * @param s1
     * @param s2s
     * @return
     */
    public static double getBLEU(String s1, ArrayList<String> s2s) {
        return 1.0 - computeLocalSmoothScore(s1, s2s, 4);
    }

    /**
     *
     * @param s1
     * @param s2s
     * @return
     */
    public static double getROUGE(String s1, ArrayList<String> s2s) {
        double maxRouge = 0.0;
        for (String s2 : s2s) {
            double rouge = ROUGE_N(s1, s2, 4);
            if (rouge > maxRouge) {
                maxRouge = rouge;
            }
        }
        return 1.0 - maxRouge;
    }
}
