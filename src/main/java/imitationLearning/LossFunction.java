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
        int avgSwitch = 0;
        int maxSwitch = 1;
        int oldSwitch = 2;
        int sw = oldSwitch;
        switch (metric) {
            case "B":
                if (sw == avgSwitch) {                    
                    double avg = 0.0;
                    int count = 0;
                    for (String s2 : s2s) {
                        ArrayList<String> newSet = new ArrayList<String>();
                        newSet.add(s2);
                        avg += getBLEU(s1, newSet);
                        count++;
                    }
                    avg /= count;                            
                    return avg;                    
                } else if (sw == maxSwitch) {                    
                    double max = 0.0;
                    for (String s2 : s2s) {
                        ArrayList<String> newSet = new ArrayList<String>();
                        newSet.add(s2);
                        double score = getBLEU(s1, newSet);
                        if (score > max) {
                            max = score;
                        }
                    }
                    return max;
                } else {
                    return getBLEU(s1, s2s);
                }
            case "R":
                if (sw == avgSwitch) {                    
                    double avg = 0.0;
                    int count = 0;
                    for (String s2 : s2s) {
                        ArrayList<String> newSet = new ArrayList<String>();
                        newSet.add(s2);
                        avg += getROUGE(s1, newSet);
                        count++;
                    }
                    avg /= count;                            
                    return avg;                    
                } else if (sw == maxSwitch) {                    
                    double max = 0.0;
                    for (String s2 : s2s) {
                        ArrayList<String> newSet = new ArrayList<String>();
                        newSet.add(s2);
                        double score = getROUGE(s1, newSet);
                        if (score > max) {
                            max = score;
                        }
                    }
                    return max;
                } else {
                    return getROUGE(s1, s2s);
                }
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
                if (sw == avgSwitch) {
                    if (coverageError == -1.0) {
                        double avg = 0.0;
                        int count = 0;
                        for (String s2 : s2s) {
                            ArrayList<String> newSet = new ArrayList<String>();
                            newSet.add(s2);
                            avg += (getBLEU(s1, newSet) + getROUGE(s1, newSet)) / 2.0;
                            count++;
                        }
                        avg /= count;                            
                        return avg;
                    }
                    double avg = 0.0;
                    int count = 0;
                    for (String s2 : s2s) {
                        ArrayList<String> newSet = new ArrayList<String>();
                        newSet.add(s2);
                        avg += (getBLEU(s1, newSet) + getROUGE(s1, newSet) + coverageError) / 3.0;
                        count++;
                    }
                    avg /= count;                            
                    return avg;
                } else if (sw == maxSwitch) {
                    if (coverageError == -1.0) {
                        double max = 0.0;
                        for (String s2 : s2s) {
                            ArrayList<String> newSet = new ArrayList<String>();
                            newSet.add(s2);
                            double score = (getBLEU(s1, newSet) + getROUGE(s1, newSet)) / 2.0;
                            if (score > max) {
                                max = score;
                            }
                        }
                        return max;
                    }
                    double max = 0.0;
                    for (String s2 : s2s) {
                        ArrayList<String> newSet = new ArrayList<String>();
                        newSet.add(s2);
                        double score = (getBLEU(s1, newSet) + getROUGE(s1, newSet)) / 2.0;
                        if (score > max) {
                            max = score;
                        }
                    }
                    return max;
                } else {                
                    if (coverageError == -1.0) {
                        return (getBLEU(s1, s2s) + getROUGE(s1, s2s)) / 2.0;
                    }
                    return (getBLEU(s1, s2s) + getROUGE(s1, s2s) + coverageError) / 3.0;
                }
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
