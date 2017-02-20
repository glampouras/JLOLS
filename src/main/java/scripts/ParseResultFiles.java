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
package scripts;

import edu.stanford.nlp.mt.metrics.BLEUMetric;
import edu.stanford.nlp.mt.metrics.NISTMetric;
import edu.stanford.nlp.mt.tools.NISTTokenizer;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.util.IStrings;
import edu.stanford.nlp.mt.util.ScoredFeaturizedTranslation;
import edu.stanford.nlp.mt.util.Sequence;
import structuredPredictionNLG.Action;
import structuredPredictionNLG.Bagel;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import similarity_measures.Rouge;

/**
 *
 * @author Gerasimos Lampouras
 */
public class ParseResultFiles {

    /**
     *
     * @param args
     */
    public static void main(String[] args) {
        //generateQSUBforBAGEL();
        //parseBagelResults();
        //generateQSUBforSFXHotel();
        //parseSFXHotelResults();
        //upperCaseFile();
        //parseWenFiles();
        //parseERR();
        //parseERRBagel();
        //parseOldGScriptQuestionnaires();
        parseGScriptQuestionnaires();
        
        //checkIfTextsInTexts();
    }

    /**
     *
     */
    public static void parseOldGScriptQuestionnaires() {
        //String mrFile = "NLG_EvalAnswers.txt";
        String mrFile = "NLG_EvalAnswers_fb.txt";

        HashMap<String, String> ages = new HashMap<>();
        HashMap<String, String> languages = new HashMap<>();
        HashMap<String, String> educationLevel = new HashMap<>();
        HashMap<String, String> englishLevel = new HashMap<>();
        HashMap<String, HashMap<String, HashMap<String, Integer>>> scoresPerUser = new HashMap<>();
        //PRINT RESULTS
        int emptyMailID = 1;
        try (BufferedReader br = new BufferedReader(new FileReader(mrFile))) {
            String s;
            String email = "";
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    if (s.startsWith("E-mail address")) {
                        email = s.substring("E-mail address (if you want a chance to win a £20 Amazon Voucher):".length() + 1).trim();
                        if (email.trim().isEmpty()) {
                            email = "emptyMail_" + emptyMailID;
                            emptyMailID++;
                        }
                    } else if (s.startsWith("How old are you?")) {
                        ages.put(email, s.substring("How old are you?".length()).trim());
                    } else if (s.startsWith("What is your native language?")) {
                        languages.put(email, s.substring("What is your native language?".length()).trim());
                    } else if (s.startsWith("What is your current education level?")) {
                        educationLevel.put(email, s.substring("What is your current education level?".length()).trim());
                    } else if (s.startsWith("What is your English level?")) {
                        englishLevel.put(email, s.substring("What is your English level?".length()).trim());
                    } else if (s.startsWith("Fluency")) {
                        String[] arr = s.split("\t");
                        if (!scoresPerUser.containsKey(email)) {
                            scoresPerUser.put(email, new HashMap<String, HashMap<String, Integer>>());
                            scoresPerUser.get(email).put("Fluency", new HashMap<String, Integer>());
                            scoresPerUser.get(email).put("Adequacy", new HashMap<String, Integer>());
                        }
                        scoresPerUser.get(email).get("Fluency").put(arr[0].substring("Fluency".length()).trim(), Integer.parseInt(arr[1].trim()));
                    } else if (s.startsWith("Adequacy")) {
                        String[] arr = s.split("\t");
                        if (!scoresPerUser.containsKey(email)) {
                            scoresPerUser.put(email, new HashMap<String, HashMap<String, Integer>>());
                            scoresPerUser.get(email).put("Fluency", new HashMap<String, Integer>());
                            scoresPerUser.get(email).put("Adequacy", new HashMap<String, Integer>());
                        }
                        scoresPerUser.get(email).get("Adequacy").put(arr[0].substring("Adequacy".length()).trim(), Integer.parseInt(arr[1].trim()));
                    }
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        ArrayList<String> emails = new ArrayList<String>(scoresPerUser.keySet());
        Random r = new Random();
        System.out.println("********** LUCKY WINNERS ************");
        System.out.println(emails.get(r.nextInt(emails.size())));
        System.out.println(emails.get(r.nextInt(emails.size())));
        System.out.println(emails.get(r.nextInt(emails.size())));
        System.out.println("********** LUCKY WINNERS ************");
        System.out.println(new HashSet(ages.values()));
        System.out.println(new HashSet(languages.values()));
        System.out.println(new HashSet(educationLevel.values()));
        System.out.println(new HashSet(englishLevel.values()));

        HashSet<String> unreliableUserIDs = new HashSet<>();
        HashSet<String> unreliableUserIDsFL = new HashSet<>();
        HashSet<String> unreliableUserIDsAD = new HashSet<>();
        scoresPerUser.keySet().forEach((userID) -> {
            if (scoresPerUser.get(userID).get("Fluency").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("CONTROL-Med")
                    && scoresPerUser.get(userID).get("Fluency").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("CONTROL-Best")
                    && scoresPerUser.get(userID).get("Fluency").get("CONTROL-Med") <= scoresPerUser.get(userID).get("Fluency").get("CONTROL-Best")
                    && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Med")
                    && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Best")
                    && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Med") <= scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Best")) {
            } else {
                /*System.out.println(userID);
                System.out.println(scores.get(userID).get("Fluency").get("CONTROL-Worse") + " < " + scores.get(userID).get("Fluency").get("CONTROL-Med") + " <= " + scores.get(userID).get("Fluency").get("CONTROL-Best"));
                System.out.println(scores.get(userID).get("Adequacy").get("CONTROL-Worse") + " < " + scores.get(userID).get("Adequacy").get("CONTROL-Med") + " <= " + scores.get(userID).get("Adequacy").get("CONTROL-Best"));
                System.out.println("");*/
                unreliableUserIDs.add(userID);
                if (scoresPerUser.get(userID).get("Fluency").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("CONTROL-Med")
                        && scoresPerUser.get(userID).get("Fluency").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("CONTROL-Best")
                        && scoresPerUser.get(userID).get("Fluency").get("CONTROL-Med") <= scoresPerUser.get(userID).get("Fluency").get("CONTROL-Best")) {
                } else {
                    unreliableUserIDsFL.add(userID);
                }
                if (scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Med")
                        && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Best")
                        && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Med") <= scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Best")) {
                } else {
                    unreliableUserIDsAD.add(userID);
                }
            }
        });

        ArrayList<String> instanceIDs = new ArrayList<>();
        scoresPerUser.keySet().forEach((userID) -> {
            scoresPerUser.get(userID).get("Fluency").keySet().stream().filter((instance) -> (!instance.contains("CONTROL-Worse")
                    && !instance.contains("CONTROL-Med")
                    && !instance.contains("CONTROL-Best"))).filter((instance) -> (!instanceIDs.contains(instance))).forEachOrdered((instance) -> {
                        instanceIDs.add(instance);
            });
        });
        Collections.sort(instanceIDs);

        System.out.println("PARTICIPATING users: " + scoresPerUser.keySet().size());
        System.out.println("UNRELIABLE: " + (unreliableUserIDs.size() / ((double) scoresPerUser.keySet().size())));
        System.out.println("UNRELIABLE on Fluency: " + (unreliableUserIDsFL.size() / ((double) scoresPerUser.keySet().size())));
        System.out.println("UNRELIABLE on Adequacy: " + (unreliableUserIDsAD.size() / ((double) scoresPerUser.keySet().size())));
        unreliableUserIDs.forEach((unreliableUserID) -> {
            scoresPerUser.remove(unreliableUserID);
        });
        System.out.println("REMAINING users: " + scoresPerUser.keySet().size());

        //AGREEMENT VECTORS
        HashSet<String> usedIDs = new HashSet<String>();
        for (String userID : scoresPerUser.keySet()) {
            if (!usedIDs.contains(userID)) {
                usedIDs.add(userID);
                HashSet<String> usersOfSameInstances = new HashSet<String>();
                usersOfSameInstances.add(userID);
                for (String userID2 : scoresPerUser.keySet()) {
                    if (!usedIDs.contains(userID2)) {
                        if (scoresPerUser.get(userID).get("Fluency").keySet().equals(scoresPerUser.get(userID2).get("Fluency").keySet())
                                && scoresPerUser.get(userID).get("Adequacy").keySet().equals(scoresPerUser.get(userID2).get("Adequacy").keySet())) {
                            usedIDs.add(userID2);
                            usersOfSameInstances.add(userID2);
                        }
                    }
                }
                if (usersOfSameInstances.size() > 1) {
                    ArrayList<String> instances = new ArrayList<>(scoresPerUser.get(userID).get("Fluency").keySet());
                    usersOfSameInstances.stream().map((ID) -> {
                        System.out.print("Fluency of " + ID + ":\t");
                        return ID;
                    }).map((ID) -> {
                        instances.forEach((instance) -> {
                            System.out.print(scoresPerUser.get(ID).get("Fluency").get(instance) + "\t");
                        });
                        return ID;
                    }).forEachOrdered((_item) -> {
                        System.out.println();
                    });
                }
            }
        }
        System.out.println("===============");
        usedIDs = new HashSet<String>();
        for (String userID : scoresPerUser.keySet()) {
            if (!usedIDs.contains(userID)) {
                usedIDs.add(userID);
                HashSet<String> usersOfSameInstances = new HashSet<String>();
                usersOfSameInstances.add(userID);
                for (String userID2 : scoresPerUser.keySet()) {
                    if (!usedIDs.contains(userID2)) {
                        if (scoresPerUser.get(userID).get("Fluency").keySet().equals(scoresPerUser.get(userID2).get("Fluency").keySet())
                                && scoresPerUser.get(userID).get("Adequacy").keySet().equals(scoresPerUser.get(userID2).get("Adequacy").keySet())) {
                            usedIDs.add(userID2);
                            usersOfSameInstances.add(userID2);
                        }
                    }
                }
                if (usersOfSameInstances.size() > 1) {
                    ArrayList<String> instances = new ArrayList<>(scoresPerUser.get(userID).get("Adequacy").keySet());
                    usersOfSameInstances.stream().map((ID) -> {
                        System.out.print("Adequacy of " + ID + ":\t");
                        return ID;
                    }).map((ID) -> {
                        instances.forEach((instance) -> {
                            System.out.print(scoresPerUser.get(ID).get("Adequacy").get(instance) + "\t");
                        });
                        return ID;
                    }).forEachOrdered((_item) -> {
                        System.out.println();
                    });
                }
            }
        }

        HashMap<String, ArrayList<Integer>> scoresPerInstanceFluency = new HashMap<>();
        HashMap<String, ArrayList<Integer>> scoresPerInstanceAdequacy = new HashMap<>();
        scoresPerUser.keySet().stream().map((userID) -> {
            scoresPerUser.get(userID).get("Fluency").keySet().stream().filter((instance) -> (!instance.equals("CONTROL-Worse")
                    && !instance.equals("CONTROL-Med")
                    && !instance.equals("CONTROL-Best"))).map((instance) -> {
                        if (!scoresPerInstanceFluency.containsKey(instance)) {
                            scoresPerInstanceFluency.put(instance, new ArrayList<Integer>());
                        }
                return instance;
            }).forEachOrdered((instance) -> {
                scoresPerInstanceFluency.get(instance).add(scoresPerUser.get(userID).get("Fluency").get(instance));
            });
            return userID;
        }).forEachOrdered((userID) -> {
            scoresPerUser.get(userID).get("Adequacy").keySet().stream().filter((instance) -> (!instance.equals("CONTROL-Worse")
                    && !instance.equals("CONTROL-Med")
                    && !instance.equals("CONTROL-Best"))).map((instance) -> {
                        if (!scoresPerInstanceAdequacy.containsKey(instance)) {
                            scoresPerInstanceAdequacy.put(instance, new ArrayList<Integer>());
                        }
                return instance;
            }).forEachOrdered((instance) -> {
                scoresPerInstanceAdequacy.get(instance).add(scoresPerUser.get(userID).get("Adequacy").get(instance));
            });
        });

        ArrayList<Double> LOLS_BAGEL_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFHOT_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFRES_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> WEN_SFHOT_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> WEN_SFRES_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> Dusek_AVG_fluency_set = new ArrayList<>();

        Double LOLS_BAGEL_AVG_fluency = 0.0;
        Double LOLS_SFHOT_AVG_fluency = 0.0;
        Double LOLS_SFRES_AVG_fluency = 0.0;
        Double WEN_SFHOT_AVG_fluency = 0.0;
        Double WEN_SFRES_AVG_fluency = 0.0;
        Double Dusek_AVG_fluency = 0.0;

        Double LOLS_BAGEL_total_fluency = 0.0;
        Double LOLS_SFHOT_total_fluency = 0.0;
        Double LOLS_SFRES_total_fluency = 0.0;
        Double WEN_SFHOT_total_fluency = 0.0;
        Double WEN_SFRES_total_fluency = 0.0;
        Double Dusek_total_fluency = 0.0;

        for (String instance : scoresPerInstanceFluency.keySet()) {
            if (instance.startsWith("LOLS_BAGEL")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    LOLS_BAGEL_AVG_fluency += score.doubleValue();
                    LOLS_BAGEL_total_fluency++;
                    LOLS_BAGEL_AVG_fluency_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("LOLS_SFHOT")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    LOLS_SFHOT_AVG_fluency += score.doubleValue();
                    LOLS_SFHOT_total_fluency++;
                    LOLS_SFHOT_AVG_fluency_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("LOLS_SFRES")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    LOLS_SFRES_AVG_fluency += score.doubleValue();
                    LOLS_SFRES_total_fluency++;
                    LOLS_SFRES_AVG_fluency_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("WEN_SFHOT")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    WEN_SFHOT_AVG_fluency += score.doubleValue();
                    WEN_SFHOT_total_fluency++;
                    WEN_SFHOT_AVG_fluency_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("WEN_SFRES")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    WEN_SFRES_AVG_fluency += score.doubleValue();
                    WEN_SFRES_total_fluency++;
                    WEN_SFRES_AVG_fluency_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("DUSEK_BAGEL")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    Dusek_AVG_fluency += score.doubleValue();
                    Dusek_total_fluency++;
                    Dusek_AVG_fluency_set.add(score.doubleValue());
                }
            }
        }
        LOLS_BAGEL_AVG_fluency /= LOLS_BAGEL_total_fluency;
        LOLS_SFHOT_AVG_fluency /= LOLS_SFHOT_total_fluency;
        LOLS_SFRES_AVG_fluency /= LOLS_SFRES_total_fluency;
        WEN_SFHOT_AVG_fluency /= WEN_SFHOT_total_fluency;
        WEN_SFRES_AVG_fluency /= WEN_SFRES_total_fluency;
        Dusek_AVG_fluency /= Dusek_total_fluency;

        ArrayList<Double> LOLS_BAGEL_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFHOT_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFRES_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> WEN_SFHOT_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> WEN_SFRES_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> Dusek_AVG_adequacy_set = new ArrayList<>();

        Double LOLS_BAGEL_AVG_adequacy = 0.0;
        Double LOLS_SFHOT_AVG_adequacy = 0.0;
        Double LOLS_SFRES_AVG_adequacy = 0.0;
        Double WEN_SFHOT_AVG_adequacy = 0.0;
        Double WEN_SFRES_AVG_adequacy = 0.0;
        Double Dusek_AVG_adequacy = 0.0;

        Double LOLS_BAGEL_total_adequacy = 0.0;
        Double LOLS_SFHOT_total_adequacy = 0.0;
        Double LOLS_SFRES_total_adequacy = 0.0;
        Double WEN_SFHOT_total_adequacy = 0.0;
        Double WEN_SFRES_total_adequacy = 0.0;
        Double Dusek_total_adequacy = 0.0;

        for (String instance : scoresPerInstanceAdequacy.keySet()) {
            if (instance.startsWith("LOLS_BAGEL")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    LOLS_BAGEL_AVG_adequacy += score.doubleValue();
                    LOLS_BAGEL_total_adequacy++;
                    LOLS_BAGEL_AVG_adequacy_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("LOLS_SFHOT")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    LOLS_SFHOT_AVG_adequacy += score.doubleValue();
                    LOLS_SFHOT_total_adequacy++;
                    LOLS_SFHOT_AVG_adequacy_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("LOLS_SFRES")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    LOLS_SFRES_AVG_adequacy += score.doubleValue();
                    LOLS_SFRES_total_adequacy++;
                    LOLS_SFRES_AVG_adequacy_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("WEN_SFHOT")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    WEN_SFHOT_AVG_adequacy += score.doubleValue();
                    WEN_SFHOT_total_adequacy++;
                    WEN_SFHOT_AVG_adequacy_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("WEN_SFRES")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    WEN_SFRES_AVG_adequacy += score.doubleValue();
                    WEN_SFRES_total_adequacy++;
                    WEN_SFRES_AVG_adequacy_set.add(score.doubleValue());
                }
            } else if (instance.startsWith("DUSEK_BAGEL")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    Dusek_AVG_adequacy += score.doubleValue();
                    Dusek_total_adequacy++;
                    Dusek_AVG_adequacy_set.add(score.doubleValue());
                }
            }
        }
        LOLS_BAGEL_AVG_adequacy /= LOLS_BAGEL_total_adequacy;
        LOLS_SFHOT_AVG_adequacy /= LOLS_SFHOT_total_adequacy;
        LOLS_SFRES_AVG_adequacy /= LOLS_SFRES_total_adequacy;
        WEN_SFHOT_AVG_adequacy /= WEN_SFHOT_total_adequacy;
        WEN_SFRES_AVG_adequacy /= WEN_SFRES_total_adequacy;
        Dusek_AVG_adequacy /= Dusek_total_adequacy;

        System.out.println();
        System.out.println("BAGEL\tLOLS\tDusek");
        System.out.println("Fluency\t" + LOLS_BAGEL_AVG_fluency + "(" + getConf(LOLS_BAGEL_AVG_fluency_set, LOLS_BAGEL_AVG_fluency, LOLS_BAGEL_total_fluency) + ")" + "\t" + Dusek_AVG_fluency + "(" + getConf(Dusek_AVG_fluency_set, Dusek_AVG_fluency, Dusek_total_fluency) + ")");
        System.out.println("Adequacy\t" + LOLS_BAGEL_AVG_adequacy + "(" + getConf(LOLS_BAGEL_AVG_adequacy_set, LOLS_BAGEL_AVG_adequacy, LOLS_BAGEL_total_adequacy) + ")" + "\t" + Dusek_AVG_adequacy + "(" + getConf(Dusek_AVG_adequacy_set, Dusek_AVG_adequacy, Dusek_total_adequacy) + ")");
        System.out.println();
        System.out.println("SF HOTEL\tLOLS\tWen");
        System.out.println("Fluency\t" + LOLS_SFHOT_AVG_fluency + "(" + getConf(LOLS_SFHOT_AVG_fluency_set, LOLS_SFHOT_AVG_fluency, LOLS_SFHOT_total_fluency) + ")" + "\t" + WEN_SFHOT_AVG_fluency + "(" + getConf(WEN_SFHOT_AVG_fluency_set, WEN_SFHOT_AVG_fluency, WEN_SFHOT_total_fluency) + ")");
        System.out.println("Adequacy\t" + LOLS_SFHOT_AVG_adequacy + "(" + getConf(LOLS_SFHOT_AVG_adequacy_set, LOLS_SFHOT_AVG_adequacy, LOLS_SFHOT_total_adequacy) + ")" + "\t" + WEN_SFHOT_AVG_adequacy + "(" + getConf(WEN_SFHOT_AVG_adequacy_set, WEN_SFHOT_AVG_adequacy, WEN_SFHOT_total_adequacy) + ")");
        System.out.println();
        System.out.println("SF REST\tLOLS\tWen");
        System.out.println("Fluency\t" + LOLS_SFRES_AVG_fluency + "(" + getConf(LOLS_SFRES_AVG_fluency_set, LOLS_SFRES_AVG_fluency, LOLS_SFRES_total_fluency) + ")" + "\t" + WEN_SFRES_AVG_fluency + "(" + getConf(WEN_SFRES_AVG_fluency_set, WEN_SFRES_AVG_fluency, WEN_SFRES_total_fluency) + ")");
        System.out.println("Adequacy\t" + LOLS_SFRES_AVG_adequacy + "(" + getConf(LOLS_SFRES_AVG_adequacy_set, LOLS_SFRES_AVG_adequacy, LOLS_SFRES_total_adequacy) + ")" + "\t" + WEN_SFRES_AVG_adequacy + "(" + getConf(WEN_SFRES_AVG_adequacy_set, WEN_SFRES_AVG_adequacy, WEN_SFRES_total_adequacy) + ")");
        System.out.println();

        HashMap<String, HashMap<String, ArrayList<Integer>>> perInstancePerSystemFluency = new HashMap<>();
        scoresPerInstanceFluency.keySet().stream().filter((instance) -> (!instance.startsWith("CONTROL"))).forEachOrdered((instance) -> {
            String inID = instance.substring(instance.indexOf('_') + 1);
            String system = instance.substring(0, instance.indexOf('_'));
            if (!perInstancePerSystemFluency.containsKey(inID)) {
                perInstancePerSystemFluency.put(inID, new HashMap<String, ArrayList<Integer>>());
            }
            perInstancePerSystemFluency.get(inID).put(system, scoresPerInstanceFluency.get(instance));
        });
        HashMap<String, HashMap<String, ArrayList<Integer>>> perInstancePerSystemAdequacy = new HashMap<>();
        scoresPerInstanceAdequacy.keySet().stream().filter((instance) -> (!instance.startsWith("CONTROL"))).forEachOrdered((instance) -> {
            String inID = instance.substring(instance.indexOf('_') + 1);
            String system = instance.substring(0, instance.indexOf('_'));
            if (!perInstancePerSystemAdequacy.containsKey(inID)) {
                perInstancePerSystemAdequacy.put(inID, new HashMap<String, ArrayList<Integer>>());
            }
            perInstancePerSystemAdequacy.get(inID).put(system, scoresPerInstanceAdequacy.get(instance));
        });
        System.out.println(perInstancePerSystemFluency);
        System.out.println(perInstancePerSystemAdequacy);

        int limit = 5;
        String overAllTitle = "";
        ArrayList<String> outs = new ArrayList<>();
        HashSet<String> emptyIDsLOLS = new HashSet<>();
        HashSet<String> emptyIDsOTHER = new HashSet<>();
        HashSet<String> emptyFluencyIDsLOLS = new HashSet<>();
        HashSet<String> emptyFluencyIDsOTHER = new HashSet<>();
        HashSet<String> emptyInfoIDsLOLS = new HashSet<>();
        HashSet<String> emptyInfoIDsOTHER = new HashSet<>();

        HashSet<String> twoFluencyIDsLOLS = new HashSet<>();
        HashSet<String> twoFluencyIDsOTHER = new HashSet<>();
        HashSet<String> twoInfoIDsLOLS = new HashSet<>();
        HashSet<String> twoInfoIDsOTHER = new HashSet<>();

        HashSet<String> oneFluencyIDsLOLS = new HashSet<>();
        HashSet<String> oneFluencyIDsOTHER = new HashSet<>();
        HashSet<String> oneInfoIDsLOLS = new HashSet<>();
        HashSet<String> oneInfoIDsOTHER = new HashSet<>();
        for (String id : instanceIDs) {
            String title = "ID\t";
            String out = id + "\t";

            int fluencyCountsLOLS = 0;
            int fluencyCountsOTHER = 0;
            int infoCountsLOLS = 0;
            int infoCountsOTHER = 0;
            id = id.substring(id.indexOf('_') + 1);
            if (perInstancePerSystemFluency.containsKey(id)) {
                boolean other = false;
                ArrayList<String> keys = new ArrayList<>();
                if (perInstancePerSystemFluency.get(id).keySet().contains("LOLS")) {
                    keys.add("LOLS");
                }
                for (String key : perInstancePerSystemFluency.get(id).keySet()) {
                    if (!keys.contains(key)) {
                        keys.add(key);
                    }
                }
                if (!keys.contains("LOLS")) {
                    int count = 1;
                    for (; count <= limit; count++) {
                        title += "LOLS fluency " + count + "\t";
                        out += "\t";
                    }
                }
                for (String system : keys) {
                    String sysStr = "LOLS";
                    if (!system.equals("LOLS")) {
                        sysStr = "OTHER";
                        other = true;
                    }
                    int count = 1;
                    for (Integer f : perInstancePerSystemFluency.get(id).get(system)) {
                        title += sysStr + " fluency " + count + "\t";
                        out += f + "\t";
                        if (!other) {
                            fluencyCountsLOLS++;
                        } else {
                            fluencyCountsOTHER++;
                        }
                        count++;
                    }
                    for (; count <= limit; count++) {
                        title += sysStr + " fluency " + count + "\t";
                        out += "\t";
                    }
                }
                if (!other) {
                    int count = 1;
                    for (; count <= limit; count++) {
                        title += "OTHER fluency " + count + "\t";
                        out += "\t";
                    }
                }
            } else {
                int count = 1;
                for (; count <= limit; count++) {
                    title += "LOLS fluency " + count + "\t";
                    out += "\t";
                }
                count = 1;
                for (; count <= limit; count++) {
                    title += "OTHER fluency " + count + "\t";
                    out += "\t";
                }
            }
            if (perInstancePerSystemAdequacy.containsKey(id)) {
                boolean other = false;
                ArrayList<String> keys = new ArrayList<>();
                if (perInstancePerSystemAdequacy.get(id).keySet().contains("LOLS")) {
                    keys.add("LOLS");
                }
                for (String key : perInstancePerSystemAdequacy.get(id).keySet()) {
                    if (!keys.contains(key)) {
                        keys.add(key);
                    }
                }
                if (!keys.contains("LOLS")) {
                    int count = 1;
                    for (; count <= limit; count++) {
                        title += "LOLS informativeness " + count + "\t";
                        out += "\t";
                    }
                }
                for (String system : keys) {
                    String sysStr = "LOLS";
                    if (!system.equals("LOLS")) {
                        sysStr = "OTHER";
                        other = true;
                    }
                    int count = 1;
                    for (Integer f : perInstancePerSystemAdequacy.get(id).get(system)) {
                        title += sysStr + " informativeness " + count + "\t";
                        out += f + "\t";
                        if (!other) {
                            infoCountsLOLS++;
                        } else {
                            infoCountsOTHER++;
                        }
                        count++;
                    }
                    for (; count <= limit; count++) {
                        title += sysStr + " informativeness " + count + "\t";
                        out += "\t";
                    }
                }
                if (!other) {
                    int count = 1;
                    for (; count <= limit; count++) {
                        title += "OTHER informativeness " + count + "\t";
                        out += "\t";
                    }
                }
            } else {
                int count = 1;
                for (; count <= limit; count++) {
                    title += "LOLS informativeness " + count + "\t";
                    out += "\t";
                }
                count = 1;
                for (; count <= limit; count++) {
                    title += "OTHER informativeness " + count + "\t";
                    out += "\t";
                }
            }
            if (overAllTitle.isEmpty()) {
                overAllTitle = title;
            }
            if (!overAllTitle.equals(title)) {
                System.out.println(overAllTitle);
                System.out.println(title);
                System.exit(0);
            }
            outs.add(out);

            if (fluencyCountsLOLS == 0
                    && infoCountsLOLS == 0) {
                emptyIDsLOLS.add(id);
            }
            if (fluencyCountsOTHER == 0
                    && infoCountsOTHER == 0) {
                emptyIDsOTHER.add(id);
            }

            if (fluencyCountsLOLS == 0) {
                emptyFluencyIDsLOLS.add(id);
            }
            if (fluencyCountsOTHER == 0) {
                emptyFluencyIDsOTHER.add(id);
            }
            if (infoCountsLOLS == 0) {
                emptyInfoIDsLOLS.add(id);
            }
            if (infoCountsOTHER == 0) {
                emptyInfoIDsOTHER.add(id);
            }

            if (fluencyCountsLOLS <= 1) {
                oneFluencyIDsLOLS.add(id);
            }
            if (fluencyCountsOTHER <= 1) {
                oneFluencyIDsOTHER.add(id);
            }
            if (infoCountsLOLS <= 1) {
                oneInfoIDsLOLS.add(id);
            }
            if (infoCountsOTHER <= 1) {
                oneInfoIDsOTHER.add(id);
            }

            if (fluencyCountsLOLS <= 2) {
                twoFluencyIDsLOLS.add(id);
            }
            if (fluencyCountsOTHER <= 2) {
                twoFluencyIDsOTHER.add(id);
            }
            if (infoCountsLOLS <= 2) {
                twoInfoIDsLOLS.add(id);
            }
            if (infoCountsOTHER <= 2) {
                twoInfoIDsOTHER.add(id);
            }
        }
        System.out.println(overAllTitle);
        outs.forEach((out) -> {
            System.out.println(out);
        });

        System.out.println("oneFluencyIDsLOLS: " + oneFluencyIDsLOLS);
        System.out.println("------------------");
        System.out.println("oneFluencyIDsOTHER: " + oneFluencyIDsOTHER);
        System.out.println("------------------");
        System.out.println("oneInfoIDsLOLS: " + oneInfoIDsLOLS);
        System.out.println("------------------");
        System.out.println("oneInfoIDsOTHER: " + oneInfoIDsOTHER);

        /*System.out.println("twoFluencyIDsLOLS: " + twoFluencyIDsLOLS);
        System.out.println("------------------");
        System.out.println("twoFluencyIDsOTHER: " + twoFluencyIDsOTHER);
        System.out.println("------------------");
        System.out.println("twoInfoIDsLOLS: " + twoInfoIDsLOLS);
        System.out.println("------------------");
        System.out.println("twoInfoIDsOTHER: " + twoInfoIDsOTHER);
        System.out.println("------------------");*/
        ArrayList<String> redoInstances = new ArrayList<>();
        instanceIDs.stream().map((id) -> {
            String modId = id.substring(0, id.indexOf('-'));
            String no = id.substring(id.indexOf('-') + 1).trim();
            if (no.length() == 1) {
                no = "00" + no;
            } else if (no.length() == 2) {
                no = "0" + no;
            }
            modId += "\t" + no;
            String modIDLOLS = "LOLS_" + modId;
            String outLOLS = "" + modIDLOLS + "\t";
            if (oneFluencyIDsLOLS.contains(id) || oneInfoIDsLOLS.contains(id)) {
                if (oneFluencyIDsLOLS.contains(id)) {
                    outLOLS += "0\t";
                } else {
                    outLOLS += "10\t";
                }
                if (oneInfoIDsLOLS.contains(id)) {
                    outLOLS += "0\t";
                } else {
                    outLOLS += "10\t";
                }
            }
            redoInstances.add(outLOLS);
            String modIDOTHER = modId;
            if (modIDOTHER.startsWith("BAGEL")) {
                modIDOTHER = "DUSEK_" + modIDOTHER;
            } else {
                modIDOTHER = "WEN_" + modIDOTHER;
            }
            String outOTHER = "" + modIDOTHER + "\t";
            if (oneFluencyIDsOTHER.contains(id) || oneInfoIDsOTHER.contains(id)) {
                if (oneFluencyIDsOTHER.contains(id)) {
                    outOTHER += "0\t";
                } else {
                    outOTHER += "10\t";
                }
                if (oneInfoIDsOTHER.contains(id)) {
                    outOTHER += "0\t";
                } else {
                    outOTHER += "10\t";
                }
            }
            return outOTHER;
        }).forEachOrdered((outOTHER) -> {
            redoInstances.add(outOTHER);
        });
        Collections.sort(redoInstances);
        redoInstances.forEach((out) -> {
            System.out.println(out);
        });
    }

    /**
     *
     */
    public static void parseOldGScriptQuestionnaires2() {
        String bagelLOLSTexts = "LOLS_BAGEL_texts.txt";
        String bagelDusekTexts = "Dusek_BAGEL_texts.txt";
        String sfHotelLOLSTexts = "LOLS_SFHOT_texts.txt";
        String sfHotelWenTexts = "WEN_SFHOT_texts.txt";
        String sfRestLOLSTexts = "LOLS_SFRES_texts.txt";
        String sfRestWenTexts = "WEN_SFRES_texts.txt";
        String controlTexts = "CONTROL_texts.txt";

        HashMap<String, String> bagelLOLSTextsMap = new HashMap<>();
        HashMap<String, String> bagelDusekTextsMap = new HashMap<>();
        HashMap<String, String> sfHotelLOLSTextsMap = new HashMap<>();
        HashMap<String, String> sfHotelWenTextsMap = new HashMap<>();
        HashMap<String, String> sfRestLOLSTextsMap = new HashMap<>();
        HashMap<String, String> sfRestWenTextsMap = new HashMap<>();
        HashMap<String, String> controlTextsMap = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(bagelLOLSTexts))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] parts = s.split("\t");
                    bagelLOLSTextsMap.put(parts[0].trim(), parts[1].trim());
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (BufferedReader br = new BufferedReader(new FileReader(bagelDusekTexts))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] parts = s.split("\t");
                    bagelDusekTextsMap.put(parts[0].trim(), parts[1].trim());
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (BufferedReader br = new BufferedReader(new FileReader(sfHotelLOLSTexts))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] parts = s.split("\t");
                    sfHotelLOLSTextsMap.put(parts[0].trim(), parts[1].trim());
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (BufferedReader br = new BufferedReader(new FileReader(sfHotelWenTexts))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] parts = s.split("\t");
                    sfHotelWenTextsMap.put(parts[0].trim(), parts[1].trim());
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (BufferedReader br = new BufferedReader(new FileReader(sfRestLOLSTexts))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] parts = s.split("\t");
                    sfRestLOLSTextsMap.put(parts[0].trim(), parts[1].trim());
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (BufferedReader br = new BufferedReader(new FileReader(sfRestWenTexts))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] parts = s.split("\t");
                    sfRestWenTextsMap.put(parts[0].trim(), parts[1].trim());
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (BufferedReader br = new BufferedReader(new FileReader(controlTexts))) {
            String s;
            int i = 0;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] parts = s.split("\t");
                    controlTextsMap.put(parts[0].trim() + "_" + i, parts[1].trim());
                    i++;
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println(bagelLOLSTextsMap);
        System.out.println("=======================");
        System.out.println(bagelDusekTextsMap);
        System.out.println("=======================");
        System.out.println(sfHotelLOLSTextsMap);
        System.out.println("=======================");
        System.out.println(sfHotelWenTextsMap);
        System.out.println("=======================");
        System.out.println(sfRestLOLSTextsMap);
        System.out.println("=======================");
        System.out.println(sfRestWenTextsMap);
        System.out.println("=======================");
        System.out.println(controlTextsMap);
        System.out.println("=======================");

        String mrFile = "NLG_EvalAnswers_COLING.txt";
        HashMap<String, String> ages = new HashMap<>();
        HashMap<String, String> languages = new HashMap<>();
        HashMap<String, String> educationLevel = new HashMap<>();
        HashMap<String, String> englishLevel = new HashMap<>();
        HashMap<String, HashMap<String, HashMap<String, Integer>>> scoresPerUser = new HashMap<>();
        //PRINT RESULTS
        int emptyMailID = 1;
        try (BufferedReader br = new BufferedReader(new FileReader(mrFile))) {
            String s;
            String email = "";
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    if (s.startsWith("E-mail address")) {
                        email = s.substring("E-mail address (if you want a chance to win a £20 Amazon Voucher):".length() + 1).trim();
                        if (email.trim().isEmpty()) {
                            email = "emptyMail_" + emptyMailID;
                            emptyMailID++;
                        }
                    } else if (s.startsWith("How old are you?")) {
                        ages.put(email, s.substring("How old are you?".length()).trim());
                    } else if (s.startsWith("What is your native language?")) {
                        languages.put(email, s.substring("What is your native language?".length()).trim());
                    } else if (s.startsWith("What is your current education level?")) {
                        educationLevel.put(email, s.substring("What is your current education level?".length()).trim());
                    } else if (s.startsWith("What is your English level?")) {
                        englishLevel.put(email, s.substring("What is your English level?".length()).trim());
                    } else if (s.startsWith("Fluency")) {
                        String[] arr = s.split("\t");
                        if (!scoresPerUser.containsKey(email)) {
                            scoresPerUser.put(email, new HashMap<String, HashMap<String, Integer>>());
                            scoresPerUser.get(email).put("Fluency", new HashMap<String, Integer>());
                            scoresPerUser.get(email).put("Adequacy", new HashMap<String, Integer>());
                        }
                        String parsedID = arr[0].substring("Fluency".length()).trim();
                        HashSet<String> ids = new HashSet<>();
                        if (bagelLOLSTextsMap.containsValue(arr[1])) {
                            bagelLOLSTextsMap.keySet().stream().filter((id) -> (bagelLOLSTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (bagelDusekTextsMap.containsValue(arr[1])) {
                            bagelDusekTextsMap.keySet().stream().filter((id) -> (bagelDusekTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfHotelLOLSTextsMap.containsValue(arr[1])) {
                            sfHotelLOLSTextsMap.keySet().stream().filter((id) -> (sfHotelLOLSTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfHotelWenTextsMap.containsValue(arr[1])) {
                            sfHotelWenTextsMap.keySet().stream().filter((id) -> (sfHotelWenTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfRestLOLSTextsMap.containsValue(arr[1])) {
                            sfRestLOLSTextsMap.keySet().stream().filter((id) -> (sfRestLOLSTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfRestWenTextsMap.containsValue(arr[1])) {
                            sfRestWenTextsMap.keySet().stream().filter((id) -> (sfRestWenTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (controlTextsMap.containsValue(arr[1])) {
                            controlTextsMap.keySet().stream().filter((id) -> (controlTextsMap.get(id).equals(arr[1]))).map((id) -> id.substring(0, id.length() - 2)).forEachOrdered((cleanID) -> {
                                ids.add(cleanID);
                            });
                        }

                        if (ids.isEmpty()) {
                            if (bagelLOLSTextsMap.containsValue(arr[1])) {
                                bagelLOLSTextsMap.keySet().stream().filter((id) -> (bagelLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (bagelDusekTextsMap.containsValue(arr[1])) {
                                bagelDusekTextsMap.keySet().stream().filter((id) -> (bagelDusekTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfHotelLOLSTextsMap.containsValue(arr[1])) {
                                sfHotelLOLSTextsMap.keySet().stream().filter((id) -> (sfHotelLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfHotelWenTextsMap.containsValue(arr[1])) {
                                sfHotelWenTextsMap.keySet().stream().filter((id) -> (sfHotelWenTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfRestLOLSTextsMap.containsValue(arr[1])) {
                                sfRestLOLSTextsMap.keySet().stream().filter((id) -> (sfRestLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfRestWenTextsMap.containsValue(arr[1])) {
                                sfRestWenTextsMap.keySet().stream().filter((id) -> (sfRestWenTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                        }
                        if (ids.isEmpty()) {
                            System.exit(0);
                        }

                        for (String id : ids) {
                            scoresPerUser.get(email).get("Fluency").put(id, Integer.parseInt(arr[2].trim()));
                        }
                    } else if (s.startsWith("Adequacy")) {
                        String[] arr = s.split("\t");
                        if (!scoresPerUser.containsKey(email)) {
                            scoresPerUser.put(email, new HashMap<String, HashMap<String, Integer>>());
                            scoresPerUser.get(email).put("Fluency", new HashMap<String, Integer>());
                            scoresPerUser.get(email).put("Adequacy", new HashMap<String, Integer>());
                        }
                        String parsedID = arr[0].substring("Adequacy".length()).trim();
                        HashSet<String> ids = new HashSet<>();
                        if (bagelLOLSTextsMap.containsValue(arr[1])) {
                            bagelLOLSTextsMap.keySet().stream().filter((id) -> (bagelLOLSTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (bagelDusekTextsMap.containsValue(arr[1])) {
                            bagelDusekTextsMap.keySet().stream().filter((id) -> (bagelDusekTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfHotelLOLSTextsMap.containsValue(arr[1])) {
                            sfHotelLOLSTextsMap.keySet().stream().filter((id) -> (sfHotelLOLSTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfHotelWenTextsMap.containsValue(arr[1])) {
                            sfHotelWenTextsMap.keySet().stream().filter((id) -> (sfHotelWenTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfRestLOLSTextsMap.containsValue(arr[1])) {
                            sfRestLOLSTextsMap.keySet().stream().filter((id) -> (sfRestLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                ids.add(id);
                            });
                        }
                        if (sfRestWenTextsMap.containsValue(arr[1])) {
                            sfRestWenTextsMap.keySet().stream().filter((id) -> (sfRestWenTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (controlTextsMap.containsValue(arr[1])) {
                            controlTextsMap.keySet().stream().filter((id) -> (controlTextsMap.get(id).equals(arr[1]))).map((id) -> id.substring(0, id.length() - 2)).forEachOrdered((cleanID) -> {
                                ids.add(cleanID);
                            });
                        }

                        if (ids.isEmpty()) {
                            if (bagelLOLSTextsMap.containsValue(arr[1])) {
                                bagelLOLSTextsMap.keySet().stream().filter((id) -> (bagelLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (bagelDusekTextsMap.containsValue(arr[1])) {
                                bagelDusekTextsMap.keySet().stream().filter((id) -> (bagelDusekTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfHotelLOLSTextsMap.containsValue(arr[1])) {
                                sfHotelLOLSTextsMap.keySet().stream().filter((id) -> (sfHotelLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfHotelWenTextsMap.containsValue(arr[1])) {
                                sfHotelWenTextsMap.keySet().stream().filter((id) -> (sfHotelWenTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfRestLOLSTextsMap.containsValue(arr[1])) {
                                sfRestLOLSTextsMap.keySet().stream().filter((id) -> (sfRestLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfRestWenTextsMap.containsValue(arr[1])) {
                                sfRestWenTextsMap.keySet().stream().filter((id) -> (sfRestWenTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                        }

                        if (ids.isEmpty()) {
                            System.exit(0);
                        }
                        for (String id : ids) {
                            scoresPerUser.get(email).get("Adequacy").put(id, Integer.parseInt(arr[2].trim()));
                        }
                    } else if (s.startsWith("Informativeness")) {
                        String[] arr = s.split("\t");
                        if (!scoresPerUser.containsKey(email)) {
                            scoresPerUser.put(email, new HashMap<String, HashMap<String, Integer>>());
                            scoresPerUser.get(email).put("Fluency", new HashMap<String, Integer>());
                            scoresPerUser.get(email).put("Adequacy", new HashMap<String, Integer>());
                        }
                        String parsedID = arr[0].substring("Informativeness".length()).trim();
                        HashSet<String> ids = new HashSet<>();
                        if (bagelLOLSTextsMap.containsValue(arr[1])) {
                            bagelLOLSTextsMap.keySet().stream().filter((id) -> (bagelLOLSTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (bagelDusekTextsMap.containsValue(arr[1])) {
                            bagelDusekTextsMap.keySet().stream().filter((id) -> (bagelDusekTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfHotelLOLSTextsMap.containsValue(arr[1])) {
                            sfHotelLOLSTextsMap.keySet().stream().filter((id) -> (sfHotelLOLSTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfHotelWenTextsMap.containsValue(arr[1])) {
                            sfHotelWenTextsMap.keySet().stream().filter((id) -> (sfHotelWenTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (sfRestLOLSTextsMap.containsValue(arr[1])) {
                            sfRestLOLSTextsMap.keySet().stream().filter((id) -> (sfRestLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                ids.add(id);
                            });
                        }
                        if (sfRestWenTextsMap.containsValue(arr[1])) {
                            sfRestWenTextsMap.keySet().stream().filter((id) -> (sfRestWenTextsMap.get(id).equals(arr[1])
                                    && (parsedID.equals(id)
                                            || parsedID.isEmpty()))).forEachOrdered((id) -> {
                                                ids.add(id);
                            });
                        }
                        if (controlTextsMap.containsValue(arr[1])) {
                            controlTextsMap.keySet().stream().filter((id) -> (controlTextsMap.get(id).equals(arr[1]))).map((id) -> id.substring(0, id.length() - 2)).forEachOrdered((cleanID) -> {
                                ids.add(cleanID);
                            });
                        }

                        if (ids.isEmpty()) {
                            if (bagelLOLSTextsMap.containsValue(arr[1])) {
                                bagelLOLSTextsMap.keySet().stream().filter((id) -> (bagelLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (bagelDusekTextsMap.containsValue(arr[1])) {
                                bagelDusekTextsMap.keySet().stream().filter((id) -> (bagelDusekTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfHotelLOLSTextsMap.containsValue(arr[1])) {
                                sfHotelLOLSTextsMap.keySet().stream().filter((id) -> (sfHotelLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfHotelWenTextsMap.containsValue(arr[1])) {
                                sfHotelWenTextsMap.keySet().stream().filter((id) -> (sfHotelWenTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfRestLOLSTextsMap.containsValue(arr[1])) {
                                sfRestLOLSTextsMap.keySet().stream().filter((id) -> (sfRestLOLSTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                            if (sfRestWenTextsMap.containsValue(arr[1])) {
                                sfRestWenTextsMap.keySet().stream().filter((id) -> (sfRestWenTextsMap.get(id).equals(arr[1]))).forEachOrdered((id) -> {
                                    ids.add(id);
                                });
                            }
                        }

                        if (ids.isEmpty()) {
                            System.exit(0);
                        }
                        for (String id : ids) {
                            scoresPerUser.get(email).get("Adequacy").put(id, Integer.parseInt(arr[2].trim()));
                        }
                    }
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        ArrayList<String> emails = new ArrayList<String>(scoresPerUser.keySet());
        Random r = new Random();
        System.out.println("********** LUCKY WINNERS ************");
        System.out.println(emails.get(r.nextInt(emails.size())));
        System.out.println(emails.get(r.nextInt(emails.size())));
        System.out.println(emails.get(r.nextInt(emails.size())));
        System.out.println("********** LUCKY WINNERS ************");
        System.out.println(new HashSet(ages.values()));
        System.out.println(new HashSet(languages.values()));
        System.out.println(new HashSet(educationLevel.values()));
        System.out.println(new HashSet(englishLevel.values()));

        HashSet<String> unreliableUserIDs = new HashSet<>();
        HashSet<String> unreliableUserIDsFL = new HashSet<>();
        HashSet<String> unreliableUserIDsAD = new HashSet<>();
        scoresPerUser.keySet().forEach((userID) -> {
            if (scoresPerUser.get(userID).get("Fluency").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("CONTROL-Med")
                    && scoresPerUser.get(userID).get("Fluency").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("CONTROL-Best")
                    && scoresPerUser.get(userID).get("Fluency").get("CONTROL-Med") <= scoresPerUser.get(userID).get("Fluency").get("CONTROL-Best")
                    && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Med")
                    && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Best")
                    && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Med") <= scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Best")) {
            } else {
                /*System.out.println(userID);
                System.out.println(scores.get(userID).get("Fluency").get("CONTROL-Worse") + " < " + scores.get(userID).get("Fluency").get("CONTROL-Med") + " <= " + scores.get(userID).get("Fluency").get("CONTROL-Best"));
                System.out.println(scores.get(userID).get("Adequacy").get("CONTROL-Worse") + " < " + scores.get(userID).get("Adequacy").get("CONTROL-Med") + " <= " + scores.get(userID).get("Adequacy").get("CONTROL-Best"));
                System.out.println("");*/
                unreliableUserIDs.add(userID);
                if (scoresPerUser.get(userID).get("Fluency").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("CONTROL-Med")
                        && scoresPerUser.get(userID).get("Fluency").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("CONTROL-Best")
                        && scoresPerUser.get(userID).get("Fluency").get("CONTROL-Med") <= scoresPerUser.get(userID).get("Fluency").get("CONTROL-Best")) {
                } else {
                    unreliableUserIDsFL.add(userID);
                }
                if (scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Med")
                        && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Best")
                        && scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Med") <= scoresPerUser.get(userID).get("Adequacy").get("CONTROL-Best")) {
                } else {
                    unreliableUserIDsAD.add(userID);
                }
            }
        });
        System.out.println("UNRELIABLE: " + (unreliableUserIDs.size() / ((double) scoresPerUser.keySet().size())));
        System.out.println("UNRELIABLE on Fluency: " + (unreliableUserIDsFL.size() / ((double) scoresPerUser.keySet().size())));
        System.out.println("UNRELIABLE on Adequacy: " + (unreliableUserIDsAD.size() / ((double) scoresPerUser.keySet().size())));

        System.out.println("Total Users: " + scoresPerUser.keySet().size());
        unreliableUserIDs.forEach((unreliableUserID) -> {
            scoresPerUser.remove(unreliableUserID);
        });
        System.out.println("Total Reliable Users: " + scoresPerUser.keySet().size());

        //AGREEMENT VECTORS
        HashSet<String> usedIDs = new HashSet<String>();
        for (String userID : scoresPerUser.keySet()) {
            if (!usedIDs.contains(userID)) {
                usedIDs.add(userID);
                HashSet<String> usersOfSameInstances = new HashSet<String>();
                usersOfSameInstances.add(userID);
                for (String userID2 : scoresPerUser.keySet()) {
                    if (!usedIDs.contains(userID2)) {
                        if (scoresPerUser.get(userID).get("Fluency").keySet().equals(scoresPerUser.get(userID2).get("Fluency").keySet())
                                && scoresPerUser.get(userID).get("Adequacy").keySet().equals(scoresPerUser.get(userID2).get("Adequacy").keySet())) {
                            usedIDs.add(userID2);
                            usersOfSameInstances.add(userID2);
                        }
                    }
                }
                if (usersOfSameInstances.size() > 1) {
                    ArrayList<String> instances = new ArrayList<>(scoresPerUser.get(userID).get("Fluency").keySet());
                    usersOfSameInstances.stream().map((ID) -> {
                        System.out.print("Fluency of " + ID + ":\t");
                        return ID;
                    }).map((ID) -> {
                        instances.forEach((instance) -> {
                            System.out.print(scoresPerUser.get(ID).get("Fluency").get(instance) + "\t");
                        });
                        return ID;
                    }).forEachOrdered((_item) -> {
                        System.out.println();
                    });
                }
            }
        }
        System.out.println("===============");
        usedIDs = new HashSet<String>();
        for (String userID : scoresPerUser.keySet()) {
            if (!usedIDs.contains(userID)) {
                usedIDs.add(userID);
                HashSet<String> usersOfSameInstances = new HashSet<String>();
                usersOfSameInstances.add(userID);
                for (String userID2 : scoresPerUser.keySet()) {
                    if (!usedIDs.contains(userID2)) {
                        if (scoresPerUser.get(userID).get("Fluency").keySet().equals(scoresPerUser.get(userID2).get("Fluency").keySet())
                                && scoresPerUser.get(userID).get("Adequacy").keySet().equals(scoresPerUser.get(userID2).get("Adequacy").keySet())) {
                            usedIDs.add(userID2);
                            usersOfSameInstances.add(userID2);
                        }
                    }
                }
                if (usersOfSameInstances.size() > 1) {
                    ArrayList<String> instances = new ArrayList<>(scoresPerUser.get(userID).get("Adequacy").keySet());
                    usersOfSameInstances.stream().map((ID) -> {
                        System.out.print("Adequacy of " + ID + ":\t");
                        return ID;
                    }).map((ID) -> {
                        instances.forEach((instance) -> {
                            System.out.print(scoresPerUser.get(ID).get("Adequacy").get(instance) + "\t");
                        });
                        return ID;
                    }).forEachOrdered((_item) -> {
                        System.out.println();
                    });
                }
            }
        }

        HashMap<String, ArrayList<Integer>> scoresPerInstanceFluency = new HashMap<>();
        HashMap<String, ArrayList<Integer>> scoresPerInstanceAdequacy = new HashMap<>();
        scoresPerUser.keySet().stream().map((userID) -> {
            scoresPerUser.get(userID).get("Fluency").keySet().stream().filter((instance) -> (!instance.equals("CONTROL-Worse")
                    && !instance.equals("CONTROL-Med")
                    && !instance.equals("CONTROL-Best"))).map((instance) -> {
                        if (!scoresPerInstanceFluency.containsKey(instance)) {
                            scoresPerInstanceFluency.put(instance, new ArrayList<Integer>());
                        }
                return instance;
            }).forEachOrdered((instance) -> {
                scoresPerInstanceFluency.get(instance).add(scoresPerUser.get(userID).get("Fluency").get(instance));
            });
            return userID;
        }).forEachOrdered((userID) -> {
            scoresPerUser.get(userID).get("Adequacy").keySet().stream().filter((instance) -> (!instance.equals("CONTROL-Worse")
                    && !instance.equals("CONTROL-Med")
                    && !instance.equals("CONTROL-Best"))).map((instance) -> {
                        if (!scoresPerInstanceAdequacy.containsKey(instance)) {
                            scoresPerInstanceAdequacy.put(instance, new ArrayList<Integer>());
                        }
                return instance;
            }).forEachOrdered((instance) -> {
                scoresPerInstanceAdequacy.get(instance).add(scoresPerUser.get(userID).get("Adequacy").get(instance));
            });
        });

        ArrayList<Double> LOLS_BAGEL_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFHOT_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFRES_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> WEN_SFHOT_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> WEN_SFRES_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> Dusek_AVG_fluency_set = new ArrayList<>();

        Double LOLS_BAGEL_AVG_fluency = 0.0;
        Double LOLS_SFHOT_AVG_fluency = 0.0;
        Double LOLS_SFRES_AVG_fluency = 0.0;
        Double WEN_SFHOT_AVG_fluency = 0.0;
        Double WEN_SFRES_AVG_fluency = 0.0;
        Double Dusek_AVG_fluency = 0.0;

        Double LOLS_BAGEL_total_fluency = 0.0;
        Double LOLS_SFHOT_total_fluency = 0.0;
        Double LOLS_SFRES_total_fluency = 0.0;
        Double WEN_SFHOT_total_fluency = 0.0;
        Double WEN_SFRES_total_fluency = 0.0;
        Double Dusek_total_fluency = 0.0;

        String LOLS_BAGEL_scores_fl = "";
        String LOLS_SFHOT_scores_fl = "";
        String LOLS_SFRES_scores_fl = "";
        String WEN_SFHOT_scores_fl = "";
        String WEN_SFRES_scores_fl = "";
        String Dusek_scores_fl = "";
        for (String instance : scoresPerInstanceFluency.keySet()) {
            if (instance.startsWith("LOLS_BAGEL")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    if (containsID("Dusek" + instance.substring("LOLS_BAGEL".length()), new HashSet<String>(scoresPerInstanceFluency.keySet()))) {
                        LOLS_BAGEL_AVG_fluency += score.doubleValue();
                        LOLS_BAGEL_total_fluency++;
                        LOLS_BAGEL_AVG_fluency_set.add(score.doubleValue());
                        LOLS_BAGEL_scores_fl += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("LOLS_SFHOT")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    if (containsID("WEN_SFHOT" + instance.substring("LOLS_SFHOT".length()), new HashSet<String>(scoresPerInstanceFluency.keySet()))) {
                        LOLS_SFHOT_AVG_fluency += score.doubleValue();
                        LOLS_SFHOT_total_fluency++;
                        LOLS_SFHOT_AVG_fluency_set.add(score.doubleValue());
                        LOLS_SFHOT_scores_fl += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("LOLS_SFRES")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    if (containsID("WEN_SFRES" + instance.substring("LOLS_SFRES".length()), new HashSet<String>(scoresPerInstanceFluency.keySet()))) {
                        LOLS_SFRES_AVG_fluency += score.doubleValue();
                        LOLS_SFRES_total_fluency++;
                        LOLS_SFRES_AVG_fluency_set.add(score.doubleValue());
                        LOLS_SFRES_scores_fl += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("WEN_SFHOT")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    if (containsID("LOLS_SFHOT" + instance.substring("WEN_SFHOT".length()), new HashSet<String>(scoresPerInstanceFluency.keySet()))) {
                        WEN_SFHOT_AVG_fluency += score.doubleValue();
                        WEN_SFHOT_total_fluency++;
                        WEN_SFHOT_AVG_fluency_set.add(score.doubleValue());
                        WEN_SFHOT_scores_fl += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("WEN_SFRES")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    if (containsID("LOLS_SFRES" + instance.substring("WEN_SFRES".length()), new HashSet<String>(scoresPerInstanceFluency.keySet()))) {
                        WEN_SFRES_AVG_fluency += score.doubleValue();
                        WEN_SFRES_total_fluency++;
                        WEN_SFRES_AVG_fluency_set.add(score.doubleValue());
                        WEN_SFRES_scores_fl += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("Dusek")) {
                for (Integer score : scoresPerInstanceFluency.get(instance)) {
                    if (containsID("LOLS_BAGEL" + instance.substring("Dusek".length()), new HashSet<String>(scoresPerInstanceFluency.keySet()))) {
                        Dusek_AVG_fluency += score.doubleValue();
                        Dusek_total_fluency++;
                        Dusek_AVG_fluency_set.add(score.doubleValue());
                        Dusek_scores_fl += "\t" + score.doubleValue();
                    }
                }
            }
        }
        LOLS_BAGEL_AVG_fluency /= LOLS_BAGEL_total_fluency;
        LOLS_SFHOT_AVG_fluency /= LOLS_SFHOT_total_fluency;
        LOLS_SFRES_AVG_fluency /= LOLS_SFRES_total_fluency;
        WEN_SFHOT_AVG_fluency /= WEN_SFHOT_total_fluency;
        WEN_SFRES_AVG_fluency /= WEN_SFRES_total_fluency;
        Dusek_AVG_fluency /= Dusek_total_fluency;

        ArrayList<Double> LOLS_BAGEL_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFHOT_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFRES_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> WEN_SFHOT_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> WEN_SFRES_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> Dusek_AVG_adequacy_set = new ArrayList<>();

        Double LOLS_BAGEL_AVG_adequacy = 0.0;
        Double LOLS_SFHOT_AVG_adequacy = 0.0;
        Double LOLS_SFRES_AVG_adequacy = 0.0;
        Double WEN_SFHOT_AVG_adequacy = 0.0;
        Double WEN_SFRES_AVG_adequacy = 0.0;
        Double Dusek_AVG_adequacy = 0.0;

        Double LOLS_BAGEL_total_adequacy = 0.0;
        Double LOLS_SFHOT_total_adequacy = 0.0;
        Double LOLS_SFRES_total_adequacy = 0.0;
        Double WEN_SFHOT_total_adequacy = 0.0;
        Double WEN_SFRES_total_adequacy = 0.0;
        Double Dusek_total_adequacy = 0.0;

        ArrayList<String> keys = new ArrayList<>(scoresPerInstanceAdequacy.keySet());
        String LOLS_BAGEL_scores_ad = "";
        String LOLS_SFHOT_scores_ad = "";
        String LOLS_SFRES_scores_ad = "";
        String WEN_SFHOT_scores_ad = "";
        String WEN_SFRES_scores_ad = "";
        String Dusek_scores_ad = "";
        Collections.sort(keys);
        for (String instance : keys) {
            if (instance.startsWith("LOLS_BAGEL")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    if (containsID("Dusek" + instance.substring("LOLS_BAGEL".length()), new HashSet<String>(scoresPerInstanceAdequacy.keySet()))) {
                        LOLS_BAGEL_AVG_adequacy += score.doubleValue();
                        LOLS_BAGEL_total_adequacy++;
                        LOLS_BAGEL_AVG_adequacy_set.add(score.doubleValue());
                        LOLS_BAGEL_scores_ad += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("LOLS_SFHOT")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    if (containsID("WEN_SFHOT" + instance.substring("LOLS_SFHOT".length()), new HashSet<String>(scoresPerInstanceAdequacy.keySet()))) {
                        LOLS_SFHOT_AVG_adequacy += score.doubleValue();
                        LOLS_SFHOT_total_adequacy++;
                        LOLS_SFHOT_AVG_adequacy_set.add(score.doubleValue());
                        LOLS_SFHOT_scores_ad += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("LOLS_SFRES")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    if (containsID("WEN_SFRES" + instance.substring("LOLS_SFRES".length()), new HashSet<String>(scoresPerInstanceAdequacy.keySet()))) {
                        LOLS_SFRES_AVG_adequacy += score.doubleValue();
                        LOLS_SFRES_total_adequacy++;
                        LOLS_SFRES_AVG_adequacy_set.add(score.doubleValue());
                        LOLS_SFRES_scores_ad += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("WEN_SFHOT")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    if (containsID("LOLS_SFHOT" + instance.substring("WEN_SFHOT".length()), new HashSet<String>(scoresPerInstanceAdequacy.keySet()))) {
                        WEN_SFHOT_AVG_adequacy += score.doubleValue();
                        WEN_SFHOT_total_adequacy++;
                        WEN_SFHOT_AVG_adequacy_set.add(score.doubleValue());
                        WEN_SFHOT_scores_ad += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("WEN_SFRES")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    if (containsID("LOLS_SFRES" + instance.substring("WEN_SFRES".length()), new HashSet<String>(scoresPerInstanceAdequacy.keySet()))) {
                        WEN_SFRES_AVG_adequacy += score.doubleValue();
                        WEN_SFRES_total_adequacy++;
                        WEN_SFRES_AVG_adequacy_set.add(score.doubleValue());
                        WEN_SFRES_scores_ad += "\t" + score.doubleValue();
                    }
                }
            } else if (instance.startsWith("Dusek")) {
                for (Integer score : scoresPerInstanceAdequacy.get(instance)) {
                    if (containsID("LOLS_BAGEL" + instance.substring("Dusek".length()), new HashSet<String>(scoresPerInstanceAdequacy.keySet()))) {
                        Dusek_AVG_adequacy += score.doubleValue();
                        Dusek_total_adequacy++;
                        Dusek_AVG_adequacy_set.add(score.doubleValue());
                        Dusek_scores_ad += "\t" + score.doubleValue();
                    }
                }
            }
        }
        LOLS_BAGEL_AVG_adequacy /= LOLS_BAGEL_total_adequacy;
        LOLS_SFHOT_AVG_adequacy /= LOLS_SFHOT_total_adequacy;
        LOLS_SFRES_AVG_adequacy /= LOLS_SFRES_total_adequacy;
        WEN_SFHOT_AVG_adequacy /= WEN_SFHOT_total_adequacy;
        WEN_SFRES_AVG_adequacy /= WEN_SFRES_total_adequacy;
        Dusek_AVG_adequacy /= Dusek_total_adequacy;

        System.out.println();
        System.out.println("BAGEL\tLOLS\tDusek");
        System.out.println("Fluency\t" + LOLS_BAGEL_AVG_fluency + "(" + getConf(LOLS_BAGEL_AVG_fluency_set, LOLS_BAGEL_AVG_fluency, LOLS_BAGEL_total_fluency) + ")" + "\t" + Dusek_AVG_fluency + "(" + getConf(Dusek_AVG_fluency_set, Dusek_AVG_fluency, Dusek_total_fluency) + ")");
        System.out.println("Adequacy\t" + LOLS_BAGEL_AVG_adequacy + "(" + getConf(LOLS_BAGEL_AVG_adequacy_set, LOLS_BAGEL_AVG_adequacy, LOLS_BAGEL_total_adequacy) + ")" + "\t" + Dusek_AVG_adequacy + "(" + getConf(Dusek_AVG_adequacy_set, Dusek_AVG_adequacy, Dusek_total_adequacy) + ")");
        System.out.println();
        System.out.println("SF HOTEL\tLOLS\tWen");
        System.out.println("Fluency\t" + LOLS_SFHOT_AVG_fluency + "(" + getConf(LOLS_SFHOT_AVG_fluency_set, LOLS_SFHOT_AVG_fluency, LOLS_SFHOT_total_fluency) + ")" + "\t" + WEN_SFHOT_AVG_fluency + "(" + getConf(WEN_SFHOT_AVG_fluency_set, WEN_SFHOT_AVG_fluency, WEN_SFHOT_total_fluency) + ")");
        System.out.println("Adequacy\t" + LOLS_SFHOT_AVG_adequacy + "(" + getConf(LOLS_SFHOT_AVG_adequacy_set, LOLS_SFHOT_AVG_adequacy, LOLS_SFHOT_total_adequacy) + ")" + "\t" + WEN_SFHOT_AVG_adequacy + "(" + getConf(WEN_SFHOT_AVG_adequacy_set, WEN_SFHOT_AVG_adequacy, WEN_SFHOT_total_adequacy) + ")");
        System.out.println();
        System.out.println("SF REST\tLOLS\tWen");
        System.out.println("Fluency\t" + LOLS_SFRES_AVG_fluency + "(" + getConf(LOLS_SFRES_AVG_fluency_set, LOLS_SFRES_AVG_fluency, LOLS_SFRES_total_fluency) + ")" + "\t" + WEN_SFRES_AVG_fluency + "(" + getConf(WEN_SFRES_AVG_fluency_set, WEN_SFRES_AVG_fluency, WEN_SFRES_total_fluency) + ")");
        System.out.println("Adequacy\t" + LOLS_SFRES_AVG_adequacy + "(" + getConf(LOLS_SFRES_AVG_adequacy_set, LOLS_SFRES_AVG_adequacy, LOLS_SFRES_total_adequacy) + ")" + "\t" + WEN_SFRES_AVG_adequacy + "(" + getConf(WEN_SFRES_AVG_adequacy_set, WEN_SFRES_AVG_adequacy, WEN_SFRES_total_adequacy) + ")");
        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println("Dusek\t" + Dusek_scores_fl);
        System.out.println("LOLS_BAGEL\t" + LOLS_BAGEL_scores_fl);
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println("Dusek\t" + Dusek_scores_ad);
        System.out.println("LOLS_BAGEL\t" + LOLS_BAGEL_scores_ad);
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println("WEN_SFHOT\t" + WEN_SFHOT_scores_fl);
        System.out.println("LOLS_SFHOT\t" + LOLS_SFHOT_scores_fl);
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println("WEN_SFHOT\t" + WEN_SFHOT_scores_ad);
        System.out.println("LOLS_SFHOT\t" + LOLS_SFHOT_scores_ad);
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println("WEN_SFRES\t" + WEN_SFRES_scores_fl);
        System.out.println("LOLS_SFRES\t" + LOLS_SFRES_scores_fl);
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println("WEN_SFRES\t" + WEN_SFRES_scores_ad);
    }
    
    /**
     *
     */
    public static void checkIfTextsInTexts() {
        String oneFile = "1.txt";
        String twoFile = "2.txt";

        HashSet<String> oneTexts = new HashSet<>();
        HashSet<String> twoTexts = new HashSet<>();
        HashMap<String,String> newTextsMRMap = new HashMap<>();
        HashMap<String,String> oldTextsMRMap = new HashMap<>();
        HashMap<String,String> oldTextsMap = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(oneFile))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] arr = s.trim().split("\t");              
                    newTextsMRMap.put(arr[0].trim(), arr[1].trim());      
                    oneTexts.add(arr[1].replaceAll(" ", "").trim().toLowerCase());
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (BufferedReader br = new BufferedReader(new FileReader(twoFile))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] arr = s.trim().split("\t");
                    String orig = arr[1].trim();
                    twoTexts.add(arr[1].replaceAll(" ", "").trim().toLowerCase());
                    oldTextsMRMap.put(arr[1].replaceAll(" ", "").trim().toLowerCase(), arr[0].trim());
                    oldTextsMap.put(arr[1].replaceAll(" ", "").trim().toLowerCase(), orig);
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        twoTexts.stream().filter((s2) -> (!oneTexts.contains(s2))).forEachOrdered((s2) -> {
            System.out.println(oldTextsMRMap.get(s2) + "\t" + oldTextsMap.get(s2) + "\t" + newTextsMRMap.get(oldTextsMRMap.get(s2)));
        });
    }

    /**
     *
     */
    public static void parseGScriptQuestionnaires() {
        String MRFile = "NLG_EvalMRs.txt";
        String textsFile = "NLG_EvalTexts_fb.txt";
        String evalFile = "NLG_EvalAnswers_fb_withFixedDispar.txt";

        HashMap<String, HashSet<String>> mrsSet = new HashMap<>();

        HashMap<String, HashSet<String>> mrsSFHOTELSet = new HashMap<>();
        HashMap<String, HashSet<String>> mrsSFRESTSet = new HashMap<>();
        HashMap<String, HashSet<String>> mrsBAGELSet = new HashMap<>();

        HashMap<String, String> mrs = new HashMap<>();
        HashMap<String, String> mrsSFHOTEL = new HashMap<>();
        HashMap<String, String> mrsSFREST = new HashMap<>();
        HashMap<String, String> mrsBAGEL = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(MRFile))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] texts = s.trim().split("\t");
                    if (texts[0].startsWith("SFHOT")) {
                        HashSet<String> mrSet = new HashSet<>();
                        mrSet.add(texts[1].trim().toLowerCase());

                        mrsSFHOTELSet.put(texts[0].trim(), mrSet);
                        mrsSet.put(texts[0].trim(), mrSet);
                        mrsSFHOTEL.put(texts[0].trim(), texts[1].trim().toLowerCase());
                        mrs.put(texts[0].trim(), texts[1].trim().toLowerCase());
                    } else if (texts[0].startsWith("SFRES")) {
                        HashSet<String> mrSet = new HashSet<>();
                        mrSet.add(texts[1].trim().toLowerCase());

                        mrsSFRESTSet.put(texts[0].trim(), mrSet);
                        mrsSet.put(texts[0].trim(), mrSet);
                        mrsSFREST.put(texts[0].trim(), texts[1].trim().toLowerCase());
                        mrs.put(texts[0].trim(), texts[1].trim().toLowerCase());
                    } else if (texts[0].startsWith("BAGEL")) {
                        String mr = texts[1].trim().substring(7, texts[1].trim().length() - 1).toLowerCase();
                        HashSet<String> mrSet = new HashSet<>();
                        Collections.addAll(mrSet, mr.split(","));

                        mrsBAGELSet.put(texts[0].trim(), mrSet);
                        mrsSet.put(texts[0].trim(), mrSet);
                        mrsBAGEL.put(texts[0].trim(), texts[1].trim().toLowerCase());
                        mrs.put(texts[0].trim(), texts[1].trim().toLowerCase());
                    }
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }

        HashMap<String, String> abstractMRs = new HashMap<>();
        mrs.values().forEach((MR) -> {
            HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
            String MRstr = MR.substring(MR.indexOf(':') + 1).replaceAll(",", ";").replaceAll("no or yes", "yes or no").replaceAll("ave ; presidio", "ave and presidio").replaceAll("point ; ste", "point and ste").trim();
            String predicate = MRstr.substring(0, MRstr.indexOf('('));
            String abstractMR = predicate + ":";
            String attributesStr = MRstr.substring(MRstr.indexOf('(') + 1, MRstr.length() - 1);
            attributeValues = new HashMap<>();
            if (!attributesStr.isEmpty()) {
                HashMap<String, Integer> attrXIndeces = new HashMap<>();
                String[] args = attributesStr.split(";");
                if (attributesStr.contains("|")) {
                    System.exit(0);
                }
                for (String arg : args) {
                    String attr = "";
                    String value = "";
                    if (arg.contains("=")) {
                        String[] subAttr = arg.split("=");
                        value = subAttr[1].toLowerCase();
                        attr = subAttr[0].toLowerCase().replaceAll("_", "");

                        if (value.startsWith("\'")) {
                            value = value.substring(1, value.length() - 1);
                        }
                        if (value.equals("true")) {
                            value = "yes";
                        }
                        if (value.equals("false")) {
                            value = "no";
                        }
                        if (value.equals("dontcare")) {
                            value = "dont_care";
                        }
                        if (value.equals("no")
                                || value.equals("yes")
                                || value.equals("yes or no")
                                || value.equals("none")
                                || value.equals("empty")) {
                            attr += "_" + value.replaceAll(" ", "_");
                            value = attr;
                        }
                        if (value.equals("dont_care")) {
                            String v = value;
                            value = attr;
                            attr = v;
                        }
                    } else {
                        attr = arg.replaceAll("_", "");
                    }
                    if (!attributeValues.containsKey(attr)) {
                        attributeValues.put(attr, new HashSet<String>());
                    }
                    if (value.isEmpty()) {
                        value = attr;
                    }
                    if (value.startsWith("\'")) {
                        value = value.substring(1, value.length() - 1);
                    }
                    if (value.toLowerCase().startsWith("x")) {
                        int index = 0;
                        if (!attrXIndeces.containsKey(attr)) {
                            attrXIndeces.put(attr, 1);
                        } else {
                            index = attrXIndeces.get(attr);
                            attrXIndeces.put(attr, index + 1);
                        }
                        value = "x" + index;
                    }
                    if (value.isEmpty()) {
                        System.exit(0);
                    }
                    attributeValues.get(attr).add(value.trim().toLowerCase());
                }
            }
            ArrayList<String> attrs = new ArrayList<>(attributeValues.keySet());
            Collections.sort(attrs);
            HashMap<String, Integer> xCounts = new HashMap<>();
            attrs.forEach((attr) -> {
                xCounts.put(attr, 0);
            });
            for (String attr : attrs) {
                abstractMR += attr + "={";

                ArrayList<String> values = new ArrayList<>(attributeValues.get(attr));
                Collections.sort(values);
                for (String value : values) {

                    if (attr.equals("name")
                            || attr.equals("type")
                            || attr.equals("pricerange")
                            || attr.equals("price")
                            || attr.equals("phone")
                            || attr.equals("address")
                            || attr.equals("postcode")
                            || attr.equals("area")
                            || attr.equals("near")
                            || attr.equals("food")
                            || attr.equals("goodformeal")
                            || attr.equals("count")) {
                        abstractMR += Action.TOKEN_X + attr + "_" + xCounts.get(attr) + ",";
                        xCounts.put(attr, xCounts.get(attr) + 1);
                    } else {
                        abstractMR += value + ",";
                    }
                }
                abstractMR += "}";
            }
            abstractMRs.put(MR, abstractMR);
        });

        HashSet<String> uniqueTextsSFHOTEL = new HashSet<>();
        HashSet<String> uniqueTextsSFREST = new HashSet<>();
        HashSet<String> uniqueTextsBAGEL = new HashSet<>();

        HashMap<String, HashSet<String>> mrsPerEvalInstanceID = new HashMap<>();
        HashMap<String, String> textsPerEvalInstanceID = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(textsFile))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    String[] texts = s.trim().split("\t");
                    String ID = "INID_" + texts[0].trim() + "-" + texts[1].trim();

                    HashSet<String> mrSet = new HashSet<>();
                    if (ID.contains("BAGEL")) {
                        String mr = texts[2].trim().substring(7, texts[2].trim().length() - 1).toLowerCase();
                        Collections.addAll(mrSet, mr.split(","));
                    } else {
                        mrSet.add(texts[2].trim().toLowerCase());
                    }
                    mrsPerEvalInstanceID.put(ID, mrSet);
                    textsPerEvalInstanceID.put(ID, texts[3].trim());

                    if (ID.contains("LOLS")) {
                        if (ID.contains("BAGEL")) {
                            uniqueTextsBAGEL.add(texts[3].trim());
                        } else if (ID.contains("SFHOT")) {
                            uniqueTextsSFHOTEL.add(texts[3].trim());
                        } else if (ID.contains("SFRES")) {
                            uniqueTextsSFREST.add(texts[3].trim());
                        }
                    }
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println(uniqueTextsBAGEL);
        System.out.println(uniqueTextsBAGEL.size());
        System.out.println(uniqueTextsSFHOTEL);
        System.out.println(uniqueTextsSFHOTEL.size());
        System.out.println(uniqueTextsSFREST);
        System.out.println(uniqueTextsSFREST.size());

        HashMap<String, HashSet<String>> evalInstanceIDToMrID_LOLS = new HashMap<>();
        mrsPerEvalInstanceID.keySet().stream().filter((evalInstanceID) -> (evalInstanceID.contains("LOLS"))).map((evalInstanceID) -> {
            if (evalInstanceID.contains("BAGEL")) {
                mrsBAGELSet.keySet().stream().filter((mrID) -> (mrsBAGELSet.get(mrID).equals(mrsPerEvalInstanceID.get(evalInstanceID)))).forEachOrdered((mrID) -> {
                    if (!evalInstanceIDToMrID_LOLS.containsKey(evalInstanceID)) {
                        evalInstanceIDToMrID_LOLS.put(evalInstanceID, new HashSet<String>());
                    }
                    evalInstanceIDToMrID_LOLS.get(evalInstanceID).add(mrID);
                });
            } else if (evalInstanceID.contains("SFHOT")) {
                mrsSFHOTELSet.keySet().stream().filter((mrID) -> (mrsSFHOTELSet.get(mrID).equals(mrsPerEvalInstanceID.get(evalInstanceID)))).forEachOrdered((mrID) -> {
                    if (!evalInstanceIDToMrID_LOLS.containsKey(evalInstanceID)) {
                        evalInstanceIDToMrID_LOLS.put(evalInstanceID, new HashSet<String>());
                    }
                    evalInstanceIDToMrID_LOLS.get(evalInstanceID).add(mrID);
                });
            } else if (evalInstanceID.contains("SFRES")) {
                mrsSFRESTSet.keySet().stream().filter((mrID) -> (mrsSFRESTSet.get(mrID).equals(mrsPerEvalInstanceID.get(evalInstanceID)))).forEachOrdered((mrID) -> {
                    if (!evalInstanceIDToMrID_LOLS.containsKey(evalInstanceID)) {
                        evalInstanceIDToMrID_LOLS.put(evalInstanceID, new HashSet<String>());
                    }
                    evalInstanceIDToMrID_LOLS.get(evalInstanceID).add(mrID);
                });
            }
            return evalInstanceID;
        }).filter((evalInstanceID) -> (!evalInstanceIDToMrID_LOLS.containsKey(evalInstanceID) || evalInstanceIDToMrID_LOLS.get(evalInstanceID).isEmpty())).map((evalInstanceID) -> {
            System.out.println("WTF " + evalInstanceID);
            return evalInstanceID;
        }).forEachOrdered((_item) -> {
            System.exit(0);
        });

        HashMap<String, HashSet<String>> evalInstanceIDToMrID_OTHER = new HashMap<>();
        mrsPerEvalInstanceID.keySet().stream().filter((evalInstanceID) -> (!evalInstanceID.contains("LOLS"))).map((evalInstanceID) -> {
            if (evalInstanceID.contains("BAGEL")) {
                mrsBAGELSet.keySet().stream().filter((mrID) -> (mrsBAGELSet.get(mrID).equals(mrsPerEvalInstanceID.get(evalInstanceID)))).forEachOrdered((mrID) -> {
                    if (!evalInstanceIDToMrID_OTHER.containsKey(evalInstanceID)) {
                        evalInstanceIDToMrID_OTHER.put(evalInstanceID, new HashSet<String>());
                    }
                    evalInstanceIDToMrID_OTHER.get(evalInstanceID).add(mrID);
                });
            } else if (evalInstanceID.contains("SFHOT")) {
                mrsSFHOTELSet.keySet().stream().filter((mrID) -> (mrsSFHOTELSet.get(mrID).equals(mrsPerEvalInstanceID.get(evalInstanceID)))).forEachOrdered((mrID) -> {
                    if (!evalInstanceIDToMrID_OTHER.containsKey(evalInstanceID)) {
                        evalInstanceIDToMrID_OTHER.put(evalInstanceID, new HashSet<String>());
                    }
                    evalInstanceIDToMrID_OTHER.get(evalInstanceID).add(mrID);
                });
            } else if (evalInstanceID.contains("SFRES")) {
                mrsSFRESTSet.keySet().stream().filter((mrID) -> (mrsSFRESTSet.get(mrID).equals(mrsPerEvalInstanceID.get(evalInstanceID)))).forEachOrdered((mrID) -> {
                    if (!evalInstanceIDToMrID_OTHER.containsKey(evalInstanceID)) {
                        evalInstanceIDToMrID_OTHER.put(evalInstanceID, new HashSet<String>());
                    }
                    evalInstanceIDToMrID_OTHER.get(evalInstanceID).add(mrID);
                });
            }
            return evalInstanceID;
        }).filter((evalInstanceID) -> (!evalInstanceIDToMrID_OTHER.containsKey(evalInstanceID) || evalInstanceIDToMrID_OTHER.get(evalInstanceID).isEmpty())).map((evalInstanceID) -> {
            System.out.println("WTF " + evalInstanceID);
            return evalInstanceID;
        }).map((evalInstanceID) -> {
            System.out.println("WTF " + mrsPerEvalInstanceID.get(evalInstanceID));
            return evalInstanceID;
        }).forEachOrdered((_item) -> {
            System.exit(0);
        });

        /*HashMap<String, HashSet<String>> mrIDToEvalInstanceID_LOLS = new HashMap<>();
        for (String mrID : mrsBAGEL.keySet()) {
        mrIDToEvalInstanceID_LOLS.put(mrID, new HashSet<String>());
        for (String evalInstanceID : evalInstanceIDToMrID_LOLS.keySet()) {
        if (evalInstanceIDToMrID_LOLS.get(evalInstanceID).contains(mrID)) {
        mrIDToEvalInstanceID_LOLS.get(mrID).add(evalInstanceID);
        }
        }
        if (mrIDToEvalInstanceID_LOLS.get(mrID).isEmpty()) {
        System.out.println("!!WTF " + mrID);
        System.out.println("!!WTF " + mrIDToEvalInstanceID_LOLS.get(mrID));
        System.out.println("!!WTF " + mrsBAGEL.get(mrID));
        System.out.println("!!WTF " + evalInstanceIDToMrID_LOLS);
        System.exit(0);
        }
        }
        for (String mrID : mrsSFHOTEL.keySet()) {
        mrIDToEvalInstanceID_LOLS.put(mrID, new HashSet<String>());
        for (String evalInstanceID : evalInstanceIDToMrID_LOLS.keySet()) {
        if (evalInstanceIDToMrID_LOLS.get(evalInstanceID).contains(mrID)) {
        mrIDToEvalInstanceID_LOLS.get(mrID).add(evalInstanceID);
        }
        }
        if (mrIDToEvalInstanceID_LOLS.get(mrID).isEmpty()) {
        System.out.println("!WTF " + mrID);
        System.out.println("!WTF " + mrsSFHOTEL.get(mrID));
        System.exit(0);
        }
        }
        for (String mrID : mrsSFREST.keySet()) {
        mrIDToEvalInstanceID_LOLS.put(mrID, new HashSet<String>());
        for (String evalInstanceID : evalInstanceIDToMrID_LOLS.keySet()) {
        if (evalInstanceIDToMrID_LOLS.get(evalInstanceID).contains(mrID)) {
        mrIDToEvalInstanceID_LOLS.get(mrID).add(evalInstanceID);
        }
        }
        if (mrIDToEvalInstanceID_LOLS.get(mrID).isEmpty()) {
        System.out.println("!WTF " + mrID);
        System.out.println("!WTF " + mrsSFREST.get(mrID));
        System.exit(0);
        }
        }
        
        HashMap<String, HashSet<String>> mrIDToEvalInstanceID_OTHER = new HashMap<>();
        for (String mrID : mrsBAGEL.keySet()) {
        mrIDToEvalInstanceID_OTHER.put(mrID, new HashSet<String>());
        for (String evalInstanceID : evalInstanceIDToMrID_OTHER.keySet()) {
        if (evalInstanceIDToMrID_OTHER.get(evalInstanceID).contains(mrID)) {
        mrIDToEvalInstanceID_OTHER.get(mrID).add(evalInstanceID);
        }
        }
        if (mrIDToEvalInstanceID_OTHER.get(mrID).isEmpty()) {
        System.out.println("!WTF " + mrID);
        System.out.println("!WTF " + mrsBAGEL.get(mrID));
        System.out.println("!WTF " + evalInstanceIDToMrID_OTHER);
        System.exit(0);
        }
        }
        for (String mrID : mrsSFHOTEL.keySet()) {
        mrIDToEvalInstanceID_OTHER.put(mrID, new HashSet<String>());
        for (String evalInstanceID : evalInstanceIDToMrID_OTHER.keySet()) {
        if (evalInstanceIDToMrID_OTHER.get(evalInstanceID).contains(mrID)) {
        mrIDToEvalInstanceID_OTHER.get(mrID).add(evalInstanceID);
        }
        }
        if (mrIDToEvalInstanceID_OTHER.get(mrID).isEmpty()) {
        System.out.println("!WTF " + mrID);
        System.out.println("!WTF " + mrsSFHOTEL.get(mrID));
        System.exit(0);
        }
        }
        for (String mrID : mrsSFREST.keySet()) {
        mrIDToEvalInstanceID_OTHER.put(mrID, new HashSet<String>());
        for (String evalInstanceID : evalInstanceIDToMrID_OTHER.keySet()) {
        if (evalInstanceIDToMrID_OTHER.get(evalInstanceID).contains(mrID)) {
        mrIDToEvalInstanceID_OTHER.get(mrID).add(evalInstanceID);
        }
        }
        if (mrIDToEvalInstanceID_OTHER.get(mrID).isEmpty()) {
        System.out.println("!WTF " + mrID);
        System.out.println("!WTF " + mrsSFREST.get(mrID));
        System.exit(0);
        }
        }
        System.out.println(mrIDToEvalInstanceID_LOLS);
        System.out.println(evalInstanceIDToMrID_OTHER);
        System.exit(0);*/
        HashMap<String, String> ages = new HashMap<>();
        HashMap<String, String> languages = new HashMap<>();
        HashMap<String, String> educationLevel = new HashMap<>();
        HashMap<String, String> englishLevel = new HashMap<>();
        HashMap<String, HashMap<String, HashMap<String, Integer>>> scoresPerUser = new HashMap<>();
        //PRINT RESULTS
        int emptyMailID = 1;
        try (BufferedReader br = new BufferedReader(new FileReader(evalFile))) {
            String s;
            String email = "";
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    if (s.startsWith("E-mail address")) {
                        email = s.substring("E-mail address (if you want a chance to win a £20 Amazon Voucher):".length() + 1).trim();
                        if (email.trim().isEmpty()) {
                            email = "emptyMail_" + emptyMailID;
                            emptyMailID++;
                        }
                    } else if (s.startsWith("How old are you?")) {
                        ages.put(email, s.substring("How old are you?".length()).trim());
                    } else if (s.startsWith("What is your native language?")) {
                        languages.put(email, s.substring("What is your native language?".length()).trim());
                    } else if (s.startsWith("What is your current education level?")) {
                        educationLevel.put(email, s.substring("What is your current education level?".length()).trim());
                    } else if (s.startsWith("What is your English level?")) {
                        englishLevel.put(email, s.substring("What is your English level?".length()).trim());
                    } else if (s.startsWith("Fluency")) {
                        String[] arr = s.split("\t");
                        if (!scoresPerUser.containsKey(email)) {
                            scoresPerUser.put(email, new HashMap<String, HashMap<String, Integer>>());
                            scoresPerUser.get(email).put("Fluency", new HashMap<String, Integer>());
                            scoresPerUser.get(email).put("Adequacy", new HashMap<String, Integer>());
                        }
                        scoresPerUser.get(email).get("Fluency").put("INID_" + arr[0].substring("Fluency".length()).trim(), Integer.parseInt(arr[1].trim()));
                    } else if (s.startsWith("Adequacy")) {
                        String[] arr = s.split("\t");
                        if (!scoresPerUser.containsKey(email)) {
                            scoresPerUser.put(email, new HashMap<String, HashMap<String, Integer>>());
                            scoresPerUser.get(email).put("Fluency", new HashMap<String, Integer>());
                            scoresPerUser.get(email).put("Adequacy", new HashMap<String, Integer>());
                        }
                        scoresPerUser.get(email).get("Adequacy").put("INID_" + arr[0].substring("Adequacy".length()).trim(), Integer.parseInt(arr[1].trim()));
                    }
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        ArrayList<String> emails = new ArrayList<String>(scoresPerUser.keySet());
        Random r = new Random();
        System.out.println("********** LUCKY WINNERS ************");
        System.out.println(emails.get(r.nextInt(emails.size())));
        System.out.println(emails.get(r.nextInt(emails.size())));
        System.out.println(emails.get(r.nextInt(emails.size())));
        System.out.println("********** LUCKY WINNERS ************");
        System.out.println(new HashSet(ages.values()));
        System.out.println(new HashSet(languages.values()));
        System.out.println(new HashSet(educationLevel.values()));
        System.out.println(new HashSet(englishLevel.values()));

        HashSet<String> unreliableUserIDs = new HashSet<>();
        HashSet<String> unreliableUserIDsFL = new HashSet<>();
        HashSet<String> unreliableUserIDsAD = new HashSet<>();
        scoresPerUser.keySet().forEach((userID) -> {
            if (scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Worse") <= scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Med")
                    && scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Best")
                    && scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Med") <= scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Best")
                    && scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Worse") <= scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Med")
                    && scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Best")
                    && scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Med") <= scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Best")) {
            } else {
                /*System.out.println(userID);
                System.out.println(scores.get(userID).get("Fluency").get("CONTROL-Worse") + " < " + scores.get(userID).get("Fluency").get("CONTROL-Med") + " <= " + scores.get(userID).get("Fluency").get("CONTROL-Best"));
                System.out.println(scores.get(userID).get("Adequacy").get("CONTROL-Worse") + " < " + scores.get(userID).get("Adequacy").get("CONTROL-Med") + " <= " + scores.get(userID).get("Adequacy").get("CONTROL-Best"));
                System.out.println("");*/
                unreliableUserIDs.add(userID);
                if (scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Worse") <= scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Med")
                        && scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Worse") < scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Best")
                        && scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Med") <= scoresPerUser.get(userID).get("Fluency").get("INID_CONTROL-Best")) {
                } else {
                    unreliableUserIDsFL.add(userID);
                }
                if (scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Worse") <= scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Med")
                        && scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Worse") < scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Best")
                        && scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Med") <= scoresPerUser.get(userID).get("Adequacy").get("INID_CONTROL-Best")) {
                } else {
                    unreliableUserIDsAD.add(userID);
                }
            }
        });

        ArrayList<String> instanceIDs = new ArrayList<>();
        scoresPerUser.keySet().forEach((userID) -> {
            scoresPerUser.get(userID).get("Fluency").keySet().stream().filter((instance) -> (!instance.contains("INID_CONTROL-Worse")
                    && !instance.contains("INID_CONTROL-Med")
                    && !instance.contains("INID_CONTROL-Best"))).filter((instance) -> (!instanceIDs.contains(instance))).forEachOrdered((instance) -> {
                        instanceIDs.add(instance);
            });
        });
        Collections.sort(instanceIDs);

        System.out.println("PARTICIPATING users: " + scoresPerUser.keySet().size());
        System.out.println("UNRELIABLE: " + (unreliableUserIDs.size() / ((double) scoresPerUser.keySet().size())));
        System.out.println("UNRELIABLE on Fluency: " + (unreliableUserIDsFL.size() / ((double) scoresPerUser.keySet().size())));
        System.out.println("UNRELIABLE on Adequacy: " + (unreliableUserIDsAD.size() / ((double) scoresPerUser.keySet().size())));
        System.out.println(unreliableUserIDs);
        unreliableUserIDs.forEach((unreliableUserID) -> {
            scoresPerUser.remove(unreliableUserID);
        });
        System.out.println("REMAINING users: " + scoresPerUser.keySet().size());

        //AGREEMENT VECTORS
        HashSet<String> usedIDs = new HashSet<String>();
        for (String userID : scoresPerUser.keySet()) {
            if (!usedIDs.contains(userID)) {
                usedIDs.add(userID);
                HashSet<String> usersOfSameInstances = new HashSet<String>();
                usersOfSameInstances.add(userID);
                for (String userID2 : scoresPerUser.keySet()) {
                    if (!usedIDs.contains(userID2)) {
                        if (scoresPerUser.get(userID).get("Fluency").keySet().equals(scoresPerUser.get(userID2).get("Fluency").keySet())
                                && scoresPerUser.get(userID).get("Adequacy").keySet().equals(scoresPerUser.get(userID2).get("Adequacy").keySet())) {
                            usedIDs.add(userID2);
                            usersOfSameInstances.add(userID2);
                        }
                    }
                }
                if (usersOfSameInstances.size() > 1) {
                    ArrayList<String> instances = new ArrayList<>(scoresPerUser.get(userID).get("Fluency").keySet());
                    usersOfSameInstances.stream().map((ID) -> {
                        System.out.print("Fluency of " + ID + ":\t");
                        return ID;
                    }).map((ID) -> {
                        instances.forEach((instance) -> {
                            System.out.print(scoresPerUser.get(ID).get("Fluency").get(instance) + "\t");
                        });
                        return ID;
                    }).forEachOrdered((_item) -> {
                        System.out.println();
                    });
                }
            }
        }
        System.out.println("===============");
        usedIDs = new HashSet<String>();
        for (String userID : scoresPerUser.keySet()) {
            if (!usedIDs.contains(userID)) {
                usedIDs.add(userID);
                HashSet<String> usersOfSameInstances = new HashSet<String>();
                usersOfSameInstances.add(userID);
                for (String userID2 : scoresPerUser.keySet()) {
                    if (!usedIDs.contains(userID2)) {
                        if (scoresPerUser.get(userID).get("Fluency").keySet().equals(scoresPerUser.get(userID2).get("Fluency").keySet())
                                && scoresPerUser.get(userID).get("Adequacy").keySet().equals(scoresPerUser.get(userID2).get("Adequacy").keySet())) {
                            usedIDs.add(userID2);
                            usersOfSameInstances.add(userID2);
                        }
                    }
                }
                if (usersOfSameInstances.size() > 1) {
                    ArrayList<String> instances = new ArrayList<>(scoresPerUser.get(userID).get("Adequacy").keySet());
                    usersOfSameInstances.stream().map((ID) -> {
                        System.out.print("Adequacy of " + ID + ":\t");
                        return ID;
                    }).map((ID) -> {
                        instances.forEach((instance) -> {
                            System.out.print(scoresPerUser.get(ID).get("Adequacy").get(instance) + "\t");
                        });
                        return ID;
                    }).forEachOrdered((_item) -> {
                        System.out.println();
                    });
                }
            }
        }

        HashMap<String, ArrayList<Integer>> scoresPerMRIDFluency = new HashMap<>();
        HashMap<String, ArrayList<Integer>> scoresPerMRIDAdequacy = new HashMap<>();
        mrsBAGELSet.keySet().stream().map((mrID) -> {
            scoresPerMRIDFluency.put("LOLS_" + mrID, new ArrayList<Integer>());
            return mrID;
        }).map((mrID) -> {
            scoresPerMRIDFluency.put("OTHER_" + mrID, new ArrayList<Integer>());
            return mrID;
        }).map((mrID) -> {
            scoresPerMRIDAdequacy.put("LOLS_" + mrID, new ArrayList<Integer>());
            return mrID;
        }).forEachOrdered((mrID) -> {
            scoresPerMRIDAdequacy.put("OTHER_" + mrID, new ArrayList<Integer>());
        });
        mrsSFHOTELSet.keySet().stream().map((mrID) -> {
            scoresPerMRIDFluency.put("LOLS_" + mrID, new ArrayList<Integer>());
            return mrID;
        }).map((mrID) -> {
            scoresPerMRIDFluency.put("OTHER_" + mrID, new ArrayList<Integer>());
            return mrID;
        }).map((mrID) -> {
            scoresPerMRIDAdequacy.put("LOLS_" + mrID, new ArrayList<Integer>());
            return mrID;
        }).forEachOrdered((mrID) -> {
            scoresPerMRIDAdequacy.put("OTHER_" + mrID, new ArrayList<Integer>());
        });
        mrsSFRESTSet.keySet().stream().map((mrID) -> {
            scoresPerMRIDFluency.put("LOLS_" + mrID, new ArrayList<Integer>());
            return mrID;
        }).map((mrID) -> {
            scoresPerMRIDFluency.put("OTHER_" + mrID, new ArrayList<Integer>());
            return mrID;
        }).map((mrID) -> {
            scoresPerMRIDAdequacy.put("LOLS_" + mrID, new ArrayList<Integer>());
            return mrID;
        }).forEachOrdered((mrID) -> {
            scoresPerMRIDAdequacy.put("OTHER_" + mrID, new ArrayList<Integer>());
        });

        ArrayList<String> mrSUBSETFOCUS = new ArrayList<>();
        //HOTEL IN TEST BUT NOT IN TRAIN MRS
        /*mrSUBSETFOCUS.add("inform(name='the carriage inn'has_internet='yes',dogs_allowed='no')");
        mrSUBSETFOCUS.add("inform(name='union hotel',dogs_allowed='none',has_internet='yes',near='mission')");
        mrSUBSETFOCUS.add("inform(name='noe 7s nest bed and breakfast',dogs_allowed='none',has_internet='yes',near='mission',accepts_credit_cards='yes')");
        mrSUBSETFOCUS.add("inform(name='mandarin oriental c san francisco',has_internet='yes',dogs_allowed='yes',area='financial district')");
        mrSUBSETFOCUS.add("inform(type='hotel',count='2',near='marina cow hollow',price_range='inexpensive')");
        mrSUBSETFOCUS.add("inform(name='marina inn',accepts_credit_cards='yes',near='marina cow hollow',price_range='inexpensive')");
        mrSUBSETFOCUS.add("inform(type='hotel',count='2',accepts_credit_cards='yes',price_range='pricey',area='nob hill')");
        mrSUBSETFOCUS.add("inform(type='hotel',count='169',has_internet='yes',dogs_allowed=dont_care)");
        mrSUBSETFOCUS.add("inform(name='nob hill motor inn',accepts_credit_cards='yes',dogs_allowed='no',near='nob hill')");
        mrSUBSETFOCUS.add("inform(name='italian american hotel',has_internet='none')");
        mrSUBSETFOCUS.add("inform_only_match(name='red victorian bed breakfast and arts cafe',accepts_credit_cards='yes',near='haight',has_internet='yes')");
        //REST IN TEST BUT NOT IN TRAIN MRS
        mrSUBSETFOCUS.add("inform(type=restaurant,count='4',good_for_meal=dinner,near='potrero hill',price_range=moderate)");
        mrSUBSETFOCUS.add("inform(type=restaurant,count='36',good_for_meal=lunch,food=dont_care)");
        mrSUBSETFOCUS.add("inform(type=restaurant,count='2',good_for_meal=dont_care,price_range=cheap,area='little russia')");
        mrSUBSETFOCUS.add("inform(type=restaurant,count='6',food=korean,kids_allowed=yes,price_range=dont_care)");
        mrSUBSETFOCUS.add("inform(type=restaurant,count='2',kids_allowed=no,area='pacific heights')");
        mrSUBSETFOCUS.add("inform(type=restaurant,count='2',good_for_meal=dont_care,food=pizza,kids_allowed=yes)");
        mrSUBSETFOCUS.add("inform(name=source,good_for_meal=lunch,price_range=cheap,kids_allowed=yes,food=pizza)");
        mrSUBSETFOCUS.add("inform(type=restaurant,count='4',good_for_meal=dont_care,kids_allowed=yes,price_range=expensive)");
        mrSUBSETFOCUS.add("inform(name='fresca',area='hayes valley or pacific heights',price='between 15 and 26 euro',phone='4154472668')");
        mrSUBSETFOCUS.add("inform(type=restaurant,count='2',kids_allowed=dont_care,area='golden gate park')");
        mrSUBSETFOCUS.add("inform(type=restaurant,count='239',kids_allowed=dont_care,good_for_meal=dont_care)");
        mrSUBSETFOCUS.add("inform_no_match(near=chinatown,area='opera plaza')");
        mrSUBSETFOCUS.add("inform_no_match(food=thai,near='inner richmond',kids_allowed=no)");*/

        int counterF = 0;
        int counterA = 0;
        for (String userID : scoresPerUser.keySet()) {
            for (String instance : scoresPerUser.get(userID).get("Fluency").keySet()) {
                if (!instance.equals("INID_CONTROL-Worse")
                        && !instance.equals("INID_CONTROL-Med")
                        && !instance.equals("INID_CONTROL-Best")) {
                    String mrID = "";
                    String mr = "";
                    if (instance.contains("LOLS")) {
                        for (String m : evalInstanceIDToMrID_LOLS.get(instance)) {
                            mr = m;
                        }
                        mrID = "LOLS_" + mr;
                    } else {
                        for (String m : evalInstanceIDToMrID_OTHER.get(instance)) {
                            mr = m;
                        }
                        mrID = "OTHER_" + mr;
                    }
                    if (!mrSUBSETFOCUS.isEmpty()) {
                        boolean isIn = false;
                        if (instance.contains("LOLS")) {
                            for (String m : evalInstanceIDToMrID_LOLS.get(instance)) {
                                if (mrSUBSETFOCUS.contains(mrs.get(m))) {
                                    isIn = true;
                                }
                            }
                        } else {
                            for (String m : evalInstanceIDToMrID_OTHER.get(instance)) {
                                if (mrSUBSETFOCUS.contains(mrs.get(m))) {
                                    isIn = true;
                                }
                            }
                        }
                        if (isIn) {
                            counterF++;
                            scoresPerMRIDFluency.get(mrID).add(scoresPerUser.get(userID).get("Fluency").get(instance));
                        }
                    } else {
                        scoresPerMRIDFluency.get(mrID).add(scoresPerUser.get(userID).get("Fluency").get(instance));
                    }
                }
            }
            for (String instance : scoresPerUser.get(userID).get("Adequacy").keySet()) {
                if (!instance.equals("INID_CONTROL-Worse")
                        && !instance.equals("INID_CONTROL-Med")
                        && !instance.equals("INID_CONTROL-Best")) {
                    String mrID = "";
                    if (instance.contains("LOLS")) {
                        String mr = "";
                        for (String m : evalInstanceIDToMrID_LOLS.get(instance)) {
                            mr = m;
                        }
                        mrID = "LOLS_" + mr;
                    } else {
                        String mr = "";
                        for (String m : evalInstanceIDToMrID_OTHER.get(instance)) {
                            mr = m;
                        }
                        mrID = "OTHER_" + mr;
                    }
                    if (!mrSUBSETFOCUS.isEmpty()) {
                        boolean isIn = false;
                        if (instance.contains("LOLS")) {
                            for (String m : evalInstanceIDToMrID_LOLS.get(instance)) {
                                if (mrSUBSETFOCUS.contains(mrs.get(m))) {
                                    isIn = true;
                                }
                            }
                        } else {
                            for (String m : evalInstanceIDToMrID_OTHER.get(instance)) {
                                if (mrSUBSETFOCUS.contains(mrs.get(m))) {
                                    isIn = true;
                                }
                            }
                        }
                        if (isIn) {
                            counterA++;
                            scoresPerMRIDAdequacy.get(mrID).add(scoresPerUser.get(userID).get("Adequacy").get(instance));
                        }
                    } else {
                        scoresPerMRIDAdequacy.get(mrID).add(scoresPerUser.get(userID).get("Adequacy").get(instance));
                    }
                }
            }
        }

        ArrayList<Double> LOLS_BAGEL_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFHOT_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFRES_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> WEN_SFHOT_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> WEN_SFRES_AVG_fluency_set = new ArrayList<>();
        ArrayList<Double> Dusek_AVG_fluency_set = new ArrayList<>();

        Double LOLS_BAGEL_AVG_fluency = 0.0;
        Double LOLS_SFHOT_AVG_fluency = 0.0;
        Double LOLS_SFRES_AVG_fluency = 0.0;
        Double WEN_SFHOT_AVG_fluency = 0.0;
        Double WEN_SFRES_AVG_fluency = 0.0;
        Double Dusek_AVG_fluency = 0.0;

        Double LOLS_BAGEL_total_fluency = 0.0;
        Double LOLS_SFHOT_total_fluency = 0.0;
        Double LOLS_SFRES_total_fluency = 0.0;
        Double WEN_SFHOT_total_fluency = 0.0;
        Double WEN_SFRES_total_fluency = 0.0;
        Double Dusek_total_fluency = 0.0;

        HashSet<HashSet<String>> mrsUniqueLOLS = new HashSet<>();
        HashSet<HashSet<String>> mrsUniqueOTHER = new HashSet<>();
        for (String mrID : scoresPerMRIDFluency.keySet()) {
            if (mrID.startsWith("LOLS_BAGEL")) {
                //if (!mrsUniqueLOLS.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueLOLS.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDFluency.get(mrID)) {
                    LOLS_BAGEL_AVG_fluency += score.doubleValue();
                    LOLS_BAGEL_total_fluency++;
                    LOLS_BAGEL_AVG_fluency_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("LOLS_SFHOT")) {
                //if (!mrsUniqueLOLS.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueLOLS.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDFluency.get(mrID)) {
                    LOLS_SFHOT_AVG_fluency += score.doubleValue();
                    LOLS_SFHOT_total_fluency++;
                    LOLS_SFHOT_AVG_fluency_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("LOLS_SFRES")) {
                //if (!mrsUniqueLOLS.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueLOLS.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDFluency.get(mrID)) {
                    LOLS_SFRES_AVG_fluency += score.doubleValue();
                    LOLS_SFRES_total_fluency++;
                    LOLS_SFRES_AVG_fluency_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("OTHER_SFHOT")) {
                //if (!mrsUniqueOTHER.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueOTHER.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDFluency.get(mrID)) {
                    WEN_SFHOT_AVG_fluency += score.doubleValue();
                    WEN_SFHOT_total_fluency++;
                    WEN_SFHOT_AVG_fluency_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("OTHER_SFRES")) {
                //if (!mrsUniqueOTHER.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueOTHER.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDFluency.get(mrID)) {
                    WEN_SFRES_AVG_fluency += score.doubleValue();
                    WEN_SFRES_total_fluency++;
                    WEN_SFRES_AVG_fluency_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("OTHER_BAGEL")) {
                //if (!mrsUniqueOTHER.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueOTHER.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDFluency.get(mrID)) {
                    Dusek_AVG_fluency += score.doubleValue();
                    Dusek_total_fluency++;
                    Dusek_AVG_fluency_set.add(score.doubleValue());
                }
                //}
            }
        }
        LOLS_BAGEL_AVG_fluency /= LOLS_BAGEL_total_fluency;
        LOLS_SFHOT_AVG_fluency /= LOLS_SFHOT_total_fluency;
        LOLS_SFRES_AVG_fluency /= LOLS_SFRES_total_fluency;
        WEN_SFHOT_AVG_fluency /= WEN_SFHOT_total_fluency;
        WEN_SFRES_AVG_fluency /= WEN_SFRES_total_fluency;
        Dusek_AVG_fluency /= Dusek_total_fluency;

        ArrayList<Double> LOLS_BAGEL_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFHOT_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> LOLS_SFRES_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> WEN_SFHOT_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> WEN_SFRES_AVG_adequacy_set = new ArrayList<>();
        ArrayList<Double> Dusek_AVG_adequacy_set = new ArrayList<>();

        Double LOLS_BAGEL_AVG_adequacy = 0.0;
        Double LOLS_SFHOT_AVG_adequacy = 0.0;
        Double LOLS_SFRES_AVG_adequacy = 0.0;
        Double WEN_SFHOT_AVG_adequacy = 0.0;
        Double WEN_SFRES_AVG_adequacy = 0.0;
        Double Dusek_AVG_adequacy = 0.0;

        Double LOLS_BAGEL_total_adequacy = 0.0;
        Double LOLS_SFHOT_total_adequacy = 0.0;
        Double LOLS_SFRES_total_adequacy = 0.0;
        Double WEN_SFHOT_total_adequacy = 0.0;
        Double WEN_SFRES_total_adequacy = 0.0;
        Double Dusek_total_adequacy = 0.0;

        mrsUniqueLOLS = new HashSet<>();
        mrsUniqueOTHER = new HashSet<>();
        for (String mrID : scoresPerMRIDAdequacy.keySet()) {
            if (mrID.startsWith("LOLS_BAGEL")) {
                //if (!mrsUniqueLOLS.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueLOLS.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDAdequacy.get(mrID)) {
                    LOLS_BAGEL_AVG_adequacy += score.doubleValue();
                    LOLS_BAGEL_total_adequacy++;
                    LOLS_BAGEL_AVG_adequacy_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("LOLS_SFHOT")) {
                //if (!mrsUniqueLOLS.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueLOLS.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDAdequacy.get(mrID)) {
                    LOLS_SFHOT_AVG_adequacy += score.doubleValue();
                    LOLS_SFHOT_total_adequacy++;
                    LOLS_SFHOT_AVG_adequacy_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("LOLS_SFRES")) {
                //if (!mrsUniqueLOLS.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueLOLS.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDAdequacy.get(mrID)) {
                    LOLS_SFRES_AVG_adequacy += score.doubleValue();
                    LOLS_SFRES_total_adequacy++;
                    LOLS_SFRES_AVG_adequacy_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("OTHER_SFHOT")) {
                //if (!mrsUniqueOTHER.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueOTHER.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDAdequacy.get(mrID)) {
                    WEN_SFHOT_AVG_adequacy += score.doubleValue();
                    WEN_SFHOT_total_adequacy++;
                    WEN_SFHOT_AVG_adequacy_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("OTHER_SFRES")) {
                //if (!mrsUniqueOTHER.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueOTHER.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDAdequacy.get(mrID)) {
                    WEN_SFRES_AVG_adequacy += score.doubleValue();
                    WEN_SFRES_total_adequacy++;
                    WEN_SFRES_AVG_adequacy_set.add(score.doubleValue());
                }
                //}
            } else if (mrID.startsWith("OTHER_BAGEL")) {
                //if (!mrsUniqueOTHER.contains(mrsSet.get(mrID.substring(mrID.indexOf("_") + 1)))) {
                mrsUniqueOTHER.add(mrsSet.get(mrID.substring(mrID.indexOf('_') + 1)));
                for (Integer score : scoresPerMRIDAdequacy.get(mrID)) {
                    Dusek_AVG_adequacy += score.doubleValue();
                    Dusek_total_adequacy++;
                    Dusek_AVG_adequacy_set.add(score.doubleValue());
                }
                //}
            }
        }
        LOLS_BAGEL_AVG_adequacy /= LOLS_BAGEL_total_adequacy;
        LOLS_SFHOT_AVG_adequacy /= LOLS_SFHOT_total_adequacy;
        LOLS_SFRES_AVG_adequacy /= LOLS_SFRES_total_adequacy;
        WEN_SFHOT_AVG_adequacy /= WEN_SFHOT_total_adequacy;
        WEN_SFRES_AVG_adequacy /= WEN_SFRES_total_adequacy;
        Dusek_AVG_adequacy /= Dusek_total_adequacy;

        System.out.println();
        System.out.println(mrsUniqueLOLS.size() + " = " + mrsUniqueOTHER.size() + " " + mrsUniqueLOLS.equals(mrsUniqueOTHER));
        System.out.println("BAGEL\tLOLS\tDusek");
        System.out.println("Fluency\t" + LOLS_BAGEL_AVG_fluency + "(" + getConf(LOLS_BAGEL_AVG_fluency_set, LOLS_BAGEL_AVG_fluency, LOLS_BAGEL_total_fluency) + ")" + "\t" + Dusek_AVG_fluency + "(" + getConf(Dusek_AVG_fluency_set, Dusek_AVG_fluency, Dusek_total_fluency) + ")");
        System.out.println("Adequacy\t" + LOLS_BAGEL_AVG_adequacy + "(" + getConf(LOLS_BAGEL_AVG_adequacy_set, LOLS_BAGEL_AVG_adequacy, LOLS_BAGEL_total_adequacy) + ")" + "\t" + Dusek_AVG_adequacy + "(" + getConf(Dusek_AVG_adequacy_set, Dusek_AVG_adequacy, Dusek_total_adequacy) + ")");
        System.out.println();
        System.out.println("SF HOTEL\tLOLS\tWen");
        System.out.println("Fluency\t" + LOLS_SFHOT_AVG_fluency + "(" + getConf(LOLS_SFHOT_AVG_fluency_set, LOLS_SFHOT_AVG_fluency, LOLS_SFHOT_total_fluency) + ")" + "\t" + WEN_SFHOT_AVG_fluency + "(" + getConf(WEN_SFHOT_AVG_fluency_set, WEN_SFHOT_AVG_fluency, WEN_SFHOT_total_fluency) + ")");
        System.out.println("Adequacy\t" + LOLS_SFHOT_AVG_adequacy + "(" + getConf(LOLS_SFHOT_AVG_adequacy_set, LOLS_SFHOT_AVG_adequacy, LOLS_SFHOT_total_adequacy) + ")" + "\t" + WEN_SFHOT_AVG_adequacy + "(" + getConf(WEN_SFHOT_AVG_adequacy_set, WEN_SFHOT_AVG_adequacy, WEN_SFHOT_total_adequacy) + ")");
        System.out.println();
        System.out.println("SF REST\tLOLS\tWen");
        System.out.println("Fluency\t" + LOLS_SFRES_AVG_fluency + "(" + getConf(LOLS_SFRES_AVG_fluency_set, LOLS_SFRES_AVG_fluency, LOLS_SFRES_total_fluency) + ")" + "\t" + WEN_SFRES_AVG_fluency + "(" + getConf(WEN_SFRES_AVG_fluency_set, WEN_SFRES_AVG_fluency, WEN_SFRES_total_fluency) + ")");
        System.out.println("Adequacy\t" + LOLS_SFRES_AVG_adequacy + "(" + getConf(LOLS_SFRES_AVG_adequacy_set, LOLS_SFRES_AVG_adequacy, LOLS_SFRES_total_adequacy) + ")" + "\t" + WEN_SFRES_AVG_adequacy + "(" + getConf(WEN_SFRES_AVG_adequacy_set, WEN_SFRES_AVG_adequacy, WEN_SFRES_total_adequacy) + ")");
        System.out.println(counterF + " " + counterA);
        System.out.println();

        HashSet<String> preds = new HashSet<>();
        scoresPerMRIDFluency.keySet().forEach((mr) -> {
            if (mr.startsWith("LOLS_SFHOT")) {
                String mrSub = mr.substring(mr.indexOf('_') + 1);
                preds.add(mrsSFHOTEL.get(mrSub).substring(0, mrsSFHOTEL.get(mrSub).indexOf('(')));
            } else if (mr.startsWith("LOLS_SFRES")) {
                String mrSub = mr.substring(mr.indexOf('_') + 1);
                preds.add(mrsSFREST.get(mrSub).substring(0, mrsSFREST.get(mrSub).indexOf('(')));
            }
        });
        preds.stream().map((predicate) -> {
            ArrayList<Double> LOLS_SFHOT_AVG_fluency_set_perPred = new ArrayList<>();
            ArrayList<Double> LOLS_SFRES_AVG_fluency_set_perPred = new ArrayList<>();
            ArrayList<Double> WEN_SFHOT_AVG_fluency_set_perPred = new ArrayList<>();
            ArrayList<Double> WEN_SFRES_AVG_fluency_set_perPred = new ArrayList<>();
            Double LOLS_SFHOT_AVG_fluency_perPred = 0.0;
            Double LOLS_SFRES_AVG_fluency_perPred = 0.0;
            Double WEN_SFHOT_AVG_fluency_perPred = 0.0;
            Double WEN_SFRES_AVG_fluency_perPred = 0.0;
            Double LOLS_SFHOT_total_fluency_perPred = 0.0;
            Double LOLS_SFRES_total_fluency_perPred = 0.0;
            Double WEN_SFHOT_total_fluency_perPred = 0.0;
            Double WEN_SFRES_total_fluency_perPred = 0.0;
            for (String instance : scoresPerMRIDFluency.keySet()) {
                if (instance.startsWith("LOLS_SFHOT")) {
                    //if (evalInstanceIDToMrID_LOLS.get(instance) != null) {
                    String mr = instance.substring(instance.indexOf('_') + 1);
                    if (mrsSFHOTEL.get(mr).substring(0, mrsSFHOTEL.get(mr).indexOf('(')).equals(predicate)) {
                        for (Integer score : scoresPerMRIDFluency.get(instance)) {
                            LOLS_SFHOT_AVG_fluency_perPred += score.doubleValue();
                            LOLS_SFHOT_total_fluency_perPred++;
                            LOLS_SFHOT_AVG_fluency_set_perPred.add(score.doubleValue());
                        }
                    }
                    //}
                } else if (instance.startsWith("LOLS_SFRES")) {
                    //if (evalInstanceIDToMrID_LOLS.get(instance) != null) {
                    String mr = instance.substring(instance.indexOf('_') + 1);
                    if (mrsSFREST.get(mr).substring(0, mrsSFREST.get(mr).indexOf('(')).equals(predicate)) {
                        for (Integer score : scoresPerMRIDFluency.get(instance)) {
                            LOLS_SFRES_AVG_fluency_perPred += score.doubleValue();
                            LOLS_SFRES_total_fluency_perPred++;
                            LOLS_SFRES_AVG_fluency_set_perPred.add(score.doubleValue());
                        }
                    }
                    //}
                } else if (instance.startsWith("OTHER_SFHOT")) {
                    // if (evalInstanceIDToMrID_OTHER.get(instance) != null) {
                    String mr = instance.substring(instance.indexOf('_') + 1);
                    if (mrsSFHOTEL.get(mr).substring(0, mrsSFHOTEL.get(mr).indexOf('(')).equals(predicate)) {
                        for (Integer score : scoresPerMRIDFluency.get(instance)) {
                            WEN_SFHOT_AVG_fluency_perPred += score.doubleValue();
                            WEN_SFHOT_total_fluency_perPred++;
                            WEN_SFHOT_AVG_fluency_set_perPred.add(score.doubleValue());
                        }
                    }
                    //}
                } else if (instance.startsWith("OTHER_SFRES")) {
                    //if (evalInstanceIDToMrID_OTHER.get(instance) != null) {
                    String mr = instance.substring(instance.indexOf('_') + 1);
                    if (mrsSFREST.get(mr).substring(0, mrsSFREST.get(mr).indexOf('(')).equals(predicate)) {
                        for (Integer score : scoresPerMRIDFluency.get(instance)) {
                            WEN_SFRES_AVG_fluency_perPred += score.doubleValue();
                            WEN_SFRES_total_fluency_perPred++;
                            WEN_SFRES_AVG_fluency_set_perPred.add(score.doubleValue());
                        }
                    }
                    //}
                }
            }
            LOLS_SFHOT_AVG_fluency_perPred /= LOLS_SFHOT_total_fluency_perPred;
            LOLS_SFRES_AVG_fluency_perPred /= LOLS_SFRES_total_fluency_perPred;
            WEN_SFHOT_AVG_fluency_perPred /= WEN_SFHOT_total_fluency_perPred;
            WEN_SFRES_AVG_fluency_perPred /= WEN_SFRES_total_fluency_perPred;
            ArrayList<Double> LOLS_SFHOT_AVG_adequacy_set_perPred = new ArrayList<>();
            ArrayList<Double> LOLS_SFRES_AVG_adequacy_set_perPred = new ArrayList<>();
            ArrayList<Double> WEN_SFHOT_AVG_adequacy_set_perPred = new ArrayList<>();
            ArrayList<Double> WEN_SFRES_AVG_adequacy_set_perPred = new ArrayList<>();
            Double LOLS_SFHOT_AVG_adequacy_perPred = 0.0;
            Double LOLS_SFRES_AVG_adequacy_perPred = 0.0;
            Double WEN_SFHOT_AVG_adequacy_perPred = 0.0;
            Double WEN_SFRES_AVG_adequacy_perPred = 0.0;
            Double LOLS_SFHOT_total_adequacy_perPred = 0.0;
            Double LOLS_SFRES_total_adequacy_perPred = 0.0;
            Double WEN_SFHOT_total_adequacy_perPred = 0.0;
            Double WEN_SFRES_total_adequacy_perPred = 0.0;
            for (String instance : scoresPerMRIDAdequacy.keySet()) {
                if (instance.startsWith("LOLS_SFHOT")) {
                    //if (evalInstanceIDToMrID_LOLS.get(instance) != null) {
                    String mr = instance.substring(instance.indexOf('_') + 1);
                    if (mrsSFHOTEL.get(mr).substring(0, mrsSFHOTEL.get(mr).indexOf('(')).equals(predicate)) {
                        for (Integer score : scoresPerMRIDAdequacy.get(instance)) {
                            LOLS_SFHOT_AVG_adequacy_perPred += score.doubleValue();
                            LOLS_SFHOT_total_adequacy_perPred++;
                            LOLS_SFHOT_AVG_adequacy_set_perPred.add(score.doubleValue());
                        }
                    }
                    //}
                } else if (instance.startsWith("LOLS_SFRES")) {
                    //if (evalInstanceIDToMrID_LOLS.get(instance) != null) {
                    String mr = instance.substring(instance.indexOf('_') + 1);
                    if (mrsSFREST.get(mr).substring(0, mrsSFREST.get(mr).indexOf('(')).equals(predicate)) {
                        for (Integer score : scoresPerMRIDAdequacy.get(instance)) {
                            LOLS_SFRES_AVG_adequacy_perPred += score.doubleValue();
                            LOLS_SFRES_total_adequacy_perPred++;
                            LOLS_SFRES_AVG_adequacy_set_perPred.add(score.doubleValue());
                        }
                    }
                    //}
                } else if (instance.startsWith("OTHER_SFHOT")) {
                    //if (evalInstanceIDToMrID_OTHER.get(instance) != null) {
                    String mr = instance.substring(instance.indexOf('_') + 1);
                    if (mrsSFHOTEL.get(mr).substring(0, mrsSFHOTEL.get(mr).indexOf('(')).equals(predicate)) {
                        for (Integer score : scoresPerMRIDAdequacy.get(instance)) {
                            WEN_SFHOT_AVG_adequacy_perPred += score.doubleValue();
                            WEN_SFHOT_total_adequacy_perPred++;
                            WEN_SFHOT_AVG_adequacy_set_perPred.add(score.doubleValue());
                        }
                    }
                    //}
                } else if (instance.startsWith("OTHER_SFRES")) {
                    //if (evalInstanceIDToMrID_OTHER.get(instance) != null) {
                    String mr = instance.substring(instance.indexOf('_') + 1);
                    if (mrsSFREST.get(mr).substring(0, mrsSFREST.get(mr).indexOf('(')).equals(predicate)) {
                        for (Integer score : scoresPerMRIDAdequacy.get(instance)) {
                            WEN_SFRES_AVG_adequacy_perPred += score.doubleValue();
                            WEN_SFRES_total_adequacy_perPred++;
                            WEN_SFRES_AVG_adequacy_set_perPred.add(score.doubleValue());
                        }
                    }
                    //}
                }
            }
            LOLS_SFHOT_AVG_adequacy_perPred /= LOLS_SFHOT_total_adequacy_perPred;
            LOLS_SFRES_AVG_adequacy_perPred /= LOLS_SFRES_total_adequacy_perPred;
            WEN_SFHOT_AVG_adequacy_perPred /= WEN_SFHOT_total_adequacy_perPred;
            WEN_SFRES_AVG_adequacy_perPred /= WEN_SFRES_total_adequacy_perPred;
            System.out.println();
            System.out.println("PREDICATE: " + predicate);
            System.out.println("SF HOTEL\tLOLS\tWen");
            System.out.println("Fluency\t" + LOLS_SFHOT_AVG_fluency_perPred + "(" + getConf(LOLS_SFHOT_AVG_fluency_set_perPred, LOLS_SFHOT_AVG_fluency_perPred, LOLS_SFHOT_total_fluency_perPred) + ")" + "\t" + WEN_SFHOT_AVG_fluency_perPred + "(" + getConf(WEN_SFHOT_AVG_fluency_set_perPred, WEN_SFHOT_AVG_fluency_perPred, WEN_SFHOT_total_fluency_perPred) + ")");
            System.out.println("Adequacy\t" + LOLS_SFHOT_AVG_adequacy_perPred + "(" + getConf(LOLS_SFHOT_AVG_adequacy_set_perPred, LOLS_SFHOT_AVG_adequacy_perPred, LOLS_SFHOT_total_adequacy_perPred) + ")" + "\t" + WEN_SFHOT_AVG_adequacy_perPred + "(" + getConf(WEN_SFHOT_AVG_adequacy_set_perPred, WEN_SFHOT_AVG_adequacy_perPred, WEN_SFHOT_total_adequacy_perPred) + ")");
            System.out.println();
            System.out.println("SF REST\tLOLS\tWen");
            System.out.println("Fluency\t" + LOLS_SFRES_AVG_fluency_perPred + "(" + getConf(LOLS_SFRES_AVG_fluency_set_perPred, LOLS_SFRES_AVG_fluency_perPred, LOLS_SFRES_total_fluency_perPred) + ")" + "\t" + WEN_SFRES_AVG_fluency_perPred + "(" + getConf(WEN_SFRES_AVG_fluency_set_perPred, WEN_SFRES_AVG_fluency_perPred, WEN_SFRES_total_fluency_perPred) + ")");
            System.out.println("Adequacy\t" + LOLS_SFRES_AVG_adequacy_perPred + "(" + getConf(LOLS_SFRES_AVG_adequacy_set_perPred, LOLS_SFRES_AVG_adequacy_perPred, LOLS_SFRES_total_adequacy_perPred) + ")" + "\t" + WEN_SFRES_AVG_adequacy_perPred + "(" + getConf(WEN_SFRES_AVG_adequacy_set_perPred, WEN_SFRES_AVG_adequacy_perPred, WEN_SFRES_total_adequacy_perPred) + ")");
            return predicate;
        }).forEachOrdered((_item) -> {
            System.out.println();
        });

        HashSet<String> uniqueMRs = new HashSet<>();
        int maxSize = 0;
        for (String mrID : scoresPerMRIDFluency.keySet()) {
            String id = mrID.substring(mrID.indexOf('_') + 1);
            String instanceID = "";
            for (String iID : instanceIDs) {
                if (mrID.contains("LOLS")
                        && iID.contains("LOLS")) {
                    for (String m : evalInstanceIDToMrID_LOLS.get(iID)) {
                        if (m.equals(id)) {
                            instanceID = iID;
                        }
                    }
                } else if (!mrID.contains("LOLS")
                        && !iID.contains("LOLS")) {
                    for (String m : evalInstanceIDToMrID_OTHER.get(iID)) {
                        if (m.equals(id)) {
                            instanceID = iID;
                        }
                    }
                }
            }

            String MR = mrs.get(id);
            String text = textsPerEvalInstanceID.get(instanceID);
            if (!uniqueMRs.contains(MR)) {
                //uniqueMRs.add(MR + "|" + text);
                //System.out.print(instanceID + "\t");
                //System.out.print(text + "\t");

                ArrayList<Integer> scores_F = new ArrayList<>();
                scores_F.addAll(scoresPerMRIDFluency.get(mrID));
                ArrayList<Integer> scores_A = new ArrayList<>();
                scores_A.addAll(scoresPerMRIDAdequacy.get(mrID));    
                if (!scores_F.isEmpty()
                        || !scores_A.isEmpty()) {
                    System.out.print(mrID + "\t");
                    System.out.print(MR + "\t");       
                    System.out.print(text + "\t");
                    for (int i = 0; i < 26; i++) {
                        if (i < scores_F.size()) {
                        } else {
                        }
                    }
                    for (int i = 0; i < 26; i++) {
                        if (i < scores_A.size()) {
                        } else {
                        }
                    }
                }
                if (scores_F.size() > maxSize) {
                    maxSize = scores_F.size();
                }
                if (scores_A.size() > maxSize) {
                    maxSize = scores_A.size();
                }
            }
        }
        /*HashSet<String> uniqueMRs = new HashSet<>();
        int maxSize = 0;
        for (String instanceID : instanceIDs) {
            String id = "";
            String mrID = "";
            if (instanceID.contains("LOLS")) {
                for (String m : evalInstanceIDToMrID_LOLS.get(instanceID)) {
                    mrID = m;
                }
                id = "LOLS_" + mrID;
            } else {
                for (String m : evalInstanceIDToMrID_OTHER.get(instanceID)) {
                    mrID = m;
                }
                id = "OTHER_" + mrID;
            }

            String MR = mrs.get(mrID);
            String text = textsPerEvalInstanceID.get(instanceID);
            if (!uniqueMRs.contains(MR + "|" + text)) {
                uniqueMRs.add(MR + "|" + text);
                System.out.print(instanceID + "\t");
                System.out.print(abstractMRs.get(MR) + "\t");
                System.out.print(text + "\t");

                ArrayList<Integer> scores_F = new ArrayList<>();
                for (String userID : scoresPerUser.keySet()) {
                    for (String iID : scoresPerUser.get(userID).get("Fluency").keySet()) {
                        if (!iID.equals("INID_CONTROL-Worse")
                                && !iID.equals("INID_CONTROL-Med")
                                && !iID.equals("INID_CONTROL-Best")) {
                            String mrID2 = "";
                            if (iID.contains("LOLS")) {
                                for (String m : evalInstanceIDToMrID_LOLS.get(iID)) {
                                    mrID2 = m;
                                }

                                String MR2 = mrs.get(mrID2);
                                String text2 = textsPerEvalInstanceID.get(iID);
                                if (MR2.equals(MR)
                                        && text.equals(text2)) {
                                    scores_F.add(scoresPerUser.get(userID).get("Fluency").get(iID));
                                }
                            } else {
                                for (String m : evalInstanceIDToMrID_OTHER.get(iID)) {
                                    mrID2 = m;
                                }
                                String MR2 = mrs.get(mrID2);
                                String text2 = textsPerEvalInstanceID.get(iID);
                                if (MR2.equals(MR)
                                        && text.equals(text2)) {
                                    scores_F.add(scoresPerUser.get(userID).get("Fluency").get(iID));
                                }
                            }
                        }
                    }
                }
                ArrayList<Integer> scores_A = new ArrayList<>();
                for (String userID : scoresPerUser.keySet()) {
                    for (String iID : scoresPerUser.get(userID).get("Adequacy").keySet()) {
                        if (!iID.equals("INID_CONTROL-Worse")
                                && !iID.equals("INID_CONTROL-Med")
                                && !iID.equals("INID_CONTROL-Best")) {
                            String mrID2 = "";
                            if (iID.contains("LOLS")) {
                                for (String m : evalInstanceIDToMrID_LOLS.get(iID)) {
                                    mrID2 = m;
                                }

                                String MR2 = mrs.get(mrID2);
                                String text2 = textsPerEvalInstanceID.get(iID);
                                if (abstractMRs.get(MR2).equals(abstractMRs.get(MR))
                                        && text.equals(text2)) {
                                    scores_A.add(scoresPerUser.get(userID).get("Adequacy").get(iID));
                                }
                            } else {
                                for (String m : evalInstanceIDToMrID_OTHER.get(iID)) {
                                    mrID2 = m;
                                }
                                String MR2 = mrs.get(mrID2);
                                String text2 = textsPerEvalInstanceID.get(iID);
                                if (abstractMRs.get(MR2).equals(abstractMRs.get(MR))
                                        && text.equals(text2)) {
                                    scores_A.add(scoresPerUser.get(userID).get("Adequacy").get(iID));
                                }
                            }
                        }
                    }
                }
                for (int i = 0; i < 16; i++) {
                    if (i < scores_F.size()) {
                        System.out.print(scores_F.get(i) + "\t");
                    } else {
                        System.out.print("\t");
                    }
                }
                for (int i = 0; i < 16; i++) {
                    if (i < scores_A.size()) {
                        System.out.print(scores_A.get(i) + "\t");
                    } else {
                        System.out.print("\t");
                    }
                }
                
                System.out.println();
                if (scores_F.size() > maxSize) {
                    maxSize = scores_F.size();
                }
                if (scores_A.size() > maxSize) {
                    maxSize = scores_A.size();
                }
            }
        }
        System.out.println(maxSize);*/
        /*for (String instance : perIdFluencyLOLS.keySet()) {
        if (perIdFluencyOTHER.containsKey(instance)) {
        double avgLOLS = 0.0;
        for (Integer f : perIdFluencyLOLS.get(instance)) {
        avgLOLS += f;
        }
        avgLOLS /= (double) perIdFluencyLOLS.get(instance).size();
        double avgOTHER = 0.0;
        for (Integer f : perIdFluencyOTHER.get(instance)) {
        avgOTHER += f;
        }
        avgOTHER /= (double) perIdFluencyOTHER.get(instance).size();
        
        if (avgLOLS < avgOTHER) {
        System.out.println(instance + "\t" + mrs.get(instance) + " " + avgLOLS + " < " + avgOTHER);
        }
        }
        }*/
        HashMap<String, ArrayList<Integer>> perSystemFluency = new HashMap<>();
        scoresPerMRIDFluency.keySet().forEach((instance) -> {
            String system = instance.substring(0, instance.indexOf('-'));
            if (!perSystemFluency.containsKey(system)) {
                perSystemFluency.put(system, new ArrayList<Integer>());
            }   perSystemFluency.get(system).addAll(scoresPerMRIDFluency.get(instance));
        });
        HashMap<String, ArrayList<Integer>> perSystemAdequacy = new HashMap<>();
        scoresPerMRIDAdequacy.keySet().forEach((instance) -> {
            String system = instance.substring(0, instance.indexOf('-'));
            if (!perSystemAdequacy.containsKey(system)) {
                perSystemAdequacy.put(system, new ArrayList<Integer>());
            }   perSystemAdequacy.get(system).addAll(scoresPerMRIDAdequacy.get(instance));
        });
        perSystemFluency.keySet().stream().map((system) -> {
            System.out.print("FLUENCY " + system + "\t");
            return system;
        }).map((system) -> {
            perSystemFluency.get(system).forEach((i) -> {
                System.out.print(i + "\t");
            });
            return system;
        }).forEachOrdered((_item) -> {
            System.out.println();
        });
        perSystemAdequacy.keySet().stream().map((system) -> {
            System.out.print("ADEQUACY " + system + "\t");
            return system;
        }).map((system) -> {
            perSystemAdequacy.get(system).forEach((i) -> {
                System.out.print(i + "\t");
            });
            return system;
        }).forEachOrdered((_item) -> {
            System.out.println();
        });
    }

    /**
     *
     * @param id
     * @param set
     * @return
     */
    public static boolean containsID(String id, HashSet<String> set) {
        if (set.stream().anyMatch((s) -> (s.contains(id)))) {
            return true;
        }
        return false;
    }

    /**
     *
     * @param values
     * @param avg
     * @param total
     * @return
     */
    public static Double getConf(ArrayList<Double> values, Double avg, Double total) {
        Double variance = 0.0;
        variance = values.stream().map((v) -> Math.pow(avg - v, 2)).reduce(variance, (accumulator, _item) -> accumulator + _item);
        variance /= total;
        if (variance == 0.0) {
            return 0.0;
        }
        Double stDev = Math.sqrt(variance);
        Double alpha = 1.96;

        Double conf = alpha * (stDev / (Math.sqrt(total)));
        return conf;
    }

    /**
     *
     */
    public static void parseWenFiles() {
        String mrFile = "results\\wenResults\\hotelMRs.txt";
        String rFile = "results\\wenResults\\sfxhotel.log";
        HashMap<String, ArrayList<Sequence<IString>>> staticReferences = new HashMap<>();
        HashMap<String, ArrayList<String>> staticReferencesStrings = new HashMap<>();
        HashMap<String, Integer> staticAbstractMRCounts = new HashMap<>();
        //PRINT RESULTS
        ArrayList<String> mrs = new ArrayList<String>();
        try (BufferedReader br = new BufferedReader(new FileReader(mrFile))) {
            String s;
            while ((s = br.readLine()) != null) {
                if (!s.trim().isEmpty()) {
                    mrs.add(s.trim());
                }
            }

        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
        HashMap<String, String> daToGen = new HashMap<>();
        HashMap<String, HashSet<String>> predictedWordSequences_overAllPredicates = new HashMap<>();
        HashMap<String, HashMap<String, HashSet<String>>> predictedWordSequences_perPredicates = new HashMap<>();
        HashMap<String, ArrayList<String>> finalReferencesWordSequences = new HashMap<>();
        HashMap<String, Double> attrCoverage = new HashMap<>();

        HashMap<String, String> abstractMRtoMR = new HashMap<>();
        HashMap<String, String> abstractMRtoText = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(rFile))) {
            String s;
            boolean inGen = false;
            boolean inRef = false;
            ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
            ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
            ArrayList<Sequence<IString>> references = new ArrayList<>();
            ArrayList<String> referencesStrings = new ArrayList<>();
            String da = "";
            String predicate = "";
            String abstractMR = "";
            String predictedWordSequence = "";
            int gens = 0;
            int MRCount = 0;
            HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
            while ((s = br.readLine()) != null) {
                if (s.startsWith("DA")) {
                    inGen = false;
                    inRef = false;
                    da = s.substring(s.indexOf(':') + 1).trim();

                    String MRstr = s.substring(s.indexOf(':') + 1).replaceAll(",", ";").replaceAll("no or yes", "yes or no").replaceAll("ave ; presidio", "ave and presidio").replaceAll("point ; ste", "point and ste").trim();
                    predicate = MRstr.substring(0, MRstr.indexOf('('));
                    abstractMR = predicate + ":";
                    String attributesStr = MRstr.substring(MRstr.indexOf('(') + 1, MRstr.length() - 1);
                    attributeValues = new HashMap<>();
                    if (!attributesStr.isEmpty()) {
                        HashMap<String, Integer> attrXIndeces = new HashMap<>();

                        String[] args = attributesStr.split(";");
                        if (attributesStr.contains("|")) {
                            System.exit(0);
                        }
                        for (String arg : args) {
                            String attr = "";
                            String value = "";
                            if (arg.contains("=")) {
                                String[] subAttr = arg.split("=");
                                value = subAttr[1].toLowerCase();
                                attr = subAttr[0].toLowerCase().replaceAll("_", "");

                                if (value.startsWith("\'")) {
                                    value = value.substring(1, value.length() - 1);
                                }
                                if (value.equals("no")
                                        || value.equals("yes")
                                        || value.equals("yes or no")
                                        || value.equals("none")
                                        || value.equals("empty")) {
                                    attr += "_" + value.replaceAll(" ", "_");
                                    value = attr;
                                }
                                if (value.equals("dont_care")) {
                                    String v = value;
                                    value = attr;
                                    attr = v;
                                }
                            } else {
                                attr = arg.replaceAll("_", "");
                            }
                            if (!attributeValues.containsKey(attr)) {
                                attributeValues.put(attr, new HashSet<String>());
                            }
                            if (value.isEmpty()) {
                                value = attr;
                            }

                            if (value.startsWith("\'")) {
                                value = value.substring(1, value.length() - 1);
                            }
                            if (value.toLowerCase().startsWith("x")) {
                                int index = 0;
                                if (!attrXIndeces.containsKey(attr)) {
                                    attrXIndeces.put(attr, 1);
                                } else {
                                    index = attrXIndeces.get(attr);
                                    attrXIndeces.put(attr, index + 1);
                                }
                                value = "x" + index;
                            }
                            if (value.isEmpty()) {
                                System.exit(0);
                            }

                            attributeValues.get(attr).add(value.trim().toLowerCase());
                        }
                        for (String attr : attributeValues.keySet()) {
                            if (attributeValues.get(attr).contains("yes")
                                    && attributeValues.get(attr).contains("no")) {
                                System.out.println(MRstr);
                                System.exit(0);
                            }
                        }
                        //System.out.println("\t" + da);
                        //System.out.println("\t" + attributeValues);

                        //System.out.println(MRstr);
                        //System.out.println(predicate);
                        //System.out.println(abstractMR);
                    }
                } else if (s.startsWith("Gen")) {
                    gens = 0;
                    inGen = true;
                } else if (s.startsWith("Ref")) {
                    inRef = true;
                    references = new ArrayList<>();
                } else if (inGen) {
                    if (s.trim().isEmpty()) {
                        inGen = false;
                    } else {
                        predictedWordSequence = s.trim().toLowerCase();
                        String ref = predictedWordSequence;

                        HashMap<String, HashMap<String, Integer>> attrValuePriorities = new HashMap<>();
                        HashMap<String, HashSet<String>> delexAttributeValues = new HashMap<>();
                        int prio = 0;
                        for (String attr : attributeValues.keySet()) {
                            if (!attr.isEmpty()) {
                                delexAttributeValues.put(attr, new HashSet<String>());
                                if (attr.equals("name")
                                        || attr.equals("type")
                                        || attr.equals("pricerange")
                                        || attr.equals("price")
                                        || attr.equals("phone")
                                        || attr.equals("address")
                                        || attr.equals("postcode")
                                        || attr.equals("area")
                                        || attr.equals("near")
                                        || attr.equals("food")
                                        || attr.equals("count")
                                        || attr.equals("dont_care")
                                        || attr.equals("goodformeal")) {
                                    attrValuePriorities.put(attr, new HashMap<String, Integer>());
                                    for (String value : attributeValues.get(attr)) {
                                        if (!value.equals("dont_care")
                                                && !value.equals("none")
                                                && !value.equals("empty")
                                                && !value.equals(attr)) {
                                            attrValuePriorities.get(attr).put(value, prio);
                                            prio++;
                                        } else {
                                            delexAttributeValues.get(attr).add(value);
                                        }
                                    }
                                } else {
                                    for (String value : attributeValues.get(attr)) {
                                        delexAttributeValues.get(attr).add(value);
                                    }
                                }
                            }
                        }
                        boolean change = true;
                        while (change) {
                            change = false;
                            for (String attr1 : attrValuePriorities.keySet()) {
                                for (String value1 : attrValuePriorities.get(attr1).keySet()) {
                                    for (String attr2 : attrValuePriorities.keySet()) {
                                        for (String value2 : attrValuePriorities.get(attr2).keySet()) {
                                            if (!value1.equals(value2)
                                                    && value1.contains(value2)
                                                    && attrValuePriorities.get(attr1).get(value1) > attrValuePriorities.get(attr2).get(value2)) {
                                                int prio1 = attrValuePriorities.get(attr1).get(value1);
                                                int prio2 = attrValuePriorities.get(attr2).get(value2);
                                                attrValuePriorities.get(attr1).put(value1, prio2);
                                                attrValuePriorities.get(attr2).put(value2, prio1);
                                                change = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        HashMap<String, Integer> xCounts = new HashMap<>();
                        HashMap<String, String> delexMap = new HashMap<>();
                        ref = " " + ref + " ";
                        for (int p = 0; p < prio; p++) {
                            for (String attr : attrValuePriorities.keySet()) {
                                if (!xCounts.containsKey(attr)) {
                                    xCounts.put(attr, 0);
                                }
                                for (String value : attrValuePriorities.get(attr).keySet()) {
                                    if (attrValuePriorities.get(attr).get(value) == p) {
                                        if (!ref.contains(" " + value + " ")
                                                && !value.contains(" and ")
                                                && !value.contains(" or ")) {
                                            delexAttributeValues.get(attr).add(value);
                                            /*System.out.println(ref);
                                            System.out.println(attr);
                                            System.out.println(value);
                                            System.out.println(attrValuePriorities);*/
                                        } else if (!ref.contains(" " + value + " ")
                                                && (value.contains(" and ")
                                                || value.contains(" or "))) {
                                            String[] values = null;
                                            if (value.contains(" and ")) {
                                                values = value.split(" and ");
                                            } else if (value.contains(" or ")) {
                                                values = value.split(" or ");
                                            }
                                            for (String value1 : values) {
                                                if (!ref.contains(" " + value1 + " ")) {
                                                    /*System.out.println(ref);
                                                    System.out.println(attr);
                                                    System.out.println(value);
                                                    System.out.println(values[v]);
                                                    System.out.println(attrValuePriorities);*/
                                                } else {
                                                    ref = ref.replace(" " + value1 + " ", " " + Action.TOKEN_X + attr + "_" + xCounts.get(attr) + " ");
                                                    ref = ref.replaceAll("  ", " ");
                                                    delexAttributeValues.get(attr).add(Action.TOKEN_X + attr + "_" + xCounts.get(attr));
                                                    delexMap.put(Action.TOKEN_X + attr + "_" + xCounts.get(attr), value1);
                                                    xCounts.put(attr, xCounts.get(attr) + 1);
                                                }
                                            }
                                        } else {
                                            ref = ref.replace(" " + value + " ", " " + Action.TOKEN_X + attr + "_" + xCounts.get(attr) + " ");
                                            ref = ref.replaceAll("  ", " ");
                                            delexAttributeValues.get(attr).add(Action.TOKEN_X + attr + "_" + xCounts.get(attr));
                                            delexMap.put(Action.TOKEN_X + attr + "_" + xCounts.get(attr), value);
                                            xCounts.put(attr, xCounts.get(attr) + 1);
                                        }
                                    }
                                }
                            }
                        }
                        ref = ref.trim();

                        ArrayList<String> attrs = new ArrayList<>(delexAttributeValues.keySet());
                        Collections.sort(attrs);
                        for (String attr : attrs) {
                            abstractMR += attr + "={";

                            ArrayList<String> values = new ArrayList<>(delexAttributeValues.get(attr));
                            Collections.sort(values);
                            abstractMR = values.stream().map((value) -> value + ",").reduce(abstractMR, String::concat);
                            abstractMR += "}";
                        }
                        if (abstractMR.equals("inform:address={@x@address_0,}name={@x@name_0,}phone={@x@phone_0,}pricerange={moderate,}")) {
                            abstractMR = "inform:address={@x@address_0,}name={@x@name_0,}phone={@x@phone_0,}pricerange={@x@pricerange_0,}";
                        }
                        if (abstractMR.equals("inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={moderate,}")) {
                            abstractMR = "inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={@x@pricerange_0,}";
                        }
                        if (abstractMR.equals("inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={pricey,}")) {
                            abstractMR = "inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={@x@pricerange_0,}";
                        }
                        if (abstractMR.equals("?confirm:pricerange={cheap,}")) {
                            abstractMR = "?confirm:pricerange={@x@pricerange_0,}";
                        }
                        if (predictedWordSequence.contains(" hotels ")) {
                            abstractMR = abstractMR.replaceAll("hotel", "@x@type_0");
                        }

                        if (!staticAbstractMRCounts.containsKey(abstractMR)) {
                            staticAbstractMRCounts.put(abstractMR, 1);
                        } else {
                            staticAbstractMRCounts.put(abstractMR, staticAbstractMRCounts.get(abstractMR) + 1);
                        }
                        abstractMRtoMR.put(abstractMR, da);
                        abstractMRtoText.put(abstractMR, predictedWordSequence);

                        String abstractMRCounted = abstractMR;
                        abstractMRCounted += MRCount;
                        MRCount++;
                        if (!predictedWordSequences_overAllPredicates.containsKey(abstractMRCounted)) {
                            predictedWordSequences_overAllPredicates.put(abstractMRCounted, new HashSet<String>());
                        }
                        predictedWordSequences_overAllPredicates.get(abstractMRCounted).add(predictedWordSequence);

                        if (!predictedWordSequences_perPredicates.containsKey(predicate)) {
                            predictedWordSequences_perPredicates.put(predicate, new HashMap<String, HashSet<String>>());
                        }
                        if (!predictedWordSequences_perPredicates.get(predicate).containsKey(abstractMRCounted)) {
                            predictedWordSequences_perPredicates.get(predicate).put(abstractMRCounted, new HashSet<String>());
                        }
                        predictedWordSequences_perPredicates.get(predicate).get(abstractMRCounted).add(predictedWordSequence);

                        Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedWordSequence));
                        ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
                        generations.add(tran);

                        inGen = false;
                        gens++;

                        if (!daToGen.containsKey(da)) {
                            daToGen.put(da.toLowerCase(), s.trim());
                        }
                        int mentioned = 0;
                        int total = 0;
                        ArrayList<String> errors = new ArrayList<String>();
                        String gen = " " + s.trim().toLowerCase() + " ";
                        for (String attr : attributeValues.keySet()) {
                            for (String value : attributeValues.get(attr)) {
                                String searchValue = value;
                                boolean ment = false;
                                if (attr.equals("hasinternet")
                                        && gen.contains("internet")) {
                                    ment = true;
                                } else if (attr.equals("dogsallowed")
                                        && gen.contains("dog")) {
                                    ment = true;
                                } else if (attr.equals("kidsallowed")
                                        && (gen.contains("kid") || gen.contains("child"))) {
                                    ment = true;
                                } else if (attr.equals("goodformeal")
                                        && (gen.contains(searchValue) || gen.contains("meal") || gen.contains("breakfast"))) {
                                    ment = true;
                                } else if (attr.equals("acceptscreditcards")
                                        && gen.contains("credit")) {
                                    ment = true;
                                }
                                if (searchValue.equals("dont_care") || searchValue.equals("none") || searchValue.equals("yes or no") || searchValue.equals("no or yes")) {
                                    searchValue = attr;
                                } else if (searchValue.contains("or")) {
                                    String[] values = searchValue.split(" or ");
                                    for (String v : values) {
                                        if (gen.contains(v)) {
                                            ment = true;
                                        }
                                    }
                                }
                                if (searchValue.equals("pricerange")
                                        && (gen.contains(searchValue) || gen.contains("price") || gen.contains("range"))) {
                                    ment = true;
                                } else if (searchValue.equals("area")
                                        && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                                    ment = true;
                                } else if (searchValue.equals("near")
                                        && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                                    ment = true;
                                } else if (gen.contains(searchValue)) {
                                    ment = true;
                                }

                                if (ment) {
                                    mentioned++;
                                } else {
                                    errors.add("not mentioned -> " + attr + ":" + attributeValues.get(attr));
                                }
                                total++;
                            }
                        }
                        double err = 1.0 - (mentioned / (double) total);
                        if (Double.isNaN(err)) {
                            err = 0.0;
                        }
                        attrCoverage.put(predictedWordSequence, err);
                        if (!errors.isEmpty()) {
                            /* System.out.println("================");
                            System.out.println(da);
                            System.out.println(s.trim());
                            for (String error : errors) {
                            System.out.println(error);
                            }
                            System.out.println("ERR: \t" + err);*/
                        }
                        da = "";
                    }
                } else if (inRef) {
                    if (s.trim().isEmpty()) {
                        for (int i = 0; i < gens; i++) {
                            finalReferences.add(references);
                            finalReferencesWordSequences.put(predictedWordSequence, referencesStrings);
                        }
                        staticReferences.put(abstractMR, references);
                        staticReferencesStrings.put(abstractMR, referencesStrings);
                        references = new ArrayList<>();
                        referencesStrings = new ArrayList<>();
                        inRef = false;
                        da = "";
                    } else {
                        String cleanedWords = s.trim().replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                        referencesStrings.add(cleanedWords);
                        references.add(IStrings.tokenize(NISTTokenizer.tokenize(cleanedWords)));
                    }
                }
            }
            double avgErr = 0.0;
            avgErr = attrCoverage.values().stream().map((err) -> err).reduce(avgErr, (accumulator, _item) -> accumulator + _item);
            avgErr /= attrCoverage.size();
            System.out.println(finalReferences.size() + "\t" + generations.size() + "\t" + attrCoverage.size());
            BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
            NISTMetric NIST = new NISTMetric(finalReferences);
            System.out.println("BLEU: \t" + BLEU.score(generations));
            /*double avgRougeScore = 0.0;
            for (String predictedString : allPredictedWordSequences) {
            double maxRouge = 0.0;
            if (!finalReferencesWordSequences.containsKey(predictedString)) {
            System.out.println(predictedString);
            System.out.println(finalReferencesWordSequences);
            }
            String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
            for (String ref : finalReferencesWordSequences.get(predictedString)) {
            double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
            if (rouge > maxRouge) {
            maxRouge = rouge;
            }
            }
            avgRougeScore += maxRouge;
            }
            System.out.println("ROUGE: \t" + (avgRougeScore / (double) allPredictedWordSequences.size()));*/
            /*double avgRougeScore = 0.0;
            for (String predictedString : allPredictedWordSequences) {
            double maxRouge = 0.0;
            if (!finalReferencesWordSequences.containsKey(predictedString)) {
            System.out.println(predictedString);
            System.out.println(finalReferencesWordSequences);
            }
            
            String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
            for (String ref : finalReferencesWordSequences.get(predictedString)) {
            double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
            if (rouge > maxRouge) {
            maxRouge = rouge;
            }
            }
            avgRougeScore += maxRouge;
            }
            System.out.println("ROUGE: \t" + (avgRougeScore / (double) allPredictedWordSequences.size()));*/
            System.out.println("NIST: \t" + NIST.score(generations));
            //System.out.println(daToGen);
            System.out.println("ERR: \t" + avgErr);
            //System.out.println(daToGen);

            System.out.println("///////////////////");

            ////////////////////////
            //ArrayList<String> bestPredictedStrings = new ArrayList<>();
            //ArrayList<String> bestPredictedStringsMRs = new ArrayList<>();
            double uniqueAllPredWordBLEU = 0.0;
            double uniqueAllPredWordROUGE = 0.0;
            double uniqueAllPredWordCOVERAGEERR = 0.0;
            double uniqueAllPredWordBRC = 0.0;

            String detailedRes = "";
            ArrayList<String> abstractMRList = new ArrayList<>(predictedWordSequences_overAllPredicates.keySet());
            Collections.sort(abstractMRList);
            for (String aMR : abstractMRList) {
                String bestPredictedString = "";
                Double bestROUGE = -100.0;
                Double bestBLEU = -100.0;
                Double bestCover = -100.0;
                Double bestAVG = -100.0;
                Double bestHarmonicMean = -100.0;
                for (String predictedString : predictedWordSequences_overAllPredicates.get(aMR)) {
                    double maxRouge = 0.0;
                    predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                    for (String ref : finalReferencesWordSequences.get(predictedString)) {
                        double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                        if (rouge > maxRouge) {
                            maxRouge = rouge;
                        }
                    }

                    double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(predictedString), 4);
                    double cover = 1.0 - attrCoverage.get(predictedString);
                    double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                    if (harmonicMean > bestHarmonicMean) {
                        bestPredictedString = predictedString;
                        bestROUGE = maxRouge;
                        bestBLEU = BLEUSmooth;
                        bestCover = cover;
                        bestAVG = (BLEUSmooth + maxRouge + cover) / 3.0;
                        bestHarmonicMean = harmonicMean;
                    }
                }
                //bestPredictedStrings.add(bestPredictedString);
                //bestPredictedStringsMRs.add(generationActionsMap.get(generationActions.get(bestPredictedString)).getMeaningRepresentation().getMRstr());

                uniqueAllPredWordBLEU += bestBLEU;
                uniqueAllPredWordROUGE += bestROUGE;
                uniqueAllPredWordCOVERAGEERR += bestCover;
                uniqueAllPredWordBRC += bestHarmonicMean;
                /*if (aMR.equals(examinedAbstractMR) || examinedAbstractMR.isEmpty()) {
                detailedRes += aMR + "\t" + bestPredictedString + "\t" + finalReferencesWordSequences.get(bestPredictedString) + "\t" + bestBLEU + "\t" + bestROUGE + "\t" + bestCover + "\t" + bestAVG + "\t" + bestHarmonicMean + "|";
                }*/
            }
            uniqueAllPredWordBLEU /= predictedWordSequences_overAllPredicates.keySet().size();
            uniqueAllPredWordROUGE /= predictedWordSequences_overAllPredicates.keySet().size();
            uniqueAllPredWordCOVERAGEERR /= predictedWordSequences_overAllPredicates.keySet().size();
            uniqueAllPredWordBRC /= predictedWordSequences_overAllPredicates.keySet().size();
            System.out.println("UNIQUE WORD ALL PRED BLEU: \t" + uniqueAllPredWordBLEU);
            System.out.println("UNIQUE WORD ALL PRED ROUGE: \t" + uniqueAllPredWordROUGE);
            System.out.println("UNIQUE WORD ALL PRED COVERAGE ERROR: \t" + (1.0 - uniqueAllPredWordCOVERAGEERR));
            System.out.println("UNIQUE WORD ALL PRED BRC: \t" + uniqueAllPredWordBRC);
            System.out.println("TOTAL SET SIZE: \t" + abstractMRList.size());
            //System.out.println(abstractMRList);
            //System.out.println(detailedRes);
            //System.out.println(abstractMRList);  
            //System.out.println(detailedRes);
            System.out.println("///////////////////");

            ////////////////////////
            for (String pred : predictedWordSequences_perPredicates.keySet()) {
                detailedRes = "";
                //bestPredictedStrings = new ArrayList<>();
                //bestPredictedStringsMRs = new ArrayList<>();
                double uniquePredWordBLEU = 0.0;
                double uniquePredWordROUGE = 0.0;
                double uniquePredWordCOVERAGEERR = 0.0;
                double uniquePredWordBRC = 0.0;

                abstractMRList = new ArrayList<>(predictedWordSequences_perPredicates.get(pred).keySet());
                Collections.sort(abstractMRList);
                for (String aMR : abstractMRList) {
                    String bestPredictedString = "";
                    Double bestROUGE = -100.0;
                    Double bestBLEU = -100.0;
                    Double bestCover = -100.0;
                    Double bestAVG = -100.0;
                    Double bestHarmonicMean = -100.0;
                    for (String predictedString : predictedWordSequences_perPredicates.get(pred).get(aMR)) {
                        double maxRouge = 0.0;
                        predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                        for (String ref : finalReferencesWordSequences.get(predictedString)) {
                            double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                            if (rouge > maxRouge) {
                                maxRouge = rouge;
                            }
                        }

                        double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(predictedString), 4);
                        double cover = 1.0 - attrCoverage.get(predictedString);
                        double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                        if (harmonicMean > bestHarmonicMean) {
                            bestPredictedString = predictedString;
                            bestROUGE = maxRouge;
                            bestBLEU = BLEUSmooth;
                            bestCover = cover;
                            bestAVG = (BLEUSmooth + maxRouge + cover) / 3.0;
                            bestHarmonicMean = harmonicMean;
                        }
                    }
                    //bestPredictedStrings.add(bestPredictedString);
                    //bestPredictedStringsMRs.add(generationActionsMap.get(generationActions.get(bestPredictedString)).getMeaningRepresentation().getMRstr());

                    uniquePredWordBLEU += bestBLEU;
                    uniquePredWordROUGE += bestROUGE;
                    uniquePredWordCOVERAGEERR += bestCover;
                    uniquePredWordBRC += bestHarmonicMean;
                    //if (aMR.equals(examinedAbstractMR) || examinedAbstractMR.isEmpty()) {
                    //    detailedRes += aMR + "\t" + bestPredictedString + "\t" + finalReferencesWordSequences.get(bestPredictedString) + "\t" + bestBLEU + "\t" + bestROUGE + "\t" + bestCover + "\t" + bestAVG + "\t" + bestHarmonicMean + "|";
                    //}
                }
                uniquePredWordBLEU /= predictedWordSequences_perPredicates.get(pred).keySet().size();
                uniquePredWordROUGE /= predictedWordSequences_perPredicates.get(pred).keySet().size();
                uniquePredWordCOVERAGEERR /= predictedWordSequences_perPredicates.get(pred).keySet().size();
                uniquePredWordBRC /= predictedWordSequences_perPredicates.get(pred).keySet().size();
                System.out.println("UNIQUE WORD " + pred + " BLEU: \t" + uniquePredWordBLEU);
                System.out.println("UNIQUE WORD " + pred + " ROUGE: \t" + uniquePredWordROUGE);
                System.out.println("UNIQUE WORD " + pred + " COVERAGE ERROR: \t" + (1.0 - uniquePredWordCOVERAGEERR));
                System.out.println("UNIQUE WORD " + pred + " BRC: \t" + uniquePredWordBRC);
                System.out.println(pred + " SET SIZE: \t" + abstractMRList.size());
                //System.out.println(detailedRes);
            }

            System.out.println("///////--LOLS--//////");
            // LOLS RECALCULATION            
            String rFileL = "bestSFLOLS_res.txt";
            BufferedReader brLOL = new BufferedReader(new FileReader(rFileL));
            String sL;
            int MRCount_L = 0;
            HashMap<String, HashSet<String>> predictedWordSequences_overAllPredicatesL = new HashMap<>();
            HashMap<String, HashMap<String, HashSet<String>>> predictedWordSequences_perPredicatesL = new HashMap<>();
            ArrayList<ArrayList<Sequence<IString>>> finalReferencesL = new ArrayList<>();
            ArrayList<ScoredFeaturizedTranslation<IString, String>> generationsL = new ArrayList<>();
            ArrayList<Sequence<IString>> referencesL = new ArrayList<>();
            ArrayList<String> referencesStringsL = new ArrayList<>();
            HashMap<String, HashSet<String>> attributeValuesL = new HashMap<>();
            while ((sL = brLOL.readLine()) != null) {
                String[] arr = sL.trim().split("\t");

                /*     if (attributesStr.contains("|")) {
                da = sL.substring(sL.indexOf(":") + 1).trim();
                
                String MRstr = sL.substring(sL.indexOf(":") + 1).replaceAll(",", ";").replaceAll("no or yes", "yes or no").replaceAll("ave ; presidio", "ave and presidio").replaceAll("point ; ste", "point and ste").trim();
                predicate = MRstr.substring(0, MRstr.indexOf("("));
                abstractMR = predicate + ":";
                String attributesStr = MRstr.substring(MRstr.indexOf('(') + 1, MRstr.length() - 1);
                attributeValuesL = new HashMap<>();
                if (!attributesStr.isEmpty()) {
                HashMap<String, Integer> attrXIndeces = new HashMap<>();
                
                String[] args = attributesStr.split(";");
                if (attributesStr.contains("|")) {
                System.out.println(attributesStr);
                System.exit(0);
                }
                for (String arg : args) {
                String attr = "";
                String value = "";
                if (arg.contains("=")) {
                String[] subAttr = arg.split("=");
                value = subAttr[1].toLowerCase();
                attr = subAttr[0].toLowerCase().replaceAll("_", "");
                
                if (value.startsWith("\'")) {
                value = value.substring(1, value.length() - 1);
                }
                if (value.equals("no")
                || value.equals("yes")
                || value.equals("yes or no")
                || value.equals("none")
                || value.equals("empty")) {
                attr += "_" + value.replaceAll(" ", "_");
                value = attr;
                }
                if (value.equals("dont_care")) {
                String v = value;
                value = attr;
                attr = v;
                }
                } else {
                attr = arg.replaceAll("_", "");
                }
                if (!attributeValuesL.containsKey(attr)) {
                attributeValuesL.put(attr, new HashSet<String>());
                }
                if (value.isEmpty()) {
                value = attr;
                }
                
                if (value.startsWith("\'")) {
                value = value.substring(1, value.length() - 1);
                }
                if (value.toLowerCase().startsWith("x")) {
                int index = 0;
                if (!attrXIndeces.containsKey(attr)) {
                attrXIndeces.put(attr, 1);
                } else {
                index = attrXIndeces.get(attr);
                attrXIndeces.put(attr, index + 1);
                }
                value = "x" + index;
                }
                if (value.isEmpty()) {
                System.out.println("EMPTY VALUE");
                System.exit(0);
                }
                
                attributeValuesL.get(attr).add(value.trim().toLowerCase());
                }
                for (String attr : attributeValuesL.keySet()) {
                if (attributeValuesL.get(attr).contains("yes")
                && attributeValuesL.get(attr).contains("no")) {
                System.out.println(MRstr);
                System.out.println(attributeValuesL);
                System.exit(0);
                }
                }
                //System.out.println("\t" + da);
                //System.out.println("\t" + attributeValues);
                
                //System.out.println(MRstr);
                //System.out.println(predicate);
                //System.out.println(abstractMR);
                }
                } else if (sL.startsWith("Gen")) {
                gens = 0;
                inGenL = true;
                } else if (sL.startsWith("Ref")) {
                inRef = true;
                referencesL = new ArrayList<>();
                } else if (inGenL) {*/
                predictedWordSequence = arr[1].trim().toLowerCase();
                /*String ref = predictedWordSequence;
                
                HashMap<String, HashMap<String, Integer>> attrValuePriorities = new HashMap<>();
                HashMap<String, HashSet<String>> delexAttributeValues = new HashMap<>();
                int prio = 0;
                for (String attr : attributeValuesL.keySet()) {
                if (!attr.isEmpty()) {
                delexAttributeValues.put(attr, new HashSet<String>());
                if (attr.equals("name")
                || attr.equals("type")
                || attr.equals("pricerange")
                || attr.equals("price")
                || attr.equals("phone")
                || attr.equals("address")
                || attr.equals("postcode")
                || attr.equals("area")
                || attr.equals("near")
                || attr.equals("food")
                || attr.equals("count")
                || attr.equals("dont_care")
                || attr.equals("goodformeal")) {
                attrValuePriorities.put(attr, new HashMap<String, Integer>());
                for (String value : attributeValuesL.get(attr)) {
                if (!value.equals("dont_care")
                && !value.equals("none")
                && !value.equals("empty")
                && !value.equals(attr)) {
                attrValuePriorities.get(attr).put(value, prio);
                prio++;
                } else {
                delexAttributeValues.get(attr).add(value);
                }
                }
                } else {
                for (String value : attributeValuesL.get(attr)) {
                delexAttributeValues.get(attr).add(value);
                }
                }
                }
                }
                boolean change = true;
                while (change) {
                change = false;
                for (String attr1 : attrValuePriorities.keySet()) {
                for (String value1 : attrValuePriorities.get(attr1).keySet()) {
                for (String attr2 : attrValuePriorities.keySet()) {
                for (String value2 : attrValuePriorities.get(attr2).keySet()) {
                if (!value1.equals(value2)
                && value1.contains(value2)
                && attrValuePriorities.get(attr1).get(value1) > attrValuePriorities.get(attr2).get(value2)) {
                int prio1 = attrValuePriorities.get(attr1).get(value1);
                int prio2 = attrValuePriorities.get(attr2).get(value2);
                attrValuePriorities.get(attr1).put(value1, prio2);
                attrValuePriorities.get(attr2).put(value2, prio1);
                change = true;
                }
                }
                }
                }
                }
                }
                HashMap<String, Integer> xCounts = new HashMap<>();
                HashMap<String, String> delexMap = new HashMap<>();
                ref = " " + ref + " ";
                for (int p = 0; p < prio; p++) {
                for (String attr : attrValuePriorities.keySet()) {
                if (!xCounts.containsKey(attr)) {
                xCounts.put(attr, 0);
                }
                for (String value : attrValuePriorities.get(attr).keySet()) {
                if (attrValuePriorities.get(attr).get(value) == p) {
                if (!ref.contains(" " + value + " ")
                && !value.contains(" and ")
                && !value.contains(" or ")) {
                delexAttributeValues.get(attr).add(value);
                } else if (!ref.contains(" " + value + " ")
                && (value.contains(" and ")
                || value.contains(" or "))) {
                String[] values = null;
                if (value.contains(" and ")) {
                values = value.split(" and ");
                } else if (value.contains(" or ")) {
                values = value.split(" or ");
                }
                for (int v = 0; v < values.length; v++) {
                if (!ref.contains(" " + values[v] + " ")) {
                } else {
                ref = ref.replace(" " + values[v] + " ", " " + Action.TOKEN_X + attr + "_" + xCounts.get(attr) + " ");
                ref = ref.replaceAll("  ", " ");
                delexAttributeValues.get(attr).add(Action.TOKEN_X + attr + "_" + xCounts.get(attr));
                delexMap.put(Action.TOKEN_X + attr + "_" + xCounts.get(attr), values[v]);
                xCounts.put(attr, xCounts.get(attr) + 1);
                }
                }
                } else {
                ref = ref.replace(" " + value + " ", " " + Action.TOKEN_X + attr + "_" + xCounts.get(attr) + " ");
                ref = ref.replaceAll("  ", " ");
                delexAttributeValues.get(attr).add(Action.TOKEN_X + attr + "_" + xCounts.get(attr));
                delexMap.put(Action.TOKEN_X + attr + "_" + xCounts.get(attr), value);
                xCounts.put(attr, xCounts.get(attr) + 1);
                }
                }
                }
                }
                }
                ref = ref.trim();
                
                ArrayList<String> attrs = new ArrayList<>(delexAttributeValues.keySet());
                Collections.sort(attrs);
                for (String attr : attrs) {
                abstractMR += attr + "={";
                
                ArrayList<String> values = new ArrayList<>(delexAttributeValues.get(attr));
                Collections.sort(values);
                for (String value : values) {
                abstractMR += value + ",";
                }
                abstractMR += "}";
                }
                if (abstractMR.equals("inform:address={@x@address_0,}name={@x@name_0,}phone={@x@phone_0,}pricerange={moderate,}")) {
                abstractMR = "inform:address={@x@address_0,}name={@x@name_0,}phone={@x@phone_0,}pricerange={@x@pricerange_0,}";
                }
                if (abstractMR.equals("inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={moderate,}")) {
                abstractMR = "inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={@x@pricerange_0,}";
                }
                if (abstractMR.equals("inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={pricey,}")) {
                abstractMR = "inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={@x@pricerange_0,}";
                }
                if (abstractMR.equals("?confirm:pricerange={cheap,}")) {
                abstractMR = "?confirm:pricerange={@x@pricerange_0,}";
                }
                if (predictedWordSequence.contains(" hotels ")) {
                abstractMR = abstractMR.replaceAll("hotel", "@x@type_0");
                }*/

                abstractMR = arr[0].trim();
                if (abstractMR.equals("inform:address={@x@address_0,}name={@x@name_0,}phone={@x@phone_0,}pricerange={moderate,}")) {
                    abstractMR = "inform:address={@x@address_0,}name={@x@name_0,}phone={@x@phone_0,}pricerange={@x@pricerange_0,}";
                }
                if (abstractMR.equals("inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={moderate,}")) {
                    abstractMR = "inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={@x@pricerange_0,}";
                }
                if (abstractMR.equals("inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={pricey,}")) {
                    abstractMR = "inform:name={@x@name_0,}phone={@x@phone_0,}pricerange={@x@pricerange_0,}";
                }
                if (abstractMR.equals("?confirm:pricerange={cheap,}")) {
                    abstractMR = "?confirm:pricerange={@x@pricerange_0,}";
                }
                //if (predictedWordSequence.contains(" hotels ")
                //      || predictedWordSequence.contains(" hotel ")) {
                abstractMR = abstractMR.replaceAll("hotel", "@x@type_0");
                //}
                //da = arr[1].trim();
                //abstractMRtoMR.put(abstractMR, da);
                abstractMRtoText.put(abstractMR, predictedWordSequence);
                predicate = abstractMR.substring(0, abstractMR.indexOf(':'));

                for (int i = 0; i < staticAbstractMRCounts.get(abstractMR); i++) {
                    String abstractMRCounted = abstractMR;
                    abstractMRCounted += MRCount_L;
                    MRCount_L++;
                    if (!predictedWordSequences_overAllPredicatesL.containsKey(abstractMRCounted)) {
                        predictedWordSequences_overAllPredicatesL.put(abstractMRCounted, new HashSet<String>());
                    }
                    predictedWordSequences_overAllPredicatesL.get(abstractMRCounted).add(predictedWordSequence);

                    if (!predictedWordSequences_perPredicatesL.containsKey(predicate)) {
                        predictedWordSequences_perPredicatesL.put(predicate, new HashMap<String, HashSet<String>>());
                    }
                    if (!predictedWordSequences_perPredicatesL.get(predicate).containsKey(abstractMRCounted)) {
                        predictedWordSequences_perPredicatesL.get(predicate).put(abstractMRCounted, new HashSet<String>());
                    }
                    predictedWordSequences_perPredicatesL.get(predicate).get(abstractMRCounted).add(predictedWordSequence);

                    Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedWordSequence));
                    ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
                    generationsL.add(tran);

                    /*if (!daToGen.containsKey(da)) {
                    daToGen.put(da.toLowerCase(), sL.trim());
                    }
                    int mentioned = 0;
                    int total = 0;
                    ArrayList<String> errors = new ArrayList<String>();
                    String gen = " " + sL.trim().toLowerCase() + " ";
                    for (String attr : attributeValuesL.keySet()) {
                    for (String value : attributeValuesL.get(attr)) {
                    String searchValue = value;
                    boolean ment = false;
                    if (attr.equals("hasinternet")
                    && gen.contains("internet")) {
                    ment = true;
                    } else if (attr.equals("dogsallowed")
                    && gen.contains("dog")) {
                    ment = true;
                    } else if (attr.equals("kidsallowed")
                    && (gen.contains("kid") || gen.contains("child"))) {
                    ment = true;
                    } else if (attr.equals("goodformeal")
                    && (gen.contains(searchValue) || gen.contains("meal") || gen.contains("breakfast"))) {
                    ment = true;
                    } else if (attr.equals("acceptscreditcards")
                    && gen.contains("credit")) {
                    ment = true;
                    }
                    if (searchValue.equals("dont_care") || searchValue.equals("none") || searchValue.equals("yes or no") || searchValue.equals("no or yes")) {
                    searchValue = attr;
                    } else if (searchValue.contains("or")) {
                    String[] values = searchValue.split(" or ");
                    for (String v : values) {
                    if (gen.contains(v)) {
                    ment = true;
                    }
                    }
                    }
                    if (searchValue.equals("pricerange")
                    && (gen.contains(searchValue) || gen.contains("price") || gen.contains("range"))) {
                    ment = true;
                    } else if (searchValue.equals("area")
                    && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                    ment = true;
                    } else if (searchValue.equals("near")
                    && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                    ment = true;
                    } else if (gen.contains(searchValue)) {
                    ment = true;
                    }
                    
                    if (ment) {
                    mentioned++;
                    } else {
                    errors.add("not mentioned -> " + attr + ":" + attributeValuesL.get(attr));
                    }
                    total++;
                    }
                    }
                    double err = 1.0 - ((double) mentioned / (double) total);
                    if (Double.isNaN(err)) {
                    err = 0.0;
                    }
                    attrCoverage.put(predictedWordSequence, err);
                    if (!errors.isEmpty()) {
                    System.out.println("================");
                    System.out.println(da);
                    System.out.println(sL.trim());
                    for (String error : errors) {
                    System.out.println(error);
                    }
                    System.out.println("ERR: \t" + err);
                    }
                    da = "";*/
                    referencesL = staticReferences.get(abstractMR);
                    referencesStringsL = staticReferencesStrings.get(abstractMR);
                    if (referencesL == null) {
                        System.out.println(abstractMR);
                    }
                    //for (int j = 0; j < gens; j++) {
                    finalReferencesL.add(referencesL);
                    finalReferencesWordSequences.put(predictedWordSequence, referencesStringsL);
                    //}
                }
            }
            System.out.println(finalReferencesL.size() + "\t" + generationsL.size() + "\t" + attrCoverage.size());
            BLEUMetric BLEU_L = new BLEUMetric(finalReferencesL, 4, false);
            NISTMetric NIST_L = new NISTMetric(finalReferencesL);
            System.out.println("BLEU_LOLS: \t" + BLEU_L.score(generationsL));

            double avgRougeScore = 0.0;
            /*for (String predictedString : allPredictedWordSequences) {
            double maxRouge = 0.0;
            if (!finalReferencesWordSequences.containsKey(predictedString)) {
            System.out.println(predictedString);
            System.out.println(finalReferencesWordSequences);
            }
            String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
            for (String ref : finalReferencesWordSequences.get(predictedString)) {
            double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
            if (rouge > maxRouge) {
            maxRouge = rouge;
            }
            }
            avgRougeScore += maxRouge;
            }
            System.out.println("ROUGE_L: \t" + (avgRougeScore / (double) allPredictedWordSequences.size()));*/
            //System.out.println("ERR_L: \t" + avgErr);
            //System.out.println(daToGen);
            /*for (String predictedString : allPredictedWordSequences) {
            double maxRouge = 0.0;
            if (!finalReferencesWordSequences.containsKey(predictedString)) {
            System.out.println(predictedString);
            System.out.println(finalReferencesWordSequences);
            }
            
            String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
            for (String ref : finalReferencesWordSequences.get(predictedString)) {
            double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
            if (rouge > maxRouge) {
            maxRouge = rouge;
            }
            }
            avgRougeScore += maxRouge;
            }
            System.out.println("ROUGE_L: \t" + (avgRougeScore / (double) allPredictedWordSequences.size()));*/
            System.out.println("NIST_LOLS: \t" + NIST_L.score(generationsL));
            //System.out.println("ERR_L: \t" + avgErr);
            //System.out.println(daToGen);

            System.out.println("///////////////////");

            ////////////////////////
            //ArrayList<String> bestPredictedStrings = new ArrayList<>();
            //ArrayList<String> bestPredictedStringsMRs = new ArrayList<>();
            double uniqueAllPredWordBLEU_L = 0.0;
            double uniqueAllPredWordROUGE_L = 0.0;

            String detailedRes_L = "";
            ArrayList<String> abstractMRList_L = new ArrayList<>(predictedWordSequences_overAllPredicatesL.keySet());
            Collections.sort(abstractMRList_L);
            for (String aMR : abstractMRList_L) {
                String bestPredictedString = "";
                Double bestROUGE = -100.0;
                Double bestBLEU = -100.0;
                Double bestCover = -100.0;
                Double bestAVG = -100.0;
                Double bestHarmonicMean = -100.0;
                for (String predictedString : predictedWordSequences_overAllPredicatesL.get(aMR)) {
                    double maxRouge = 0.0;
                    predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                    for (String ref : finalReferencesWordSequences.get(predictedString)) {
                        double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                        if (rouge > maxRouge) {
                            maxRouge = rouge;
                        }
                    }

                    double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(predictedString), 4);
//                    double cover = 1.0 - attrCoverage.get(predictedString);
                    double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge/* + 1.0 / cover*/);

                    if (harmonicMean > bestHarmonicMean) {
                        bestPredictedString = predictedString;
                        bestROUGE = maxRouge;
                        bestBLEU = BLEUSmooth;
                        //bestCover = cover;
                        //bestAVG = (BLEUSmooth + maxRouge + cover) / 3.0;
                        bestHarmonicMean = harmonicMean;
                    }
                }
                //bestPredictedStrings.add(bestPredictedString);
                //bestPredictedStringsMRs.add(generationActionsMap.get(generationActions.get(bestPredictedString)).getMeaningRepresentation().getMRstr());

                uniqueAllPredWordBLEU_L += bestBLEU;
                uniqueAllPredWordROUGE_L += bestROUGE;
                /*if (aMR.equals(examinedAbstractMR) || examinedAbstractMR.isEmpty()) {
                detailedRes += aMR + "\t" + bestPredictedString + "\t" + finalReferencesWordSequences.get(bestPredictedString) + "\t" + bestBLEU + "\t" + bestROUGE + "\t" + bestCover + "\t" + bestAVG + "\t" + bestHarmonicMean + "|";
                }*/
            }
            uniqueAllPredWordBLEU_L /= predictedWordSequences_overAllPredicatesL.keySet().size();
            uniqueAllPredWordROUGE_L /= predictedWordSequences_overAllPredicatesL.keySet().size();
            System.out.println("UNIQUE WORD ALL PRED BLEU_LOLS: \t" + uniqueAllPredWordBLEU_L);
            System.out.println("UNIQUE WORD ALL PRED ROUGE_LOLS: \t" + uniqueAllPredWordROUGE_L);
            System.out.println("TOTAL SET SIZE: \t" + abstractMRList_L.size());
            //System.out.println(abstractMRList);  
            //System.out.println(detailedRes);

            ////////////////////////
            for (String pred : predictedWordSequences_perPredicatesL.keySet()) {
                detailedRes_L = "";
                //bestPredictedStrings = new ArrayList<>();
                //bestPredictedStringsMRs = new ArrayList<>();
                double uniquePredWordBLEU = 0.0;
                double uniquePredWordROUGE = 0.0;
                double uniquePredWordCOVERAGEERR = 0.0;
                double uniquePredWordBRC = 0.0;

                abstractMRList_L = new ArrayList<>(predictedWordSequences_perPredicatesL.get(pred).keySet());
                Collections.sort(abstractMRList_L);
                for (String aMR : abstractMRList_L) {
                    String bestPredictedString = "";
                    Double bestROUGE = -100.0;
                    Double bestBLEU = -100.0;
                    Double bestCover = -100.0;
                    Double bestAVG = -100.0;
                    Double bestHarmonicMean = -100.0;
                    for (String predictedString : predictedWordSequences_perPredicatesL.get(pred).get(aMR)) {
                        double maxRouge = 0.0;
                        predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                        for (String ref : finalReferencesWordSequences.get(predictedString)) {
                            double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                            if (rouge > maxRouge) {
                                maxRouge = rouge;
                            }
                        }

                        double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(predictedString), 4);
                        //double cover = 1.0 - attrCoverage.get(predictedString);
                        double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge/* + 1.0 / cover*/);

                        if (harmonicMean > bestHarmonicMean) {
                            bestPredictedString = predictedString;
                            bestROUGE = maxRouge;
                            bestBLEU = BLEUSmooth;
                            //bestCover = cover;
                            //bestAVG = (BLEUSmooth + maxRouge + cover) / 3.0;
                            bestHarmonicMean = harmonicMean;
                        }
                    }
                    //bestPredictedStrings.add(bestPredictedString);
                    //bestPredictedStringsMRs.add(generationActionsMap.get(generationActions.get(bestPredictedString)).getMeaningRepresentation().getMRstr());

                    uniquePredWordBLEU += bestBLEU;
                    uniquePredWordROUGE += bestROUGE;
                    uniquePredWordCOVERAGEERR += bestCover;
                    uniquePredWordBRC += bestHarmonicMean;
                    //if (aMR.equals(examinedAbstractMR) || examinedAbstractMR.isEmpty()) {
                    //    detailedRes += aMR + "\t" + bestPredictedString + "\t" + finalReferencesWordSequences.get(bestPredictedString) + "\t" + bestBLEU + "\t" + bestROUGE + "\t" + bestCover + "\t" + bestAVG + "\t" + bestHarmonicMean + "|";
                    //}
                }
                uniquePredWordBLEU /= predictedWordSequences_perPredicatesL.get(pred).keySet().size();
                uniquePredWordROUGE /= predictedWordSequences_perPredicatesL.get(pred).keySet().size();
                uniquePredWordCOVERAGEERR /= predictedWordSequences_perPredicatesL.get(pred).keySet().size();
                uniquePredWordBRC /= predictedWordSequences_perPredicatesL.get(pred).keySet().size();
                System.out.println("UNIQUE WORD " + pred + " BLEU_LOLS: \t" + uniquePredWordBLEU);
                System.out.println("UNIQUE WORD " + pred + " ROUGE_LOLS: \t" + uniquePredWordROUGE);
                //System.out.println("UNIQUE WORD " + pred + " COVERAGE ERROR: \t" + (1.0 - uniquePredWordCOVERAGEERR));
                //System.out.println("UNIQUE WORD " + pred + " BRC: \t" + uniquePredWordBRC);
                //System.out.println("UNIQUE WORD " + pred + " COVERAGE ERROR: \t" + (1.0 - uniquePredWordCOVERAGEERR));
                //System.out.println("UNIQUE WORD " + pred + " BRC: \t" + uniquePredWordBRC);
                System.out.println(pred + " SET SIZE: \t" + abstractMRList_L.size());
                //System.out.println(detailedRes);
            }

        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }

        /*for (int i = 0; i < mrs.size(); i++) {
        String MR = mrs.get(i);
        if (daToGen.get(MR.toLowerCase()) != null) {
        System.out.println(daToGen.get(MR.toLowerCase()));
        } else {
        System.out.println("!!!!!!!!!!!!!! " + MR.toLowerCase());
        }
        }*/
        /*for (String abstractMR : abstractMRtoMR.keySet()) {
        System.out.println(abstractMR + "\t" + abstractMRtoMR.get(abstractMR) + "\t" + abstractMRtoText.get(abstractMR));
        }*/
    }

    /**
     *
     */
    public static void parseERR() {
        String mrFile = "tempGens.txt";
        HashMap<String, String> daToGen = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(mrFile))) {
            String s;
            String da = "";
            int gens = 0;
            ArrayList<Double> errs = new ArrayList<>();
            HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
            while ((s = br.readLine()) != null) {
                String[] arr = s.trim().split("\t");
                da = arr[0].substring(arr[0].indexOf(':') + 1).toLowerCase().trim();
                String gen = " " + arr[1].trim().toLowerCase() + " ";

                String MRstr = arr[0].substring(arr[0].indexOf(':') + 1).replaceAll(",", ";").replaceAll("no or yes", "yes or no").replaceAll("ave ; presidio", "ave and presidio").replaceAll("point ; ste", "point and ste").trim();
                String attributesStr = MRstr.substring(MRstr.indexOf('(') + 1, MRstr.length() - 1);
                attributeValues = new HashMap<>();
                if (!attributesStr.isEmpty()) {
                    HashMap<String, Integer> attrXIndeces = new HashMap<>();

                    String[] args = attributesStr.split(";");
                    if (attributesStr.contains("|")) {
                        System.exit(0);
                    }
                    for (String arg : args) {
                        String attr = "";
                        String value = "";
                        if (arg.contains("=")) {
                            String[] subAttr = arg.split("=");
                            value = subAttr[1].toLowerCase();
                            attr = subAttr[0].toLowerCase().replaceAll("_", "");
                        } else {
                            attr = arg.replaceAll("_", "");
                        }
                        if (!attributeValues.containsKey(attr)) {
                            attributeValues.put(attr, new HashSet<String>());
                        }
                        if (value.isEmpty()) {
                            value = attr;
                        }

                        if (value.startsWith("\'")) {
                            value = value.substring(1, value.length() - 1);
                        }
                        if (value.toLowerCase().startsWith("x")) {
                            int index = 0;
                            if (!attrXIndeces.containsKey(attr)) {
                                attrXIndeces.put(attr, 1);
                            } else {
                                index = attrXIndeces.get(attr);
                                attrXIndeces.put(attr, index + 1);
                            }
                            value = "x" + index;
                        }
                        if (value.isEmpty()) {
                            System.exit(0);
                        }

                        attributeValues.get(attr).add(value.trim().toLowerCase());
                    }
                    for (String attr : attributeValues.keySet()) {
                        if (attributeValues.get(attr).contains("yes")
                                && attributeValues.get(attr).contains("no")) {
                            System.out.println(MRstr);
                            System.exit(0);
                        }
                    }
                    //System.out.println("\t" + da);
                    //System.out.println("\t" + attributeValues);
                    //System.out.println("\t" + gen);

                    int mentioned = 0;
                    int total = 0;
                    ArrayList<String> errors = new ArrayList<String>();
                    for (String attr : attributeValues.keySet()) {
                        for (String value : attributeValues.get(attr)) {
                            String searchValue = value;
                            boolean ment = false;
                            if (attr.equals("hasinternet")
                                    && gen.contains("internet")) {
                                ment = true;
                            } else if (attr.equals("dogsallowed")
                                    && gen.contains("dog")) {
                                ment = true;
                            } else if (attr.equals("kidsallowed")
                                    && (gen.contains("kid") || gen.contains("child"))) {
                                ment = true;
                            } else if (attr.equals("goodformeal")
                                    && (gen.contains(searchValue) || gen.contains("meal") || gen.contains("breakfast"))) {
                                ment = true;
                            } else if (attr.equals("acceptscreditcards")
                                    && gen.contains("credit")) {
                                ment = true;
                            }
                            if (searchValue.equals("dont_care") || searchValue.equals("none") || searchValue.equals("yes or no") || searchValue.equals("no or yes")) {
                                searchValue = attr;
                            } else if (searchValue.contains("or")) {
                                String[] values = searchValue.split(" or ");
                                for (String v : values) {
                                    if (gen.contains(v)) {
                                        ment = true;
                                    }
                                }
                            }
                            if (searchValue.equals("pricerange")
                                    && (gen.contains(searchValue) || gen.contains("price") || gen.contains("range"))) {
                                ment = true;
                            } else if (searchValue.equals("area")
                                    && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                                ment = true;
                            } else if (searchValue.equals("near")
                                    && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                                ment = true;
                            } else if (gen.contains(searchValue)) {
                                ment = true;
                            }

                            if (ment) {
                                mentioned++;
                            } else {
                                errors.add("not mentioned -> " + attr + ":" + attributeValues.get(attr));
                            }
                            total++;
                        }
                    }
                    double err = 1.0 - (mentioned / (double) total);
                    if (Double.isNaN(err)) {
                        err = 0.0;
                    }
                    errs.add(err);
                    if (!errors.isEmpty()) {
                        System.out.println("================");
                        System.out.println(da);
                        System.out.println(s.trim());
                        errors.forEach((error) -> {
                            System.out.println(error);
                        });
                    }
                } else {
                    errs.add(0.0);
                }

            }
            double avgErr = 0.0;
            avgErr = errs.stream().map((err) -> err).reduce(avgErr, (accumulator, _item) -> accumulator + _item);
            avgErr = avgErr / errs.size() * 100;
            System.out.println("\t" + errs.size());
            System.out.println("ERR: \t" + avgErr);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     *
     */
    public static void parseERRBagel() {
        String mrFile = "tempGens.txt";
        HashMap<String, String> daToGen = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(mrFile))) {
            String s;
            String da = "";
            int gens = 0;
            ArrayList<Double> errs = new ArrayList<>();
            HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
            while ((s = br.readLine()) != null) {
                String[] arr = s.trim().split("\t");
                da = arr[0].substring(arr[0].indexOf(':') + 1).toLowerCase().trim();
                String gen = " " + arr[1].trim().toLowerCase() + " ";

                String MRstr = new String(da.substring(da.indexOf('(') + 1, da.lastIndexOf(')')));

                HashMap<String, String> names = new HashMap<>();
                int s1 = MRstr.indexOf('"');
                int a = 0;
                while (s1 != -1) {
                    int e = MRstr.indexOf('"', s1 + 1);

                    String name = MRstr.substring(s1, e + 1);
                    MRstr = MRstr.replace(name, "x" + a);
                    names.put("x" + a, name);
                    a++;

                    s1 = MRstr.indexOf('"');
                }

                attributeValues = new HashMap<>();
                String[] args = MRstr.split(",");

                HashMap<String, Integer> attrXIndeces = new HashMap<>();
                for (String arg : args) {
                    String[] subAttr = arg.split("=");
                    String value = subAttr[1];
                    if (names.containsKey(value)) {
                        value = names.get(value);
                    }
                    String attr = subAttr[0].toLowerCase();
                    if (!attributeValues.containsKey(attr)) {
                        attributeValues.put(attr, new HashSet<String>());
                    }
                    if (value.startsWith("\"")) {
                        value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                    }
                    if (value.toLowerCase().startsWith("x")) {
                        int index = 0;
                        if (!attrXIndeces.containsKey(attr)) {
                            attrXIndeces.put(attr, 1);
                        } else {
                            index = attrXIndeces.get(attr);
                            attrXIndeces.put(attr, index + 1);
                        }
                        value = "x";
                    }
                    attributeValues.get(attr).add(value.toLowerCase());
                }
                //System.out.println("\t" + da);
                //System.out.println("\t" + attributeValues);
                //System.out.println("\t" + gen);

                int mentioned = 0;
                int total = 0;
                ArrayList<String> errors = new ArrayList<String>();
                for (String attr : attributeValues.keySet()) {
                    for (String value : attributeValues.get(attr)) {
                        String searchValue = value;
                        boolean ment = false;
                        if (attr.equals("hasinternet")
                                && gen.contains("internet")) {
                            ment = true;
                        } else if (attr.equals("dogsallowed")
                                && gen.contains("dog")) {
                            ment = true;
                        } else if (attr.equals("kidsallowed")
                                && (gen.contains("kid") || gen.contains("child"))) {
                            ment = true;
                        } else if (attr.equals("goodformeal")
                                && (gen.contains(searchValue) || gen.contains("meal") || gen.contains("breakfast"))) {
                            ment = true;
                        } else if (attr.equals("acceptscreditcards")
                                && gen.contains("credit")) {
                            ment = true;
                        }
                        if (searchValue.equals("dont_care") || searchValue.equals("none") || searchValue.equals("yes or no") || searchValue.equals("no or yes")) {
                            searchValue = attr;
                        } else if (searchValue.contains("or")) {
                            String[] values = searchValue.split(" or ");
                            for (String v : values) {
                                if (gen.contains(v)) {
                                    ment = true;
                                }
                            }
                        } else if (searchValue.contains("_")) {
                            String[] values = searchValue.split("_");
                            for (String v : values) {
                                if (gen.contains(v)) {
                                    ment = true;
                                }
                            }
                        }
                        if (searchValue.equals("pricerange")
                                && (gen.contains(searchValue) || gen.contains("price") || gen.contains("range"))) {
                            ment = true;
                        } else if (searchValue.equals("citycentre")
                                && (gen.contains(searchValue) || gen.contains("city") || gen.contains("centre"))) {
                            ment = true;
                        } else if (searchValue.equals("restaurant")
                                && (gen.contains(searchValue) || gen.contains("eatery"))) {
                            ment = true;
                        } else if (searchValue.contains("takeaway")
                                && (gen.contains(searchValue) || gen.contains("take"))) {
                            ment = true;
                        } else if (searchValue.equals("fastfood")
                                && (gen.contains(searchValue) || gen.contains("fast") || gen.contains("food"))) {
                            ment = true;
                        } else if (searchValue.equals("area")
                                && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                            ment = true;
                        } else if (searchValue.equals("near")
                                && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                            ment = true;
                        } else if (gen.contains(searchValue)) {
                            ment = true;
                        }

                        if (ment && !value.equals("none")) {
                            mentioned++;
                        } else {
                            errors.add("not mentioned -> " + attr + ":" + value);
                        }
                        total++;
                    }
                }
                double err = 1.0 - (mentioned / (double) total);
                if (Double.isNaN(err)) {
                    err = 0.0;
                }
                errs.add(err);
                if (!errors.isEmpty()) {
                    if (errors.size() == 1 && (errors.get(0).contains("placetoeat") || !errors.get(0).equals("none"))) {
                    } else {
                        System.out.println("================");
                        System.out.println(da);
                        System.out.println(gen.trim());
                        errors.forEach((error) -> {
                            System.out.println(error);
                        });
                    }
                }
            }
            double avgErr = 0.0;
            avgErr = errs.stream().map((err) -> err).reduce(avgErr, (accumulator, _item) -> accumulator + _item);
            avgErr = avgErr / errs.size() * 100;
            System.out.println("\t" + errs.size());
            System.out.println("ERR: \t" + avgErr);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     *
     */
    public static void generateQSUBforSFXHotel() {
        String command = "";

        String data = "HOTEL";
        String dataset = "hotel";

        String folder = "C:\\Dropbox\\TO-DO\\JAROW-master\\qsubs\\SFXHotel\\";
        //String folder = "D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW-master\\qsubs\\SFXHotel\\";
        //PRINT RESULTS
        ArrayList<Double> learningRates = new ArrayList<Double>();
        //learningRates.add(0.0);
        learningRates.add(0.1);
        learningRates.add(0.2);
        learningRates.add(0.3);
        //learningRates.add(0.4);
        ArrayList<String> earlyTerminationParams = new ArrayList<String>();
        earlyTerminationParams.add("0");
        earlyTerminationParams.add("1");
        earlyTerminationParams.add("2");
        //earlyTerminationParams.add("3");
        //earlyTerminationParams.add("4");
        earlyTerminationParams.add("inf");
        ArrayList<String> costingFunctions = new ArrayList<String>();
        costingFunctions.add("B");
        costingFunctions.add("R");
        costingFunctions.add("BRC");
        for (Double learningRate : learningRates) {
            for (String earlyTerminationParam : earlyTerminationParams) {
                for (String costingFunction : costingFunctions) {
                    BufferedWriter bw = null;
                    File f = null;
                    try {
                        f = new File(folder + "run_SFX_" + data + "_" + earlyTerminationParam + "_" + learningRate + "_" + costingFunction + ".sh");
                        command += "qsub " + "run_SFX_" + data + "_" + earlyTerminationParam + "_" + learningRate + "_" + costingFunction + ".sh;";
                    } catch (NullPointerException e) {
                    }

                    try {
                        bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
                    } catch (FileNotFoundException e) {
                    }

                    try {
                        bw.write("#!/bin/bash" + "\n");
                        bw.write("#$ -l mem=32g -l rmem=32g -l h_rt=24:00:00" + "\n");
                        bw.write("module load apps/java/1.8u71" + "\n");
                        bw.write("\n");
                        if (earlyTerminationParam.equals("inf")) {
                            bw.write("java -Xmx20g -jar JDagger-" + data + ".jar " + 100 + " " + learningRate + " " + dataset + " " + costingFunction + " -classpath /lib > results_SFX_" + data + "_" + earlyTerminationParam + "_" + learningRate + "_" + costingFunction + ".txt" + "\n");
                        } else {
                            bw.write("java -Xmx20g -jar JDagger-" + data + ".jar " + earlyTerminationParam + " " + learningRate + " " + dataset + " " + costingFunction + " -classpath /lib > results_SFX_" + data + "_" + earlyTerminationParam + "_" + learningRate + "_" + costingFunction + ".txt" + "\n");
                        }
                    } catch (IOException e) {
                    }
                    try {
                        bw.close();
                    } catch (IOException e) {
                    }
                }
            }
        }

        //Write command file
        BufferedWriter bw = null;
        File f = null;
        try {
            f = new File(folder + "command" + data + ".txt");
        } catch (NullPointerException e) {
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
        } catch (FileNotFoundException e) {
        }

        try {
            bw.write(command);
        } catch (IOException e) {
        }
        try {
            bw.close();
        } catch (IOException e) {
        }
    }

    /**
     *
     */
    public static void generateQSUBforBAGEL() {
        String command = "";

        String folder = "C:\\Dropbox\\TO-DO\\JAROW-master\\qsubs\\BAGEL\\";
        //PRINT RESULTS
        ArrayList<Double> learningRates = new ArrayList<Double>();
        //learningRates.add(0.0);
        //learningRates.add(0.1);
        learningRates.add(0.2);
        //learningRates.add(0.3);
        //learningRates.add(0.4);
        ArrayList<String> earlyTerminationParams = new ArrayList<String>();
        //earlyTerminationParams.add("0");
        earlyTerminationParams.add("1");
        //earlyTerminationParams.add("2");
        //earlyTerminationParams.add("3");
        //earlyTerminationParams.add("4");
        //earlyTerminationParams.add("inf");
        ArrayList<String> costingFunctions = new ArrayList<String>();
        //costingFunctions.add("B");
        //costingFunctions.add("R");
        costingFunctions.add("BRC");
        for (Double learningRate : learningRates) {
            for (String earlyTerminationParam : earlyTerminationParams) {
                for (String costingFunction : costingFunctions) {
                    for (int fold = 0; fold <= 9; fold++) {
                        BufferedWriter bw = null;
                        File f = null;
                        try {
                            f = new File(folder + "run_" + fold + "_" + earlyTerminationParam + "_" + learningRate + ".sh");
                            command += "qsub " + "run_" + fold + "_" + earlyTerminationParam + "_" + learningRate + ".sh;";
                        } catch (NullPointerException e) {
                        }

                        try {
                            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
                        } catch (FileNotFoundException e) {
                        }

                        try {
                            bw.write("#!/bin/bash" + "\n");
                            bw.write("#$ -l mem=32g -l rmem=32g -l h_rt=48:00:00" + "\n");
                            bw.write("module load apps/java/1.8u71" + "\n");
                            bw.write("\n");
                            if (earlyTerminationParam.equals("inf")) {
                                bw.write("java -Xmx20g -jar JDagger-BAGEL.jar " + fold + " " + 100 + " " + learningRate + " " + costingFunction + " -classpath /lib > results_BAGEL_fold_" + fold + "_" + earlyTerminationParam + "_" + learningRate + ".txt" + "\n");
                            } else {
                                bw.write("java -Xmx20g -jar JDagger-BAGEL.jar " + fold + " " + earlyTerminationParam + " " + learningRate + " " + costingFunction + " -classpath /lib > results_BAGEL_fold_" + fold + "_" + earlyTerminationParam + "_" + learningRate + ".txt" + "\n");
                            }
                        } catch (IOException e) {
                        }
                        try {
                            bw.close();
                        } catch (IOException e) {
                        }
                    }
                }
            }
        }

        //Write command file
        BufferedWriter bw = null;
        File f = null;
        try {
            f = new File(folder + "command.txt");
        } catch (NullPointerException e) {
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
        } catch (FileNotFoundException e) {
        }

        try {
            bw.write(command);
        } catch (IOException e) {
        }
        try {
            bw.close();
        } catch (IOException e) {
        }
    }

    /**
     *
     */
    public static void upperCaseFile() {
        File rFol = new File("C:\\Dropbox\\TO-DO\\JAROW-master\\temp.txt");

        try (BufferedReader br = new BufferedReader(new FileReader(rFol))) {
            String s;
            while ((s = br.readLine()) != null) {
                //String txt = (" " + s + " ").replaceAll(" \\. ", "\\. ").replaceAll(" \\, ", "\\, ").replaceAll(" \\? ", "\\? ").replaceAll(" i ", " I ").trim();
                //System.out.println(txt.substring(0, 1).toUpperCase() + txt.substring(1));
                
                String txt = (" " + s + " ").replaceAll("\\. ", " \\. ").replaceAll("\\, ", " \\, ").replaceAll("\\? ", " \\? ").replaceAll("  ", " ").trim();
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     *
     */
    public static void parseBagelResults() {
        String resultFolder = "D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW-master\\results\\Bagel_8_16\\";
        String metric = "COVERAGE";
        File rFol = new File(resultFolder);

        HashMap<Integer, Integer> foldSizes = new HashMap<>();
        foldSizes.put(0, 21);
        foldSizes.put(1, 21);
        foldSizes.put(2, 20);
        foldSizes.put(3, 20);
        foldSizes.put(4, 20);
        foldSizes.put(5, 20);
        foldSizes.put(6, 20);
        foldSizes.put(7, 20);
        foldSizes.put(8, 20);
        foldSizes.put(9, 20);

        //Keys are values of learningRate, value is HashMap whose keys are values of earlyTerminationParam, final keys are folds and values are corresponding BLUE scores across epochs
        HashMap<Double, HashMap<String, HashMap<Integer, ArrayList<Double>>>> results = new HashMap();
        for (File rFile : rFol.listFiles()) {
            try (BufferedReader br = new BufferedReader(new FileReader(rFile))) {
                String fileNameParams = rFile.getName().substring(rFile.getName().indexOf("fold_") + 5, rFile.getName().indexOf(".txt"));
                int fold = Integer.parseInt(fileNameParams.substring(0, fileNameParams.indexOf('_')));
                String earlyTerminationParam = fileNameParams.substring(fileNameParams.indexOf('_') + 1, fileNameParams.lastIndexOf('_'));
                double learningRate = Double.parseDouble(fileNameParams.substring(fileNameParams.lastIndexOf('_') + 1));
                if (!results.containsKey(learningRate)) {
                    results.put(learningRate, new HashMap<String, HashMap<Integer, ArrayList<Double>>>());
                }
                if (!results.get(learningRate).containsKey(earlyTerminationParam)) {
                    results.get(learningRate).put(earlyTerminationParam, new HashMap<Integer, ArrayList<Double>>());
                }
                if (results.get(learningRate).get(earlyTerminationParam).containsKey(fold)) {
                    System.exit(0);
                }
                results.get(learningRate).get(earlyTerminationParam).put(fold, new ArrayList<Double>());
                String s;
                while ((s = br.readLine()) != null) {
                    if (s.startsWith(metric + ":")) {
                        results.get(learningRate).get(earlyTerminationParam).get(fold).add(Double.parseDouble(s.substring(metric.length() + 1).trim()));
                    }
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        //AVERAGE OVER FOLDS
        ArrayList<Double> learningRates = new ArrayList<Double>();
        //learningRates.add(0.0);
        //learningRates.add(0.1);
        learningRates.add(0.2);
        //learningRates.add(0.3);
        HashMap<Double, HashMap<String, ArrayList<Double>>> avgResults = new HashMap();
        learningRates.stream().map((learningRate) -> {
            if (!avgResults.containsKey(learningRate)) {
                avgResults.put(learningRate, new HashMap<String, ArrayList<Double>>());
            }
            return learningRate;
        }).forEachOrdered((learningRate) -> {
            results.get(learningRate).keySet().stream().map((earlyTerminationParam) -> {
                if (!avgResults.get(learningRate).containsKey(earlyTerminationParam)) {
                    avgResults.get(learningRate).put(earlyTerminationParam, new ArrayList<Double>());
                }
                return earlyTerminationParam;
            }).forEachOrdered((earlyTerminationParam) -> {
                for (int epoch = 0; epoch <= 6; epoch++) {
                    boolean allFoldsHaveResult = true;
                    double result = 0.0;
                    for (int fold = 0; fold <= 9; fold++) {
                        if (results.get(learningRate).get(earlyTerminationParam).get(fold).size() <= epoch) {
                            allFoldsHaveResult = false;
                        } else {
                            result += results.get(learningRate).get(earlyTerminationParam).get(fold).get(epoch) * foldSizes.get(fold);
                        }
                    }
                    if (allFoldsHaveResult) {
                        int total = 0;
                        total = foldSizes.values().stream().map((i) -> i).reduce(total, Integer::sum);
                        result /= total;
                        avgResults.get(learningRate).get(earlyTerminationParam).add(result);
                    } else {
                        epoch = 100_000;
                    }
                }
            });
        });

        //PRINT RESULTS
        BufferedWriter bw = null;
        File f = null;
        try {
            f = new File("BAGELparsedResults.txt");
        } catch (NullPointerException e) {
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
        } catch (FileNotFoundException e) {
        }

        try {
            for (Double learningRate : results.keySet()) {
                bw.write("beta = " + learningRate);
                bw.write("\n");
                bw.write(metric);
                bw.write("\n");
                boolean headerOnce = true;
                String under1 = "\t";
                String under2 = "\t";
                for (String earlyTerminationParam : results.get(learningRate).keySet()) {
                    String header = "\t";
                    under1 = "\t";
                    under2 = "\t";
                    String text = earlyTerminationParam + "\t";
                    for (int fold = 0; fold <= 9; fold++) {
                        header += "FOLD = " + fold;
                        for (int epoch = 0; epoch <= 6; epoch++) {
                            under1 += "p" + epoch + "\t";
                            switch (epoch) {
                                case 0:
                                    under2 += "Before LOLS" + "\t";
                                    break;
                                case 1:
                                    under2 += "After " + epoch + " epoch" + "\t";
                                    break;
                                default:
                                    under2 += "After " + epoch + " epochs" + "\t";
                                    break;
                            }
                            if (results.get(learningRate).get(earlyTerminationParam).get(fold).size() > epoch) {
                                text += results.get(learningRate).get(earlyTerminationParam).get(fold).get(epoch);
                            }
                            header += "\t";
                            text += "\t";
                        }
                        header += "\t";
                        under1 += "\t";
                        under2 += "\t";
                        text += "\t";
                    }
                    if (!headerOnce) {
                        bw.write(header);
                        bw.write("\n");
                        headerOnce = false;
                    }
                    bw.write(text);
                    bw.write("\n");
                }
                bw.write(under1);
                bw.write("\n");
                bw.write(under2);
                bw.write("\n");
                bw.write("\n");
            }
            bw.write("\n");
            bw.write("\n");
            bw.write("\n");
            bw.write("AVERAGE ACROSS FOLDS\n");
            bw.write("\n");
            for (Double learningRate : avgResults.keySet()) {
                bw.write("beta = " + learningRate);
                bw.write("\n");
                bw.write(metric);
                bw.write("\n");
                String under1 = "\t";
                String under2 = "\t";
                for (String earlyTerminationParam : avgResults.get(learningRate).keySet()) {
                    under1 = "\t";
                    under2 = "\t";
                    String text = earlyTerminationParam + "\t";
                    for (int epoch = 0; epoch <= 6; epoch++) {
                        under1 += "p" + epoch + "\t";
                        switch (epoch) {
                            case 0:
                                under2 += "Before LOLS" + "\t";
                                break;
                            case 1:
                                under2 += "After " + epoch + " epoch" + "\t";
                                break;
                            default:
                                under2 += "After " + epoch + " epochs" + "\t";
                                break;
                        }
                        if (avgResults.get(learningRate).get(earlyTerminationParam).size() > epoch) {
                            text += avgResults.get(learningRate).get(earlyTerminationParam).get(epoch);
                        }
                        text += "\t";
                    }
                    text += "\t";
                    under1 += "\t";
                    under2 += "\t";
                    bw.write(text);
                    bw.write("\n");
                }
                bw.write(under1);
                bw.write("\n");
                bw.write(under2);
                bw.write("\n");
                bw.write("\n");
            }
        } catch (IOException e) {
        }
        try {
            bw.close();
        } catch (IOException e) {
        }
    }

    /**
     *
     */
    public static void parseSFXHotelResults() {
        String resultFolder = "D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW-master\\results\\SFX_hotel_28_9_16_valid\\";
        File rFol = new File(resultFolder);

        //OVERALL KEYS ARE LOSS FUNCTION
        //Keys are values of learningRate, value is HashMap whose keys are values of earlyTerminationParam, values are corresponding BLUE scores across epochs
        String data = "HOTEL";
        for (int r = 0; r < 4; r++) {
            String metric = "BLEU";
            switch (r) {
                case 1:
                    metric = "ROUGE";
                    break;
                case 2:
                    metric = "COVERAGE ERROR";
                    break;
                case 3:
                    metric = "BRC";
                    break;
                default:
                    break;
            }

            HashMap<String, HashMap<Double, HashMap<String, ArrayList<Double>>>> results = new HashMap<>();
            HashMap<String, HashMap<Double, HashMap<String, ArrayList<Double>>>> resultsUNIQUE = new HashMap<>();
            HashMap<String, HashMap<String, HashMap<Double, HashMap<String, ArrayList<Double>>>>> resultsUNIQUEPerPredicate = new HashMap<>();
            for (File rFile : rFol.listFiles()) {
                try (BufferedReader br = new BufferedReader(new FileReader(rFile))) {
                    String fileNameParams = rFile.getName().substring(rFile.getName().indexOf('_') + 5, rFile.getName().indexOf(".txt"));
                    int place1 = fileNameParams.indexOf('_');
                    int place2 = fileNameParams.indexOf('_', place1 + 1);
                    int place3 = fileNameParams.lastIndexOf('_');
                    String earlyTerminationParam = fileNameParams.substring(place1 + 1, place2);
                    double learningRate = Double.parseDouble(fileNameParams.substring(place2 + 1, place3));
                    String lossFunction = fileNameParams.substring(place3 + 1);

                    if (!results.containsKey(lossFunction)) {
                        results.put(lossFunction, new HashMap<Double, HashMap<String, ArrayList<Double>>>());
                        resultsUNIQUE.put(lossFunction, new HashMap<Double, HashMap<String, ArrayList<Double>>>());
                    }
                    if (!results.get(lossFunction).containsKey(learningRate)) {
                        results.get(lossFunction).put(learningRate, new HashMap<String, ArrayList<Double>>());
                        resultsUNIQUE.get(lossFunction).put(learningRate, new HashMap<String, ArrayList<Double>>());
                    }
                    if (!results.get(lossFunction).get(learningRate).containsKey(earlyTerminationParam)) {
                        results.get(lossFunction).get(learningRate).put(earlyTerminationParam, new ArrayList<Double>());
                        resultsUNIQUE.get(lossFunction).get(learningRate).put(earlyTerminationParam, new ArrayList<Double>());
                    }
                    String s;
                    while ((s = br.readLine()) != null) {
                        if (s.startsWith(metric + ":")) {
                            results.get(lossFunction).get(learningRate).get(earlyTerminationParam).add(Double.parseDouble(s.substring((metric + ":").length()).trim()));
                        }
                        if (s.startsWith("UNIQUE WORD ALL PRED " + metric + ":")) {
                            resultsUNIQUE.get(lossFunction).get(learningRate).get(earlyTerminationParam).add(Double.parseDouble(s.substring(("UNIQUE WORD ALL PRED " + metric + ":").length()).trim()));
                        } else if (s.startsWith("UNIQUE WORD ") && s.contains(" " + metric + ":")) {
                            String predicate = s.substring("UNIQUE WORD ".length(), s.indexOf(" " + metric + ":"));

                            if (!resultsUNIQUEPerPredicate.containsKey(predicate)) {
                                resultsUNIQUEPerPredicate.put(predicate, new HashMap<String, HashMap<Double, HashMap<String, ArrayList<Double>>>>());
                            }
                            if (!resultsUNIQUEPerPredicate.get(predicate).containsKey(lossFunction)) {
                                resultsUNIQUEPerPredicate.get(predicate).put(lossFunction, new HashMap<Double, HashMap<String, ArrayList<Double>>>());
                            }
                            if (!resultsUNIQUEPerPredicate.get(predicate).get(lossFunction).containsKey(learningRate)) {
                                resultsUNIQUEPerPredicate.get(predicate).get(lossFunction).put(learningRate, new HashMap<String, ArrayList<Double>>());
                            }
                            if (!resultsUNIQUEPerPredicate.get(predicate).get(lossFunction).get(learningRate).containsKey(earlyTerminationParam)) {
                                resultsUNIQUEPerPredicate.get(predicate).get(lossFunction).get(learningRate).put(earlyTerminationParam, new ArrayList<Double>());
                            }
                            resultsUNIQUEPerPredicate.get(predicate).get(lossFunction).get(learningRate).get(earlyTerminationParam).add(Double.parseDouble(s.substring(s.indexOf(':') + 1).trim()));
                        }
                    }
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
                }
            }

            //PRINT RESULTS
            /*ArrayList<String> vectors = new ArrayList<>();
            for (String lossFunction : results.keySet()) {
            for (Double learningRate : results.get(lossFunction).keySet()) {
            vectors.add("ALL PREDICATES\t\t\t\t\t\t\t\t\t\t\t\t\t");
            vectors.add("LOSS = " + lossFunction + "\t" + "beta = " + learningRate + "\t\t\t\t\t\t\t\t\t\t\t\t");
            vectors.add("" + metric + "\t\t\t\t\t\t\t\t\t\t\t\t\t");
            boolean headerOnce = true;
            String under1 = "\t";
            String under2 = "\t";
            for (String earlyTerminationParam : results.get(lossFunction).get(learningRate).keySet()) {
            String header = "\t";
            under1 = "\t";
            under2 = "\t";
            String text = earlyTerminationParam + "\t";
            for (int epoch = 0; epoch <= 10; epoch++) {
            under1 += "p" + epoch + "\t";
            if (epoch == 0) {
            under2 += "Before LOLS" + "\t";
            } else if (epoch == 1) {
            under2 += "After " + epoch + " epoch" + "\t";
            } else {
            under2 += "After " + epoch + " epochs" + "\t";
            }
            if (results.get(lossFunction).get(learningRate).get(earlyTerminationParam).size() > epoch) {
            text += results.get(lossFunction).get(learningRate).get(earlyTerminationParam).get(epoch);
            }
            header += "\t";
            text += "\t";
            }
            header += "\t";
            under1 += "\t";
            under2 += "\t";
            text += "\t";
            vectors.add(text);
            }
            vectors.add(under1);
            vectors.add(under2);
            vectors.add("");
            }
            vectors.add("");
            vectors.add("");
            vectors.add("");
            }
            
            ArrayList<String> vectorsUnique = new ArrayList<>();
            for (String lossFunction : resultsUNIQUE.keySet()) {
            for (Double learningRate : resultsUNIQUE.get(lossFunction).keySet()) {
            vectorsUnique.add("ALL UNIQUE PREDICATES\t\t\t\t\t\t\t\t\t\t\t\t\t");
            vectorsUnique.add("LOSS = " + lossFunction + "\t" + "beta = " + learningRate + "\t\t\t\t\t\t\t\t\t\t\t\t");
            vectorsUnique.add("" + metric + "\t\t\t\t\t\t\t\t\t\t\t\t\t");
            boolean headerOnce = true;
            String under1 = "\t";
            String under2 = "\t";
            for (String earlyTerminationParam : resultsUNIQUE.get(lossFunction).get(learningRate).keySet()) {
            String header = "\t";
            under1 = "\t";
            under2 = "\t";
            String text = earlyTerminationParam + "\t";
            for (int epoch = 0; epoch <= 10; epoch++) {
            under1 += "p" + epoch + "\t";
            if (epoch == 0) {
            under2 += "Before LOLS" + "\t";
            } else if (epoch == 1) {
            under2 += "After " + epoch + " epoch" + "\t";
            } else {
            under2 += "After " + epoch + " epochs" + "\t";
            }
            if (resultsUNIQUE.get(lossFunction).get(learningRate).get(earlyTerminationParam).size() > epoch) {
            text += resultsUNIQUE.get(lossFunction).get(learningRate).get(earlyTerminationParam).get(epoch);
            }
            header += "\t";
            text += "\t";
            }
            header += "\t";
            under1 += "\t";
            under2 += "\t";
            text += "\t";
            vectorsUnique.add(text);
            }
            vectorsUnique.add(under1);
            vectorsUnique.add(under2);
            vectorsUnique.add("");
            }
            vectorsUnique.add("");
            vectorsUnique.add("");
            vectorsUnique.add("");
            }
            
            HashMap<String, ArrayList<String>> vectorsUniquePerPred = new HashMap<>();
            for (String predicate : resultsUNIQUEPerPredicate.keySet()) {
            vectorsUniquePerPred.put(predicate, new ArrayList<String>());
            for (String lossFunction : resultsUNIQUEPerPredicate.get(predicate).keySet()) {
            for (Double learningRate : resultsUNIQUEPerPredicate.get(predicate).get(lossFunction).keySet()) {
            vectorsUniquePerPred.get(predicate).add(predicate + "\t\t\t\t\t\t\t\t\t\t\t\t\t");
            vectorsUniquePerPred.get(predicate).add("LOSS = " + lossFunction + "\t" + "beta = " + learningRate + "\t\t\t\t\t\t\t\t\t\t\t\t");
            vectorsUniquePerPred.get(predicate).add("" + metric + "\t\t\t\t\t\t\t\t\t\t\t\t\t");
            boolean headerOnce = true;
            String under1 = "\t";
            String under2 = "\t";
            for (String earlyTerminationParam : resultsUNIQUEPerPredicate.get(predicate).get(lossFunction).get(learningRate).keySet()) {
            String header = "\t";
            under1 = "\t";
            under2 = "\t";
            String text = earlyTerminationParam + "\t";
            for (int epoch = 0; epoch <= 10; epoch++) {
            under1 += "p" + epoch + "\t";
            if (epoch == 0) {
            under2 += "Before LOLS" + "\t";
            } else if (epoch == 1) {
            under2 += "After " + epoch + " epoch" + "\t";
            } else {
            under2 += "After " + epoch + " epochs" + "\t";
            }
            if (resultsUNIQUEPerPredicate.get(predicate).get(lossFunction).get(learningRate).get(earlyTerminationParam).size() > epoch) {
            text += resultsUNIQUEPerPredicate.get(predicate).get(lossFunction).get(learningRate).get(earlyTerminationParam).get(epoch);
            }
            header += "\t";
            text += "\t";
            }
            header += "\t";
            under1 += "\t";
            under2 += "\t";
            text += "\t";
            vectorsUniquePerPred.get(predicate).add(text);
            }
            vectorsUniquePerPred.get(predicate).add(under1);
            vectorsUniquePerPred.get(predicate).add(under2);
            vectorsUniquePerPred.get(predicate).add("");
            }
            vectorsUniquePerPred.get(predicate).add("");
            vectorsUniquePerPred.get(predicate).add("");
            vectorsUniquePerPred.get(predicate).add("");
            }
            }
            
            ArrayList<String> finalVectors = new ArrayList<>();
            for (int i = 0; i < vectors.size(); i++) {
            String fin = vectors.get(i) + "\t\t\t\t\t" + vectorsUnique.get(i) + "\t\t\t\t\t";
            for (String predicate : vectorsUniquePerPred.keySet()) {
            fin += vectorsUniquePerPred.get(predicate).get(i) + "\t\t\t\t\t";
            }
            finalVectors.add(fin);
            }*/
            ArrayList<String> vectors = new ArrayList<>();
            for (String lossFunction : results.keySet()) {
                for (Double learningRate : results.get(lossFunction).keySet()) {
                    vectors.add("ALL PREDICATES\t\t\t\t\t\t\t\t\t\t\t\t\t");
                    vectors.add("LOSS = " + lossFunction + "\t" + "beta = " + learningRate + "\t\t\t\t\t\t\t\t\t\t\t\t");
                    vectors.add("" + metric + "\t\t\t\t\t\t\t\t\t\t\t\t\t");
                    boolean headerOnce = true;
                    String under1 = "\t";
                    String under2 = "\t";
                    for (String earlyTerminationParam : results.get(lossFunction).get(learningRate).keySet()) {
                        String header = "\t";
                        under1 = "\t";
                        under2 = "\t";
                        String text = earlyTerminationParam + "\t";
                        double bestEpoch = 0.0;
                        for (int epoch = 0; epoch <= 10; epoch++) {
                            under1 += "p" + epoch + "\t";
                            switch (epoch) {
                                case 0:
                                    under2 += "Before LOLS" + "\t";
                                    break;
                                case 1:
                                    under2 += "After " + epoch + " epoch" + "\t";
                                    break;
                                default:
                                    under2 += "After " + epoch + " epochs" + "\t";
                                    break;
                            }
                            if (results.get(lossFunction).get(learningRate).get(earlyTerminationParam).size() > epoch) {
                                if (results.get(lossFunction).get(learningRate).get(earlyTerminationParam).get(epoch) > bestEpoch) {
                                    bestEpoch = results.get(lossFunction).get(learningRate).get(earlyTerminationParam).get(epoch);
                                }
                            }
                        }
                        text += bestEpoch;
                        header += "\t";
                        text += "\t";

                        header += "\t";
                        under1 += "\t";
                        under2 += "\t";
                        text += "\t";
                        vectors.add(text);
                    }
                    //vectors.add(under1);
                    //vectors.add(under2);
                    vectors.add("");
                }
                vectors.add("");
                vectors.add("");
                vectors.add("");
            }

            writeVectors("SFX" + data + "parsedResults_" + metric + ".txt", vectors);
        }
    }

    /**
     *
     * @param fileName
     * @param vectors
     */
    public static void writeVectors(String fileName, List<String> vectors) {
        try {
            try (PrintWriter out = new PrintWriter(fileName)) {
                vectors.forEach((vector) -> {
                    out.println(vector);
                });
            }
        } catch (FileNotFoundException ex) {
        }
    }
    private static final Logger LOG = Logger.getLogger(ParseResultFiles.class.getName());
}
