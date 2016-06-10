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
import imitationNLG.Bagel;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author lampouras
 */
public class parseResultFiles {

    public static void main(String[] args) {
        //generateQSUBforBAGEL();
        //parseBagelResults();
        //generateQSUBforSFXHotel();
        //parseSFXHotelResults();
        //upperCaseFile();
        //parseWenFiles();
        //parseERR();
        parseERRBagel();
    }

    public static void parseWenFiles() {
        String mrFile = "results\\wenResults\\hotelMRs.txt";
        String rFile = "results\\wenResults\\sfxhotel.log";
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
        try (BufferedReader br = new BufferedReader(new FileReader(rFile))) {
            String s;
            boolean inGen = false;
            boolean inRef = false;
            ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
            ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
            ArrayList<Sequence<IString>> references = new ArrayList<>();
            String da = "";
            int gens = 0;
            ArrayList<Double> errs = new ArrayList<>();
            HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
            while ((s = br.readLine()) != null) {
                if (s.startsWith("DA")) {
                    inGen = false;
                    inRef = false;
                    da = s.substring(s.indexOf(":") + 1).trim();
                    
                    String MRstr = s.substring(s.indexOf(":") + 1).replaceAll(",", ";").replaceAll("no or yes", "yes or no").replaceAll("ave ; presidio", "ave and presidio").replaceAll("point ; ste", "point and ste").trim();
                    String attributesStr = MRstr.substring(MRstr.indexOf('(') + 1, MRstr.length() - 1);
                    attributeValues = new HashMap<>();
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
                                System.out.println("EMPTY VALUE");
                                System.exit(0);
                            }

                            attributeValues.get(attr).add(value.trim().toLowerCase());
                        }
                        for (String attr : attributeValues.keySet()) {
                            if (attributeValues.get(attr).contains("yes")
                                    && attributeValues.get(attr).contains("no")) {
                                System.out.println(MRstr);
                                System.out.println(attributeValues);
                                System.exit(0);
                            }
                        }
                        //System.out.println("\t" + da);
                        //System.out.println("\t" + attributeValues);
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
                        Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(s.trim().toLowerCase()));
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
                                        && (gen.contains(searchValue) || gen.contains("meal")  || gen.contains("breakfast") )) {
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
                                }  else if (gen.contains(searchValue)) {
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
                        double err = 1.0 - ((double)mentioned/(double)total);
                        if (Double.isNaN(err)) {
                            err = 0.0;
                        }
                        errs.add(err);
                        if (!errors.isEmpty()) {                                
                            System.out.println("================");
                            System.out.println(da);
                            System.out.println(s.trim());
                            for (String error : errors) {
                                System.out.println(error);
                            }
                            System.out.println("ERR: \t" + err);
                        }
                        da = "";
                    }
                } else if (inRef) {
                    if (s.trim().isEmpty()) {
                        for (int i = 0; i < gens; i++) {
                            finalReferences.add(references);
                        }
                        references = new ArrayList<>();
                        inRef = false;
                        da = "";
                    } else {
                        String cleanedWords = s.trim().replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                        references.add(IStrings.tokenize(NISTTokenizer.tokenize(cleanedWords)));
                    }
                }
            }
            double avgErr = 0.0;
            for (double err : errs) {
                avgErr += err;
            }
            avgErr = avgErr/(double)errs.size() * 100;
            System.out.println(finalReferences.size() + "\t" + generations.size() + "\t" + errs.size());
            BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
            NISTMetric NIST = new NISTMetric(finalReferences);
            System.out.println("BLEU: \t" + BLEU.score(generations));
            System.out.println("NIST: \t" + NIST.score(generations));
            System.out.println("ERR: \t" + avgErr);
            System.out.println(daToGen);
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
    }
    
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
                da = arr[0].substring(arr[0].indexOf(":") + 1).toLowerCase().trim();
                String gen = " " + arr[1].trim().toLowerCase() + " ";

                String MRstr = arr[0].substring(arr[0].indexOf(":") + 1).replaceAll(",", ";").replaceAll("no or yes", "yes or no").replaceAll("ave ; presidio", "ave and presidio").replaceAll("point ; ste", "point and ste").trim();
                String attributesStr = MRstr.substring(MRstr.indexOf('(') + 1, MRstr.length() - 1);
                attributeValues = new HashMap<>();
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
                            System.out.println("EMPTY VALUE");
                            System.exit(0);
                        }

                        attributeValues.get(attr).add(value.trim().toLowerCase());
                    }
                    for (String attr : attributeValues.keySet()) {
                        if (attributeValues.get(attr).contains("yes")
                                && attributeValues.get(attr).contains("no")) {
                            System.out.println(MRstr);
                            System.out.println(attributeValues);
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
                                    && (gen.contains(searchValue) || gen.contains("meal")  || gen.contains("breakfast") )) {
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
                            }  else if (gen.contains(searchValue)) {
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
                    double err = 1.0 - ((double)mentioned/(double)total);
                    if (Double.isNaN(err)) {
                        err = 0.0;
                    }
                    errs.add(err);
                    if (!errors.isEmpty()) {                                
                        System.out.println("================");
                        System.out.println(da);
                        System.out.println(s.trim());
                        for (String error : errors) {
                            System.out.println(error);
                        }
                        System.out.println("ERR: \t" + err);
                    }
                } else {
                    errs.add(0.0);
                }
    
            }
            double avgErr = 0.0;
            for (double err : errs) {
                avgErr += err;
            }
            avgErr = avgErr/(double)errs.size() * 100;
            System.out.println("\t" + errs.size());
            System.out.println("ERR: \t" + avgErr);
            System.out.println(daToGen);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
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
                da = arr[0].substring(arr[0].indexOf(":") + 1).toLowerCase().trim();
                String gen = " " + arr[1].trim().toLowerCase() + " ";

                String MRstr = new String(da.substring(da.indexOf("(") + 1, da.lastIndexOf(")")));

                HashMap<String, String> names = new HashMap<>();
                int s1 = MRstr.indexOf("\"");
                int a = 0;
                while (s1 != -1) {
                    int e = MRstr.indexOf("\"", s1 + 1);

                    String name = MRstr.substring(s1, e + 1);
                    MRstr = MRstr.replace(name, "x" + a);
                    names.put("x" + a, name);
                    a++;

                    s1 = MRstr.indexOf("\"");
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
                                && (gen.contains(searchValue) || gen.contains("meal")  || gen.contains("breakfast") )) {
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
                                && (gen.contains(searchValue) || gen.contains("fast")  || gen.contains("food") )) {
                            ment = true;
                        } else if (searchValue.equals("area")
                                && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                            ment = true;
                        } else if (searchValue.equals("near")
                                && (gen.contains(searchValue) || gen.contains("location") || gen.contains("part") || gen.contains("where"))) {
                            ment = true;
                        }  else if (gen.contains(searchValue)) {
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
                double err = 1.0 - ((double)mentioned/(double)total);
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
                        for (String error : errors) {
                            System.out.println(error);
                        }
                        System.out.println("ERR: \t" + err);
                    }
                }
            }
            double avgErr = 0.0;
            for (double err : errs) {
                avgErr += err;
            }
            avgErr = avgErr/(double)errs.size() * 100;
            System.out.println("\t" + errs.size());
            System.out.println("ERR: \t" + avgErr);
            System.out.println(daToGen);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void generateQSUBforSFXHotel() {
        String command = "";

        String data = "HOTEL";
        String folder = "D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW\\qsubs\\SFXHotel\\";
        //PRINT RESULTS
        ArrayList<Double> learningRates = new ArrayList<Double>();
        learningRates.add(0.0);
        learningRates.add(0.1);
        learningRates.add(0.2);
        learningRates.add(0.3);
        ArrayList<String> earlyTerminationParams = new ArrayList<String>();
        earlyTerminationParams.add("0");
        earlyTerminationParams.add("1");
        earlyTerminationParams.add("2");
        earlyTerminationParams.add("3");
        earlyTerminationParams.add("4");
        earlyTerminationParams.add("inf");
        for (Double learningRate : learningRates) {
            for (String earlyTerminationParam : earlyTerminationParams) {
                BufferedWriter bw = null;
                File f = null;
                try {
                    f = new File(folder + "run_SFX_" + data + "_" + earlyTerminationParam + "_" + learningRate + ".sh");
                    command += "qsub " + "run_SFX_" + data + "_" + earlyTerminationParam + "_" + learningRate + ".sh;";
                } catch (NullPointerException e) {
                    System.err.println("File not found." + e);
                }

                try {
                    bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
                } catch (FileNotFoundException e) {
                    System.err.println("Error opening file for writing! " + e);
                }

                try {
                    bw.write("#!/bin/bash" + "\n");
                    bw.write("#$ -l mem=32g -l rmem=32g -m bea -M g.lampouras@sheffield.ac.uk -l h_rt=8:00:00" + "\n");
                    bw.write("module load apps/java/1.8u71" + "\n");
                    bw.write("\n");
                    if (earlyTerminationParam.equals("inf")) {
                        bw.write("java -Xmx6g -jar JDagger-" + data + ".jar " + 100 + " " + learningRate + " -classpath /lib > results_SFX_" + data + "_" + earlyTerminationParam + "_" + learningRate + ".txt" + "\n");
                    } else {
                        bw.write("java -Xmx6g -jar JDagger-" + data + ".jar " + earlyTerminationParam + " " + learningRate + " -classpath /lib > results_SFX_" + data + "_" + earlyTerminationParam + "_" + learningRate + ".txt" + "\n");
                    }
                } catch (IOException e) {
                    System.err.println("Write error!");
                }
                try {
                    bw.close();
                } catch (IOException e) {
                    System.err.println("Error closing file.");
                } catch (Exception e) {
                }
            }
        }

        //Write command file
        BufferedWriter bw = null;
        File f = null;
        try {
            f = new File(folder + "command" + data + ".txt");
        } catch (NullPointerException e) {
            System.err.println("File not found." + e);
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
        } catch (FileNotFoundException e) {
            System.err.println("Error opening file for writing! " + e);
        }

        try {
            bw.write(command);
        } catch (IOException e) {
            System.err.println("Write error!");
        }
        try {
            bw.close();
        } catch (IOException e) {
            System.err.println("Error closing file.");
        } catch (Exception e) {
        }
    }

    public static void generateQSUBforBAGEL() {
        String command = "";

        String folder = "D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW\\qsubs\\BAGEL\\";
        //PRINT RESULTS
        ArrayList<Double> learningRates = new ArrayList<Double>();
        learningRates.add(0.0);
        learningRates.add(0.1);
        learningRates.add(0.2);
        learningRates.add(0.3);
        ArrayList<String> earlyTerminationParams = new ArrayList<String>();
        earlyTerminationParams.add("0");
        earlyTerminationParams.add("1");
        earlyTerminationParams.add("2");
        earlyTerminationParams.add("3");
        earlyTerminationParams.add("4");
        earlyTerminationParams.add("inf");
        for (Double learningRate : learningRates) {
            for (String earlyTerminationParam : earlyTerminationParams) {
                for (int fold = 0; fold <= 9; fold++) {
                    BufferedWriter bw = null;
                    File f = null;
                    try {
                        f = new File(folder + "run_" + fold + "_" + earlyTerminationParam + "_" + learningRate + ".sh");
                        command += "qsub " + "run_" + fold + "_" + earlyTerminationParam + "_" + learningRate + ".sh;";
                    } catch (NullPointerException e) {
                        System.err.println("File not found." + e);
                    }

                    try {
                        bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
                    } catch (FileNotFoundException e) {
                        System.err.println("Error opening file for writing! " + e);
                    }

                    try {
                        bw.write("#!/bin/bash" + "\n");
                        bw.write("#$ -l mem=40g -l rmem=40g -m bea -M g.lampouras@sheffield.ac.uk -l h_rt=4:00:00" + "\n");
                        bw.write("module load apps/java/1.8u71" + "\n");
                        bw.write("\n");
                        if (earlyTerminationParam.equals("inf")) {
                            bw.write("java -Xmx6g -jar JDagger-BAGEL.jar " + fold + " " + 100 + " " + learningRate + " -classpath /lib > results_BAGEL_fold_" + fold + "_" + earlyTerminationParam + "_" + learningRate + ".txt" + "\n");
                        } else {
                            bw.write("java -Xmx6g -jar JDagger-BAGEL.jar " + fold + " " + earlyTerminationParam + " " + learningRate + " -classpath /lib > results_BAGEL_fold_" + fold + "_" + earlyTerminationParam + "_" + learningRate + ".txt" + "\n");
                        }
                    } catch (IOException e) {
                        System.err.println("Write error!");
                    }
                    try {
                        bw.close();
                    } catch (IOException e) {
                        System.err.println("Error closing file.");
                    } catch (Exception e) {
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
            System.err.println("File not found." + e);
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
        } catch (FileNotFoundException e) {
            System.err.println("Error opening file for writing! " + e);
        }

        try {
            bw.write(command);
        } catch (IOException e) {
            System.err.println("Write error!");
        }
        try {
            bw.close();
        } catch (IOException e) {
            System.err.println("Error closing file.");
        } catch (Exception e) {
        }
    }

    public static void upperCaseFile() {
        File rFol = new File("D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW\\temp.txt");

        try (BufferedReader br = new BufferedReader(new FileReader(rFol))) {
            String s;
            while ((s = br.readLine()) != null) {
                System.out.println(s.substring(0, 1).toUpperCase() + s.substring(1));
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void parseBagelResults() {
        String resultFolder = "C:\\Users\\lampouras\\Dropbox\\TO-DO\\JAROW\\results\\Bagel\\";
        File rFol = new File(resultFolder);

        //Keys are values of learningRate, value is HashMap whose keys are values of earlyTerminationParam, final keys are folds and values are corresponding BLUE scores across epochs
        HashMap<Double, HashMap<String, HashMap<Integer, ArrayList<Double>>>> results = new HashMap();
        for (File rFile : rFol.listFiles()) {
            try (BufferedReader br = new BufferedReader(new FileReader(rFile))) {
                String fileNameParams = rFile.getName().substring(rFile.getName().indexOf("fold_") + 5, rFile.getName().indexOf(".txt"));
                int fold = Integer.parseInt(fileNameParams.substring(0, fileNameParams.indexOf("_")));
                String earlyTerminationParam = fileNameParams.substring(fileNameParams.indexOf("_") + 1, fileNameParams.lastIndexOf("_"));
                double learningRate = Double.parseDouble(fileNameParams.substring(fileNameParams.lastIndexOf("_") + 1));
                if (!results.containsKey(learningRate)) {
                    results.put(learningRate, new HashMap<String, HashMap<Integer, ArrayList<Double>>>());
                }
                if (!results.get(learningRate).containsKey(earlyTerminationParam)) {
                    results.get(learningRate).put(earlyTerminationParam, new HashMap<Integer, ArrayList<Double>>());
                }
                if (results.get(learningRate).get(earlyTerminationParam).containsKey(fold)) {
                    System.out.println(rFile.getName() + ": Fold " + fold + " for earlyTerminationParam = " + earlyTerminationParam + " and learningRate = " + learningRate + " has already been parsed!");
                    System.exit(0);
                }
                results.get(learningRate).get(earlyTerminationParam).put(fold, new ArrayList<Double>());
                String s;
                while ((s = br.readLine()) != null) {
                    if (s.startsWith("BLEU:")) {
                        results.get(learningRate).get(earlyTerminationParam).get(fold).add(Double.parseDouble(s.substring(5).trim()));
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
        learningRates.add(0.0);
        learningRates.add(0.1);
        learningRates.add(0.2);
        learningRates.add(0.3);
        HashMap<Double, HashMap<String, ArrayList<Double>>> avgResults = new HashMap();
        for (Double learningRate : learningRates) {
            if (!avgResults.containsKey(learningRate)) {
                avgResults.put(learningRate, new HashMap<String, ArrayList<Double>>());
            }
            for (String earlyTerminationParam : results.get(learningRate).keySet()) {
                if (!avgResults.get(learningRate).containsKey(earlyTerminationParam)) {
                    avgResults.get(learningRate).put(earlyTerminationParam, new ArrayList<Double>());
                }
                for (int epoch = 0; epoch <= 6; epoch++) {
                    boolean allFoldsHaveResult = true;
                    double result = 0.0;
                    for (int fold = 0; fold <= 9; fold++) {
                        if (results.get(learningRate).get(earlyTerminationParam).get(fold).size() <= epoch) {
                            allFoldsHaveResult = false;
                        } else {
                            result += results.get(learningRate).get(earlyTerminationParam).get(fold).get(epoch);
                        }
                    }
                    if (allFoldsHaveResult) {
                        result /= 10.0;
                        avgResults.get(learningRate).get(earlyTerminationParam).add(result);
                    } else {
                        epoch = 100000;
                    }
                }
            }
        }

        //PRINT RESULTS
        BufferedWriter bw = null;
        File f = null;
        try {
            f = new File("BAGELparsedResults.txt");
        } catch (NullPointerException e) {
            System.err.println("File not found." + e);
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
        } catch (FileNotFoundException e) {
            System.err.println("Error opening file for writing! " + e);
        }

        try {
            for (Double learningRate : results.keySet()) {
                bw.write("beta = " + learningRate);
                bw.write("\n");
                bw.write("BLEU");
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
                            if (epoch == 0) {
                                under2 += "Before LOLS" + "\t";
                            } else if (epoch == 1) {
                                under2 += "After " + epoch + " epoch" + "\t";
                            } else {
                                under2 += "After " + epoch + " epochs" + "\t";
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
                bw.write("BLEU");
                bw.write("\n");
                String under1 = "\t";
                String under2 = "\t";
                for (String earlyTerminationParam : avgResults.get(learningRate).keySet()) {
                    under1 = "\t";
                    under2 = "\t";
                    String text = earlyTerminationParam + "\t";
                    for (int epoch = 0; epoch <= 6; epoch++) {
                        under1 += "p" + epoch + "\t";
                        if (epoch == 0) {
                            under2 += "Before LOLS" + "\t";
                        } else if (epoch == 1) {
                            under2 += "After " + epoch + " epoch" + "\t";
                        } else {
                            under2 += "After " + epoch + " epochs" + "\t";
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
            System.err.println("Write error!");
        }
        try {
            bw.close();
        } catch (IOException e) {
            System.err.println("Error closing file.");
        } catch (Exception e) {
        }
    }

    public static void parseSFXHotelResults() {
        String resultFolder = "D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW\\results\\SFX_hotel_constant_beta\\";
        File rFol = new File(resultFolder);

        //Keys are values of learningRate, value is HashMap whose keys are values of earlyTerminationParam, final keys are folds and values are corresponding BLUE scores across epochs
        String data = "HOTEL";
        HashMap<Double, HashMap<String, ArrayList<Double>>> results = new HashMap();
        for (File rFile : rFol.listFiles()) {
            try (BufferedReader br = new BufferedReader(new FileReader(rFile))) {
                String fileNameParams = rFile.getName().substring(rFile.getName().indexOf("_") + 5, rFile.getName().indexOf(".txt"));
                String earlyTerminationParam = fileNameParams.substring(fileNameParams.indexOf("_") + 1, fileNameParams.lastIndexOf("_"));
                double learningRate = Double.parseDouble(fileNameParams.substring(fileNameParams.lastIndexOf("_") + 1));
                if (!results.containsKey(learningRate)) {
                    results.put(learningRate, new HashMap<String, ArrayList<Double>>());
                }
                if (!results.get(learningRate).containsKey(earlyTerminationParam)) {
                    results.get(learningRate).put(earlyTerminationParam, new ArrayList<Double>());
                }
                String s;
                while ((s = br.readLine()) != null) {
                    if (s.startsWith("BLEU:")) {
                        results.get(learningRate).get(earlyTerminationParam).add(Double.parseDouble(s.substring(5).trim()));
                    }
                }
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Bagel.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        //PRINT RESULTS
        BufferedWriter bw = null;
        File f = null;
        try {
            f = new File("SFX" + data + "parsedResults.txt");
        } catch (NullPointerException e) {
            System.err.println("File not found." + e);
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
        } catch (FileNotFoundException e) {
            System.err.println("Error opening file for writing! " + e);
        }

        try {
            for (Double learningRate : results.keySet()) {
                bw.write("beta = " + learningRate);
                bw.write("\n");
                bw.write("BLEU");
                bw.write("\n");
                boolean headerOnce = true;
                String under1 = "\t";
                String under2 = "\t";
                for (String earlyTerminationParam : results.get(learningRate).keySet()) {
                    String header = "\t";
                    under1 = "\t";
                    under2 = "\t";
                    String text = earlyTerminationParam + "\t";
                    for (int epoch = 0; epoch <= 6; epoch++) {
                        under1 += "p" + epoch + "\t";
                        if (epoch == 0) {
                            under2 += "Before LOLS" + "\t";
                        } else if (epoch == 1) {
                            under2 += "After " + epoch + " epoch" + "\t";
                        } else {
                            under2 += "After " + epoch + " epochs" + "\t";
                        }
                        if (results.get(learningRate).get(earlyTerminationParam).size() > epoch) {
                            text += results.get(learningRate).get(earlyTerminationParam).get(epoch);
                        }
                        header += "\t";
                        text += "\t";
                    }
                    header += "\t";
                    under1 += "\t";
                    under2 += "\t";
                    text += "\t";
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
        } catch (IOException e) {
            System.err.println("Write error!");
        }
        try {
            bw.close();
        } catch (IOException e) {
            System.err.println("Error closing file.");
        } catch (Exception e) {
        }
    }
}
