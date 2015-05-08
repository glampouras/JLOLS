package jarow;

import edu.stanford.nlp.mt.metrics.BLEUMetric;
import edu.stanford.nlp.mt.metrics.NISTMetric;
import edu.stanford.nlp.mt.util.Sequence;
import edu.stanford.nlp.mt.util.SimpleSequence;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

public class RoboCup {

    static ArrayList<String> dictionary = new ArrayList<>();
    static ArrayList<String> argDictionary = new ArrayList<>();
    static HashMap<String, HashMap<String, Integer>> argDictionaryMap = new HashMap<>();
    static ArrayList<String> arguments = new ArrayList<>();
    static ArrayList<String> predicates = new ArrayList<>();

    final public static String TOKEN_END = "@end@";
    final public static String TOKEN_ARG1 = "@arg1@";
    final public static String TOKEN_ARG2 = "@arg2@";

    public static void main(String[] args) {
        File dataFolder = new File("robocup_data\\gold\\");
        /*createLists(dataFolder, -1);
         saveLists("robocup_data\\");
         //readLists("robocup_data\\");        
         createTrainingDatasets(new File("robocup_data\\gold\\"), "robocup_data\\goldTrainingData", -1);
                
         nlgTest("robocup_data\\");*/

        if (dataFolder.isDirectory()) {
            for (int f = 0; f < dataFolder.listFiles().length; f++) {
                File file = dataFolder.listFiles()[f];
                createLists(dataFolder, f);
                createTrainingDatasets(new File("robocup_data\\gold\\"), "robocup_data\\goldTrainingData", f);

                genTest("robocup_data\\", file, f);
            }
        }
    }

    public static void genTest(String modelPath, File testFile, int excludeFile) {
        String line;
        ArrayList<Instance> wordInstances = new ArrayList<>();
        ArrayList<Instance> argInstances = new ArrayList<>();

        try (
                InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_words_pass_excl" + excludeFile);
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader br = new BufferedReader(isr);) {
            System.out.println("Reading the word data");
            while ((line = br.readLine()) != null) {
                String[] details;
                details = line.split(" ");

                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                for (String word : dictionary) {
                    costs.put(word, 1.0);
                }
                costs.put(dictionary.get(Integer.parseInt(details[0])), 0.0);

                for (int i = 1; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");

                    featureVector.put(feature[0], Double.parseDouble(feature[1]));
                }
                wordInstances.add(new Instance(featureVector, costs));
                //System.out.println(instances.get(instances.size() - 1).getCosts());
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        /*try (
         InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_args_pass_excl" + excludeFile);
         InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
         BufferedReader br = new BufferedReader(isr);) {            
         System.out.println("Reading the arg data");
         while ((line = br.readLine()) != null) {
         String[] details;
         details = line.split(" ");
                
         TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
         TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
                
         for (String word : argDictionary) {                    
         costs.put(word, 1.0);
         }
         costs.put(argDictionary.get(Integer.parseInt(details[0])), 0.0);
                
         for (int i = 1; i < details.length; i++) {
         String[] feature;
         feature = details[i].split(":");
                    
         featureVector.put(feature[0], Double.parseDouble(feature[1]));
         }
         argInstances.add(new Instance(featureVector, costs));
         //System.out.println(instances.get(instances.size() - 1).getCosts());
         }
         } catch (FileNotFoundException ex) {
         Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
         } catch (IOException ex) {
         Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
         }*/

        Collections.shuffle(wordInstances);
        //Collections.shuffle(argInstances);

        evaluateGeneration(wordInstances, /*argInstances, */ testFile, 10.0);
    }

    public static void nlgTest(String modelPath) {
        String line;
        ArrayList<Instance> wordInstances = new ArrayList<>();
        ArrayList<Instance> argInstances = new ArrayList<>();

        try (
                InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_words_pass");
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader br = new BufferedReader(isr);) {
            System.out.println("Reading the word data");
            while ((line = br.readLine()) != null) {
                String[] details;
                details = line.split(" ");

                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                for (String word : dictionary) {
                    costs.put(word, 1.0);
                }
                costs.put(dictionary.get(Integer.parseInt(details[0])), 0.0);

                for (int i = 1; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");

                    featureVector.put(feature[0], Double.parseDouble(feature[1]));
                }
                wordInstances.add(new Instance(featureVector, costs));
                //System.out.println(instances.get(instances.size() - 1).getCosts());
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (
                InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_args_pass");
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader br = new BufferedReader(isr);) {
            System.out.println("Reading the arg data");
            while ((line = br.readLine()) != null) {
                String[] details;
                details = line.split(" ");

                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                for (String word : argDictionary) {
                    costs.put(word, 1.0);
                }
                costs.put(argDictionary.get(Integer.parseInt(details[0])), 0.0);

                for (int i = 1; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");

                    featureVector.put(feature[0], Double.parseDouble(feature[1]));
                }
                argInstances.add(new Instance(featureVector, costs));
                //System.out.println(instances.get(instances.size() - 1).getCosts());
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }

        Collections.shuffle(wordInstances);
        Collections.shuffle(argInstances);

        getGradedTrainingErrorRate(wordInstances, 10.0);
        //getCrossValidatedGradedErrorRate(wordInstances, 10.0);
    }

    public static void generalTest() {
        String line;
        ArrayList<Instance> instances = new ArrayList<>();
        try (
                InputStream fis = new FileInputStream("robocup_data\\goldTrainingData_pass");
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader br = new BufferedReader(isr);) {
            System.out.println("Reading the data");
            while ((line = br.readLine()) != null) {
                String[] details;
                details = line.split(" ");

                TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                for (String word : dictionary) {
                    costs.put(word, 0.0);
                }
                costs.put(dictionary.get(Integer.parseInt(details[0])), 1.0);

                for (int i = 1; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");

                    featureVector.put(feature[0], Double.parseDouble(feature[1]));
                }
                instances.add(new Instance(featureVector, costs));
                //System.out.println(instances.get(instances.size()).getCosts());
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }

        Collections.shuffle(instances);

        //instances = instances[:100]
        // ORIGINAL EVALUATION
        // Keep some instances to check the performance        
        /*ArrayList<Instance> testingInstances = new ArrayList(instances.subList(((int) Math.round(instances.size() * 0.75)) + 1, instances.size()));
         ArrayList<Instance> trainingInstances = new ArrayList(instances.subList(0, (int) Math.round(instances.size() * 0.75)));

         System.out.println("training data: " + trainingInstances.size() + " instances");
         //classifier_p.train(trainingInstances, True, True, 10, 10)

         // the last parameter can be set to True if probabilities are needed.
         Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
         JAROW classifier_p = JAROW.trainOpt(trainingInstances, 10, params, 0.1, true, false);

         Double cost = classifier_p.batchPredict(testingInstances);
         Double avgCost = cost/(double)testingInstances.size();
         System.out.println("Avg Cost per instance " + avgCost + " on " + testingInstances.size() + " testing instances");*/
        //10-FOLD CROSS VALIDATION
        for (double f = 0.0; f < 1.0; f += 0.1) {
            int from = ((int) Math.round(instances.size() * f)) + 1;
            if (from < instances.size()) {
                int to = (int) Math.round(instances.size() * (f + 0.1));
                if (to > instances.size()) {
                    to = instances.size();
                }
                ArrayList<Instance> testingInstances = new ArrayList(instances.subList(from, to));
                ArrayList<Instance> trainingInstances = new ArrayList(instances);
                for (Instance testInstance : testingInstances) {
                    trainingInstances.remove(testInstance);
                }

                System.out.println("training data: " + trainingInstances.size() + " instances");
                //classifier_p.train(trainingInstances, True, True, 10, 10)

                // the last parameter can be set to True if probabilities are needed.
                Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
                JAROW classifier_p = JAROW.trainOpt(trainingInstances, 10, params, 0.1, true, false);

                System.out.println("test data: " + testingInstances.size() + " instances");
                Double cost = classifier_p.batchPredict(testingInstances);
                Double avgCost = cost / (double) testingInstances.size();
                System.out.println("Avg Cost per instance " + avgCost + " on " + testingInstances.size() + " testing instances");
            }
        }
    }

    public static void createLists(File dataFolder, int excludeFileID) {
        dictionary = new ArrayList<>();
        argDictionary = new ArrayList<>();
        argDictionaryMap = new HashMap<>();
        arguments = new ArrayList<>();
        predicates = new ArrayList<>();

        dictionary.add(RoboCup.TOKEN_END);
        dictionary.add(RoboCup.TOKEN_ARG1);
        dictionary.add(RoboCup.TOKEN_ARG2);
        if (dataFolder.isDirectory()) {
            for (int f = 0; f < dataFolder.listFiles().length; f++) {
                if (f != excludeFileID) {
                    File file = dataFolder.listFiles()[f];
                    if (file.isFile()) {
                        Document dom = null;
                        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
                        try {
                            DocumentBuilder db = dbf.newDocumentBuilder();
                            dom = db.parse(file);
                        } catch (ParserConfigurationException pce) {
                            pce.printStackTrace();
                        } catch (SAXException se) {
                            se.printStackTrace();
                        } catch (IOException ioe) {
                            ioe.printStackTrace();
                        }
                        if (dom != null) {
                            Element docEle = dom.getDocumentElement();
                            NodeList nodeList = docEle.getElementsByTagName("example");
                            if (nodeList != null && nodeList.getLength() > 0) {
                                for (int i = 0; i < nodeList.getLength(); i++) {
                                    Element node = (Element) nodeList.item(i);

                                    NodeList nl = node.getElementsByTagName("nl");
                                    if (nl != null && nl.getLength() > 0) {
                                        String[] nlWords = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().replaceAll("\\'", " \\'").trim().toLowerCase().split(" ");

                                        boolean isFirst = true;
                                        for (int w = 0; w < nlWords.length; w++) {
                                            if (nlWords[w].startsWith("pink") || nlWords[w].startsWith("purple")) {
                                                argDictionary.add(nlWords[w].trim());
                                                if (isFirst) {
                                                    nlWords[w] = RoboCup.TOKEN_ARG1;
                                                    isFirst = false;
                                                } else {
                                                    nlWords[w] = RoboCup.TOKEN_ARG2;
                                                }
                                            }
                                        }

                                        for (String word : nlWords) {
                                            if (!word.trim().isEmpty() && !dictionary.contains(word.trim())) {
                                                dictionary.add(word.trim());
                                            }
                                        }
                                    }
                                    NodeList mrl = node.getElementsByTagName("mrl");
                                    if (mrl != null && mrl.getLength() > 0) {
                                        String mrNode = ((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase();
                                        mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ");
                                        String[] words = mrNode.split(" ");

                                        String predicate = null;
                                        String arg1 = null;
                                        String arg2 = null;
                                        for (String word : words) {
                                            if (!word.trim().isEmpty()) {
                                                if (predicate == null) {
                                                    predicate = word.trim();
                                                } else if (arg1 == null) {
                                                    arg1 = word.trim();
                                                } else if (arg2 == null) {
                                                    arg2 = word.trim();
                                                }
                                            }
                                        }
                                        if (!predicates.contains(predicate) && predicate != null) {
                                            predicates.add(predicate);
                                        }
                                        if (!arguments.contains(arg1) && arg1 != null) {
                                            arguments.add(arg1);
                                        }
                                        if (!argDictionaryMap.containsKey(arg1) && arg1 != null) {
                                            argDictionaryMap.put(arg1, new HashMap<String, Integer>());
                                        }
                                        if (!arguments.contains(arg2) && arg2 != null) {
                                            arguments.add(arg2);
                                        }
                                        if (!argDictionaryMap.containsKey(arg2) && arg2 != null) {
                                            argDictionaryMap.put(arg2, new HashMap<String, Integer>());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public static void saveLists(String writeFolderPath) {
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_dictionary"), "utf-8"))) {
            for (int i = 0; i < dictionary.size(); i++) {
                writer.write(i + ":" + dictionary.get(i) + "\n");
            }
            writer.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_arguments"), "utf-8"))) {
            for (int i = 0; i < arguments.size(); i++) {
                writer.write(i + ":" + arguments.get(i) + "\n");
            }
            writer.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(writeFolderPath + "_predicates"), "utf-8"))) {
            for (int i = 0; i < predicates.size(); i++) {
                writer.write(i + ":" + predicates.get(i) + "\n");
            }
            writer.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void readLists(String readFolderPath) {
        String line;
        try (
                InputStream fis = new FileInputStream(readFolderPath + "_dictionary");
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader reader = new BufferedReader(isr);) {

            ArrayList<String> lines = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line.trim());
                }
            }

            dictionary = new ArrayList();
            for (int i = 0; i <= Integer.parseInt(lines.get(lines.size() - 1).split(":")[0]); i++) {
                dictionary.add("");
            }

            for (String l : lines) {
                String[] details;
                details = l.split(":");

                dictionary.set(Integer.parseInt(details[0]), details[1]);
            }
            reader.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (
                InputStream fis = new FileInputStream(readFolderPath + "_arguments");
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader reader = new BufferedReader(isr);) {

            ArrayList<String> lines = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line.trim());
                }
            }

            arguments = new ArrayList();
            for (int i = 0; i <= Integer.parseInt(lines.get(lines.size() - 1).split(":")[0]); i++) {
                arguments.add("");
            }

            for (String l : lines) {
                String[] details;
                details = l.split(":");

                arguments.set(Integer.parseInt(details[0]), details[1]);
            }
            reader.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (
                InputStream fis = new FileInputStream(readFolderPath + "_predicates");
                InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
                BufferedReader reader = new BufferedReader(isr);) {

            ArrayList<String> lines = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line.trim());
                }
            }

            predicates = new ArrayList();
            for (int i = 0; i <= Integer.parseInt(lines.get(lines.size() - 1).split(":")[0]); i++) {
                predicates.add("");
            }

            for (String l : lines) {
                String[] details;
                details = l.split(":");

                predicates.set(Integer.parseInt(details[0]), details[1]);
            }
            reader.close();
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void createTrainingDatasets(File dataFolder, String trainingFilePath, int excludeFileID) {
        if (!dictionary.isEmpty() && !predicates.isEmpty() && !arguments.isEmpty()) {
            HashMap<String, ArrayList<String>> predicateWordTrainingData = new HashMap<>();
            HashMap<String, ArrayList<String>> predicateArgTrainingData = new HashMap<>();
            for (String predicate : predicates) {
                predicateWordTrainingData.put(predicate, new ArrayList<String>());
                predicateArgTrainingData.put(predicate, new ArrayList<String>());
            }
            if (dataFolder.isDirectory()) {
                for (int f = 0; f < dataFolder.listFiles().length; f++) {
                    if (f != excludeFileID) {
                        File file = dataFolder.listFiles()[f];
                        if (file.isFile()) {
                            Document dom = null;
                            DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
                            try {
                                DocumentBuilder db = dbf.newDocumentBuilder();
                                dom = db.parse(file);
                            } catch (ParserConfigurationException pce) {
                                pce.printStackTrace();
                            } catch (SAXException se) {
                                se.printStackTrace();
                            } catch (IOException ioe) {
                                ioe.printStackTrace();
                            }
                            if (dom != null) {
                                Element docEle = dom.getDocumentElement();

                                NodeList nodeList = docEle.getElementsByTagName("example");
                                if (nodeList != null && nodeList.getLength() > 0) {
                                    for (int i = 0; i < nodeList.getLength(); i++) {
                                        Element node = (Element) nodeList.item(i);

                                        String[] nlWords = null;
                                        String predicate = null;
                                        String arg1 = null;
                                        String arg2 = null;

                                        NodeList nl = node.getElementsByTagName("nl");
                                        if (nl != null && nl.getLength() > 0) {
                                            String nlPhrase = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim() + " " + RoboCup.TOKEN_END;
                                            nlWords = nlPhrase.replaceAll("\\'", " \\'").split(" ");
                                        }
                                        boolean isFirst = true;
                                        String arg1word = "";
                                        String arg2word = "";
                                        for (int w = 0; w < nlWords.length; w++) {
                                            if (nlWords[w].startsWith("pink") || nlWords[w].startsWith("purple")) {
                                                if (isFirst) {
                                                    arg1word = nlWords[w];
                                                    nlWords[w] = RoboCup.TOKEN_ARG1;
                                                    isFirst = false;
                                                } else {
                                                    arg2word = nlWords[w];
                                                    nlWords[w] = RoboCup.TOKEN_ARG2;
                                                }
                                            }
                                        }
                                        NodeList mrl = node.getElementsByTagName("mrl");
                                        if (mrl != null && mrl.getLength() > 0) {
                                            String mrNode = ((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase();
                                            mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ");
                                            String[] words = mrNode.split(" ");

                                            for (String word : words) {
                                                if (!word.trim().isEmpty()) {
                                                    if (predicate == null) {
                                                        predicate = word.trim();
                                                    } else if (arg1 == null) {
                                                        arg1 = word.trim();
                                                    } else if (arg2 == null) {
                                                        arg2 = word.trim();
                                                    }
                                                }
                                            }
                                        }

                                        // The ID of the first argument
                                        int arg1ID = arguments.indexOf(arg1);
                                        // The ID of the second argument
                                        int arg2ID = arguments.indexOf(arg2);

                                        boolean arg1mentioned = false;
                                        boolean arg2mentioned = false;
                                        if (nlWords != null) {
                                            for (int w = 0; w < nlWords.length; w++) {
                                                String wordTrainingVector = createWordTrainingVectorChoice(arg1ID, arg2ID, nlWords, w, arg1mentioned, arg2mentioned);
                                                if (!wordTrainingVector.isEmpty()) {
                                                    predicateWordTrainingData.get(predicate).add(wordTrainingVector);
                                                }
                                                if (nlWords[w].equals(RoboCup.TOKEN_ARG1)) {
                                                    arg1mentioned = true;
                                                } else if (nlWords[w].equals(RoboCup.TOKEN_ARG2)) {
                                                    arg2mentioned = true;
                                                }
                                            }
                                        }
                                        if (arg1 != null) {
                                            String argTrainingVector1 = createArgTrainingVector(arg1ID, arg1word);
                                            if (!argTrainingVector1.isEmpty()) {
                                                predicateArgTrainingData.get(predicate).add(argTrainingVector1);
                                            }
                                            if (argDictionaryMap.get(arg1).containsKey(arg1word)) {
                                                argDictionaryMap.get(arg1).put(arg1word, argDictionaryMap.get(arg1).get(arg1word) + 1);
                                            } else {
                                                argDictionaryMap.get(arg1).put(arg1word, 1);
                                            }
                                        }
                                        if (arg2 != null) {
                                            String argTrainingVector2 = createArgTrainingVector(arg2ID, arg2word);
                                            if (!argTrainingVector2.isEmpty()) {
                                                predicateArgTrainingData.get(predicate).add(argTrainingVector2);
                                            }
                                            if (argDictionaryMap.get(arg2).containsKey(arg2word)) {
                                                argDictionaryMap.get(arg2).put(arg2word, argDictionaryMap.get(arg2).get(arg2word) + 1);
                                            } else {
                                                argDictionaryMap.get(arg2).put(arg2word, 1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (String predicate : predicateWordTrainingData.keySet()) {
                String wordsPath = trainingFilePath + "_words_" + predicate;
                if (excludeFileID != -1) {
                    wordsPath += "_excl" + excludeFileID;
                }
                String argsPath = trainingFilePath + "_args_" + predicate;
                if (excludeFileID != -1) {
                    argsPath += "_excl" + excludeFileID;
                }
                try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(wordsPath), "utf-8"))) {
                    for (String trainingVector : predicateWordTrainingData.get(predicate)) {
                        writer.write(trainingVector + "\n");
                    }
                    writer.close();
                } catch (UnsupportedEncodingException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                }
                try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(argsPath), "utf-8"))) {
                    for (String trainingVector : predicateArgTrainingData.get(predicate)) {
                        writer.write(trainingVector + "\n");
                    }
                    writer.close();
                } catch (UnsupportedEncodingException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IOException ex) {
                    Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
    }

    public static String createWordTrainingVectorChoice(int arg1ID, int arg2ID, String[] nlWords, int w, boolean arg1mentioned, boolean arg2mentioned) {
        return createTrainingVectorDiffFeaturesPerClass(arg1ID, arg2ID, nlWords, w, arg1mentioned, arg2mentioned);
    }

    public static String createTrainingVector(int arg1ID, int arg2ID, String[] nlWords, int w) {
        /*System.out.print("NLW ");
         for (String word : nlWords) {
         System.out.print(word + " ");
         }
         System.out.println();*/

        String trainingVector = "";

        String word = nlWords[w].toLowerCase().trim();
        int featureNo = 1;
        if (!word.isEmpty()) {
            // The ID of the class (the word which should be generated) according to the dictionary
            int wordClassID = dictionary.indexOf(word);
            //Class and argument features
            trainingVector = wordClassID + "";
            trainingVector += " " + (featureNo++) + ":" + arg1ID;
            trainingVector += " " + (featureNo++) + ":" + arg2ID;
            //Previous word features
            for (int j = 1; j <= 5; j++) {
                int previousWordID = -1;
                if (w - j >= 0) {
                    String previousWord = nlWords[w - j].trim();
                    previousWordID = dictionary.indexOf(previousWord);
                }
                trainingVector += " " + (featureNo++) + ":" + previousWordID;
            }
            //Word Positions
            //trainingVector += " " + (featureNo++) + ":" + w;
            //THIS WORD 
            //trainingVector += " " + (featureNo++) + ":" + dictionary.indexOf(word);
        }
        return trainingVector;
    }

    public static String createTrainingVectorDiffFeaturesPerClass(int arg1ID, int arg2ID, String[] nlWords, int w, boolean arg1mentioned, boolean arg2mentioned) {
        String trainingVector = "";

        String word = nlWords[w].toLowerCase().trim();
        int featureNo = 1;
        if (!word.isEmpty()) {
            // The ID of the class (the word which should be generated) according to the dictionary
            int wordClassID = dictionary.indexOf(word);
            //Class and argument features
            trainingVector = wordClassID + "";
            //Arg1 ID
            for (int i = 0; i < arguments.size(); i++) {
                int featureValue = 0;
                if (i == arg1ID) {
                    featureValue = 1;
                }
                trainingVector += " " + (featureNo++) + ":" + featureValue;
            }
            //Arg2 ID
            for (int i = 0; i < arguments.size(); i++) {
                int featureValue = 0;
                if (i == arg2ID) {
                    featureValue = 1;
                }
                trainingVector += " " + (featureNo++) + ":" + featureValue;
            }
            //Previous word features
            for (int j = 1; j <= 5; j++) {
                /*int previousWordID = -1;
                 if (w - j >= 0) {
                 String previousWord = nlWords[w - j].trim();
                 previousWordID = dictionary.indexOf(previousWord);
                 }*/
                if (w - j >= 0) {
                    String previousWord = nlWords[w - j].trim();
                    for (int i = 0; i < dictionary.size(); i++) {
                        int featureValue = 0;
                        if (dictionary.get(i).equals(previousWord)) {
                            featureValue = 1;
                        }
                        trainingVector += " " + (featureNo++) + ":" + featureValue;
                    }
                }
            }
            //Word Positions
            //trainingVector += " " + (featureNo++) + ":" + w;
            //If arguments have already been generated or not
            if (arg1mentioned) {
                trainingVector += " " + (featureNo++) + ":1";
            } else {
                trainingVector += " " + (featureNo++) + ":0";
            }
            if (arg2mentioned) {
                trainingVector += " " + (featureNo++) + ":1";
            } else {
                trainingVector += " " + (featureNo++) + ":0";
            }
        }
        return trainingVector;
    }

    public static String createArgTrainingVector(int argID, String word) {
        String trainingVector = "";

        int featureNo = 1;
        if (!word.isEmpty()) {
            // The ID of the class (the word which should be generated) according to the dictionary
            int wordClassID = argDictionary.indexOf(word);
            //Class and argument features
            trainingVector = wordClassID + "";
            //Arg ID
            for (int i = 0; i < arguments.size(); i++) {
                int featureValue = 0;
                if (i == argID) {
                    featureValue = 1;
                }
                trainingVector += " " + (featureNo++) + ":" + featureValue;
            }
        }
        return trainingVector;
    }

    public static void getGradedTrainingErrorRate(ArrayList<Instance> instances, Double param) {
        System.out.println("training data: " + instances.size() + " instances");

        for (double g = 0.1; g < 1.0; g += 0.1) {
            int grade = (int) Math.round(instances.size() * (g + 0.1));
            if (grade > instances.size()) {
                grade = instances.size();
            }
            ArrayList<Instance> gradedTrainingInstances = new ArrayList(instances.subList(0, grade));

            // the last parameter can be set to True if probabilities are needed.
            JAROW classifierGraded = new JAROW();
            classifierGraded.train(gradedTrainingInstances, true, false, 10, param, true);

            System.out.println("test data: " + gradedTrainingInstances.size() + " instances");
            int errors = 0;
            for (Instance instance : gradedTrainingInstances) {
                Prediction predict = classifierGraded.predict(instance);
                if (!instance.getCorrectLabels().contains(predict.getLabel())) {
                    errors++;
                }
            }
            Double errorRate = errors / (double) gradedTrainingInstances.size();
            System.out.println("Training error rate of training on\t" + g + "\tof training data is:\t" + errorRate);
        }
    }

    public static void getCrossValidatedGradedErrorRate(ArrayList<Instance> instances, Double param) {
        //10-FOLD STAGED CROSS VALIDATION ON WORDS
        for (double f = 0.0; f < 1.0; f += 0.1) {
            int from = ((int) Math.round(instances.size() * f)) + 1;
            if (from < instances.size()) {
                int to = (int) Math.round(instances.size() * (f + 0.1));
                if (to > instances.size()) {
                    to = instances.size();
                }
                ArrayList<Instance> testingInstances = new ArrayList(instances.subList(from, to));
                ArrayList<Instance> trainingInstances = new ArrayList(instances);
                for (Instance testInstance : testingInstances) {
                    trainingInstances.remove(testInstance);
                }

                System.out.println("training data: " + trainingInstances.size() + " instances");
                //classifier_p.train(trainingInstances, True, True, 10, 10)

                for (double g = 0.1; g < 1.0; g += 0.1) {
                    int grade = (int) Math.round(instances.size() * (g + 0.1));
                    if (grade > instances.size()) {
                        grade = instances.size();
                    }
                    ArrayList<Instance> gradedTrainingInstances = new ArrayList(instances.subList(0, grade));

                    // the last parameter can be set to True if probabilities are needed.
                    JAROW classifierGraded = new JAROW();
                    classifierGraded.train(gradedTrainingInstances, true, false, 10, param, true);

                    System.out.println("test data: " + testingInstances.size() + " instances");
                    int errors = 0;
                    for (Instance instance : testingInstances) {
                        Prediction predict = classifierGraded.predict(instance);
                        if (!instance.getCorrectLabels().contains(predict.getLabel())) {
                            errors++;
                        }
                    }
                    Double errorRate = errors / (double) testingInstances.size();
                    System.out.println("Testing error rate of training on\t" + g + "\tof training data is:\t" + errorRate);
                }
            }
        }
    }

    public static void evaluateGeneration(ArrayList<Instance> trainingWordInstances, /*ArrayList<Instance> trainingArgInstances, */ File testFile, Double param) {
        ArrayList<String> mrNodes = new ArrayList<>();
        HashMap<String, ArrayList<Sequence<String>>> references = new HashMap<>();
        ArrayList<Sequence<String>> generations = new ArrayList<>();
        
        JAROW classifierWords = new JAROW();
        classifierWords.train(trainingWordInstances, true, false, 10, 10.0, true);

        Document dom = null;
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        try {
            DocumentBuilder db = dbf.newDocumentBuilder();
            dom = db.parse(testFile);
        } catch (ParserConfigurationException pce) {
            pce.printStackTrace();
        } catch (SAXException se) {
            se.printStackTrace();
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
        if (dom != null) {
            Element docEle = dom.getDocumentElement();

            NodeList nodeList = docEle.getElementsByTagName("example");
            if (nodeList != null && nodeList.getLength() > 0) {
                for (int i = 0; i < nodeList.getLength(); i++) {
                    Element node = (Element) nodeList.item(i);
                    ArrayList<String> seqList = new ArrayList<>();

                    String[] nlWords = null;
                    String predicate = null;
                    String arg1 = null;
                    String arg2 = null;
                    NodeList nl = node.getElementsByTagName("nl");
                    if (nl != null && nl.getLength() > 0) {
                        String nlPhrase = ((Element) nl.item(0)).getFirstChild().getNodeValue().toLowerCase().trim() + " " + RoboCup.TOKEN_END;
                        nlWords = nlPhrase.replaceAll("\\'", " \\'").split(" ");

                        for (int w = 0; w < nlWords.length; w++) {
                            if (!nlWords[w].isEmpty()) {
                                seqList.add(nlWords[w]);
                            }
                        }
                        SimpleSequence<String> seq = new SimpleSequence<>(seqList);

                        String[] predictWords = new String[nlWords.length];
                        for (int w = 0; w < predictWords.length; w++) {
                            predictWords[w] = "";
                        }

                        NodeList mrl = node.getElementsByTagName("mrl");
                        if (mrl != null && mrl.getLength() > 0) {
                            String mrNode = ((Element) mrl.item(0)).getFirstChild().getNodeValue().toLowerCase();
                            mrNode = mrNode.replaceAll("\\(", " ").replaceAll("\\)", " ").replaceAll("\\,", " ").replaceAll("\\s+", " ");
                            String[] words = mrNode.split(" ");

                            for (String word : words) {
                                if (!word.trim().isEmpty()) {
                                    if (predicate == null) {
                                        predicate = word.trim();
                                    } else if (arg1 == null) {
                                        arg1 = word.trim();
                                    } else if (arg2 == null) {
                                        arg2 = word.trim();
                                    }
                                }
                            }
                            
                            mrNodes.add(mrNode);
                            if (references.containsKey(mrNode)) {
                                references.get(mrNode).add(seq);
                            } else {                                
                                references.put(mrNode, new ArrayList<Sequence<String>>());
                                references.get(mrNode).add(seq);
                            }
                        }
                        
                        if (predicate != null && predicate.equals("pass")) {
                            // The ID of the first argument
                            int arg1ID = arguments.indexOf(arg1);
                            // The ID of the second argument
                            int arg2ID = arguments.indexOf(arg2);

                            //PHRASE GENERATION EVALUATION
                            String predictedWord = "";
                            int w = 0;
                            ArrayList<String> predictedWordsList = new ArrayList<>();
                            boolean arg1mentioned = false;
                            boolean arg2mentioned = false;
                            while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10) {
                                predictedWordsList.add("@TOK@");

                                String trainingVector = createWordTrainingVectorChoice(arg1ID, arg2ID, predictedWordsList.toArray(new String[0]), w, arg1mentioned, arg2mentioned);

                                //System.out.println("TV " + trainingVector);
                                if (!trainingVector.isEmpty()) {
                                    TObjectDoubleHashMap<String> featureVector = new TObjectDoubleHashMap<>();
                                    TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

                                    String[] details;
                                    details = trainingVector.split(" ");

                                    for (int j = 1; j < details.length; j++) {
                                        String[] feature;
                                        feature = details[j].split(":");

                                        featureVector.put(feature[0], Double.parseDouble(feature[1]));
                                    }

                                    Prediction predict = classifierWords.predict(new Instance(featureVector, costs));
                                    predictedWord = predict.getLabel().trim();
                                    predictedWordsList.set(w, predictedWord);

                                    if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                                        arg1mentioned = true;
                                    } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                                        arg2mentioned = true;
                                    }

                                    //System.out.println(trainingVector);
                                    //System.out.println("T: " + nlWords[w] + " P: " + predict.getLabel());
                                }
                                w++;
                            }

                            String predictedString = "";
                            for (String word : predictedWordsList) {
                                predictedString += word + " ";
                            }
                            predictedString = predictedString.trim();

                            /*for (int p = 0; p < predictedWordsList.size(); p++) {
                             predictedWord = predictedWordsList.get(p);
                             if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                             String argTrainingVector = createArgTrainingVector(arg1ID, arg1);
                             if (!argTrainingVector.isEmpty()) {
                             TObjectDoubleHashMap<String> argFeatureVector = new TObjectDoubleHashMap<>();
                             TObjectDoubleHashMap<String> argCosts = new TObjectDoubleHashMap<>();

                             String[] argDetails;
                             argDetails = argTrainingVector.split(" ");

                             for (int j = 1; j < argDetails.length; j++) {
                             String[] feature;
                             feature = argDetails[j].split(":");

                             argFeatureVector.put(feature[0], Double.parseDouble(feature[1]));
                             }        

                             Prediction argPredict = classifierArgs.predict(new Instance(argFeatureVector, argCosts));
                             predictedWord = argPredict.getLabel().trim();
                             predictedWordsList.set(p, predictedWord);
                             }
                             } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                             String argTrainingVector = createArgTrainingVector(arg2ID, arg2);
                             if (!argTrainingVector.isEmpty()) {
                             TObjectDoubleHashMap<String> argFeatureVector = new TObjectDoubleHashMap<>();
                             TObjectDoubleHashMap<String> argCosts = new TObjectDoubleHashMap<>();

                             String[] argDetails;
                             argDetails = argTrainingVector.split(" ");

                             for (int j = 1; j < argDetails.length; j++) {
                             String[] feature;
                             feature = argDetails[j].split(":");

                             argFeatureVector.put(feature[0], Double.parseDouble(feature[1]));
                             }        

                             Prediction argPredict = classifierArgs.predict(new Instance(argFeatureVector, argCosts));
                             predictedWord = argPredict.getLabel().trim();
                             predictedWordsList.set(p, predictedWord);
                             }
                             }
                             }

                             String predictedStringWithArgs = "";
                             for (String word : predictedWordsList) {
                             predictedStringWithArgs += word + " ";
                             }*/
                            for (int p = 0; p < predictedWordsList.size(); p++) {
                                predictedWord = predictedWordsList.get(p);
                                switch (predictedWord) {
                                    case RoboCup.TOKEN_ARG1: {
                                        String name = "";
                                        if (argDictionaryMap.containsKey(arg1)) {
                                            if (!argDictionaryMap.get(arg1).isEmpty()) {
                                                int max = 0;
                                                for (String n : argDictionaryMap.get(arg1).keySet()) {
                                                    int freq = argDictionaryMap.get(arg1).get(n);
                                                    if (freq > max) {
                                                        max = freq;
                                                        name = n;
                                                    }
                                                }
                                            }
                                        }
                                        if (name.isEmpty()) {
                                            name = arg1.trim();
                                        }
                                        predictedWordsList.set(p, name);
                                        break;
                                    }
                                    case RoboCup.TOKEN_ARG2: {
                                        String name = "";
                                        if (argDictionaryMap.containsKey(arg2)) {
                                            if (!argDictionaryMap.get(arg2).isEmpty()) {
                                                int max = 0;
                                                for (String n : argDictionaryMap.get(arg2).keySet()) {
                                                    int freq = argDictionaryMap.get(arg2).get(n);
                                                    if (freq > max) {
                                                        max = freq;
                                                        name = n;
                                                    }
                                                }
                                            }
                                        }
                                        if (name.isEmpty()) {
                                            name = arg2.trim();
                                        }
                                        predictedWordsList.set(p, name);
                                        break;
                                    }
                                }
                            }

                            String predictedStringWithArgs = "";
                            ArrayList<String> seqGenList = new ArrayList<>();
                            for (String word : predictedWordsList) {
                                if (!word.equals(RoboCup.TOKEN_END) && !word.isEmpty()) {
                                    predictedStringWithArgs += word + " ";
                                    seqGenList.add(word);
                                }
                            }
                            predictedStringWithArgs = predictedStringWithArgs.trim();
                            
                            generations.add(new SimpleSequence<>(seqGenList));

                            System.out.println("T: " + ((Element) nl.item(0)).getFirstChild().getNodeValue().trim().toLowerCase());
                            System.out.println("P: " + predictedString);
                            System.out.println("A: " + predictedStringWithArgs);
                            System.out.println("==============");
                            
                        }
                    }
                }
            }
            ArrayList<ArrayList<Sequence<String>>> finalReferences = new ArrayList<>();
            for (String mrNode : mrNodes) {
                finalReferences.add(references.get(mrNode));
            }
            BLEUMetric BLEU = new BLEUMetric(finalReferences, true);
            Double BLEUScore = BLEU.scoreSeq(generations);
            NISTMetric NIST = new NISTMetric(finalReferences);
            Double NISTScore = NIST.scoreSeq(generations);
            
            System.out.println("BATCH BLEU SCORE:\t" + BLEUScore);
            System.out.println("BATCH NIST SCORE:\t" + NISTScore);
        }
    }
}
