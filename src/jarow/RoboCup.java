package jarow;

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
    static ArrayList<String> arguments = new ArrayList<>();
    static ArrayList<String> predicates = new ArrayList<>();
    
    final public static String TOKEN_END = "@end@";
    final public static String TOKEN_ARG1 = "@arg1@";
    final public static String TOKEN_ARG2 = "@arg2@";

    public static void main(String[] args) {
        createLists(new File("robocup_data\\gold\\"));
        saveLists("robocup_data\\");
        //readLists("robocup_data\\");
        
        createTrainingData(new File("robocup_data\\gold\\"), "robocup_data\\goldTrainingData");
                
        nlgTest(new File("robocup_data\\gold\\"), "robocup_data\\");
    }
    
    public static void nlgTest(File dataFolder, String modelPath) {
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
                    costs.put(word, 1.0);
                }
                costs.put(dictionary.get(Integer.parseInt(details[0])), 0.0);
                
                for (int i = 1; i < details.length; i++) {
                    String[] feature;
                    feature = details[i].split(":");
                    
                    featureVector.put(feature[0], Double.parseDouble(feature[1]));
                }
                instances.add(new Instance(featureVector, costs));
                //System.out.println(instances.get(instances.size() - 1).getCosts());
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        Collections.shuffle(instances);
        
        // the last parameter can be set to True if probabilities are needed.
        //Double[] params = {0.01, 0.1, 1.0, 10.0, 100.0};
        //Double[] params = {100.0, 80.0, 120.0, 150.0, 200.0};        
        //JAROW classifier_p = JAROW.trainOpt(instances, 10, params, 0.1, true, false);
        JAROW classifier_p = new JAROW();
        classifier_p.train(instances, true, false, 10, 10.0, true);
        
        try {
            classifier_p.save(classifier_p, modelPath + "_model");
        } catch (IOException ex) {
            Logger.getLogger(RoboCup.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        int errors = 0;
        int total = 0;
        if (dataFolder.isDirectory()) {
            for (File file : dataFolder.listFiles()) {
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
                                    
                                    boolean isFirst = true;
                                    for (int w = 0; w < nlWords.length; w++) {
                                        if (nlWords[w].startsWith("pink") || nlWords[w].startsWith("purple")) {
                                            if (isFirst) {
                                                nlWords[w] = RoboCup.TOKEN_ARG1;
                                                isFirst = false;
                                            } else {
                                                nlWords[w] = RoboCup.TOKEN_ARG2;
                                            }
                                        }
                                    }
                                    
                                    String[] predictWords = new String[nlWords.length];
                                    for (int w = 0; w < predictWords.length; w++) {
                                        predictWords[w] = "";
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
                                    if (predicate != null && predicate.equals("pass")) {
                                        // The ID of the first argument
                                        int arg1ID = arguments.indexOf(arg1);
                                        // The ID of the second argument
                                        int arg2ID = arguments.indexOf(arg2);
                                        
                                        //WORD PREDICTION EVALUATION
                                        boolean arg1mentioned = false;
                                        boolean arg2mentioned = false;
                                        for (int w = 0; w < nlWords.length; w++) {
                                            String trainingVector = createTrainingVectorChoice(arg1ID, arg2ID, nlWords, w, arg1mentioned, arg2mentioned);
                                            
                                            if (nlWords[w].equals(RoboCup.TOKEN_ARG1)) {
                                                arg1mentioned = true;
                                            } else if (nlWords[w].equals(RoboCup.TOKEN_ARG2)) {
                                                arg2mentioned = true;
                                            }
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

                                                Prediction predict = classifier_p.predict(new Instance(featureVector, costs));
                                                if (!nlWords[w].equalsIgnoreCase(predict.getLabel())) {
                                                    errors++;
                                                }
                                                total++;

                                                predictWords[w] = predict.getLabel();
                                                //System.out.println(trainingVector);
                                                //System.out.println("T: " + nlWords[w] + " P: " + predict.getLabel());
                                            }
                                        }
                                        
                                        //System.out.println("------");
                                        
                                        //PHRASE GENERATION EVALUATION
                                        String predictedWord = "";
                                        int w = 0;
                                        ArrayList<String> predictedWordsList = new ArrayList<>();
                                        arg1mentioned = false;
                                        arg2mentioned = false;
                                        while (!predictedWord.equals(RoboCup.TOKEN_END) && predictedWordsList.size() < 10) {
                                            predictedWordsList.add("@TOK@");
                                            
                                            String trainingVector = createTrainingVectorChoice(arg1ID, arg2ID, predictedWordsList.toArray(new String[0]), w, arg1mentioned, arg2mentioned);
                                            
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

                                                Prediction predict = classifier_p.predict(new Instance(featureVector, costs));
                                                predictedWord = predict.getLabel().trim();
                                                predictedWordsList.set(w, predictedWord);
                                                
                                                //System.out.println(trainingVector);
                                                //System.out.println("T: " + nlWords[w] + " P: " + predict.getLabel());
                                                
                                                if (predictedWord.equals(RoboCup.TOKEN_ARG1)) {
                                                    arg1mentioned = true;
                                                } else if (predictedWord.equals(RoboCup.TOKEN_ARG2)) {
                                                    arg2mentioned = true;
                                                }
                                            }
                                            w++;
                                        }
                                        String predictedString = "";
                                        for (String word : predictedWordsList) {
                                            predictedString += word + " ";
                                        }
                                        predictedString = predictedString.trim();
                                        System.out.println("T: " + ((Element) nl.item(0)).getFirstChild().getNodeValue().trim().toLowerCase());
                                        System.out.println("P: " + predictedString);
                                        System.out.println("==============");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        System.out.println("Error rate: " + ((double)errors/(double)total));
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
                Double avgCost = cost/(double)testingInstances.size();
                System.out.println("Avg Cost per instance " + avgCost + " on " + testingInstances.size() + " testing instances");
            }
        }
    }

    public static void createLists(File dataFolder) {
        dictionary.add(RoboCup.TOKEN_END);
        dictionary.add(RoboCup.TOKEN_ARG1);
        dictionary.add(RoboCup.TOKEN_ARG2);
        if (dataFolder.isDirectory()) {
            for (File file : dataFolder.listFiles()) {
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
                                    if (!arguments.contains(arg2) && arg2 != null) {
                                        arguments.add(arg2);
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

    public static void createTrainingData(File dataFolder, String trainingFilePath) {
        if (!dictionary.isEmpty() && !predicates.isEmpty() && !arguments.isEmpty()) {
            HashMap<String, ArrayList<String>> predicateTrainingData = new HashMap<>();
            for (String predicate : predicates) {
                predicateTrainingData.put(predicate, new ArrayList<String>());
            }
            if (dataFolder.isDirectory()) {
                for (File file : dataFolder.listFiles()) {
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
                                    for (int w = 0; w < nlWords.length; w++) {
                                        if (nlWords[w].startsWith("pink") || nlWords[w].startsWith("purple")) {
                                            if (isFirst) {
                                                nlWords[w] = RoboCup.TOKEN_ARG1;
                                                isFirst = false;
                                            } else {
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
                                            String trainingVector = createTrainingVectorChoice(arg1ID, arg2ID, nlWords, w, arg1mentioned, arg2mentioned);
                                            if (!trainingVector.isEmpty()) {
                                                predicateTrainingData.get(predicate).add(trainingVector);
                                            }
                                            if (nlWords[w].equals(RoboCup.TOKEN_ARG1)) {
                                                arg1mentioned = true;
                                            } else if (nlWords[w].equals(RoboCup.TOKEN_ARG2)) {
                                                arg2mentioned = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        
            for (String predicate : predicateTrainingData.keySet()) {
                try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(trainingFilePath + "_" + predicate), "utf-8"))) {
                    for (String trainingVector : predicateTrainingData.get(predicate)) {
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
    
    public static String createTrainingVectorChoice(int arg1ID, int arg2ID, String[] nlWords, int w, boolean arg1mentioned, boolean arg2mentioned) {
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
}