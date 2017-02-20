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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.logging.Logger;

/**
 *
 * @author Gerasimos Lampouras
 */
public class CleanBAGELResults {

    /**
     *
     * @param as
     */
    public static void main(String[] as) {
        HashSet<HashMap<String, HashSet<String>>> nonUniqueAttributes = new HashSet<>();
        HashMap<HashMap<String, HashSet<String>>, String> attributes = new HashMap<>();
        File dataFile = new File("D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW\\ExperimentData\\BAGELOndrejResults.txt");
        try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {
            String readLine;

            while ((readLine = br.readLine()) != null) {
                int a1 = readLine.indexOf("MR@@");
                int b1 = readLine.indexOf("@@", a1 + "MR@@".length());

                String line = readLine.substring(a1 + "MR@@".length(), b1);

                HashMap<String, String> names = new HashMap<>();
                int s = line.indexOf('"');
                int a = 0;
                while (s != -1) {
                    int e = line.indexOf('"', s + 1);

                    String name = line.substring(s, e + 1);
                    line = line.replace(name, "x" + a);
                    names.put("x" + a, name);
                    a++;

                    s = line.indexOf('"');
                }

                HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
                String[] args = line.replaceAll("inform\\(", "").replaceAll("\\)", "").split("&");

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
                        value = "x" + index;
                    }
                    attributeValues.get(attr).add(value.toLowerCase());
                }
                if (attributes.containsKey(attributeValues)) {
                    nonUniqueAttributes.add(attributeValues);
                }
                attributes.put(attributeValues, readLine);
            }
        } catch (FileNotFoundException ex) {
        } catch (IOException ex) {
        }
        
        BufferedWriter bw = null;
        File f = null;
        try {
            f = new File("D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW\\ExperimentData\\BAGELOndrejResultsUNIQUE.txt");
        } catch (NullPointerException e) {
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
        } catch (FileNotFoundException e) {
        }

        for (HashMap<String, HashSet<String>> attr : attributes.keySet()) {
            try {
                if (!nonUniqueAttributes.contains(attr)) {
                    bw.write(attributes.get(attr));
                    bw.write("\n");
                }

            } catch (IOException e) {
            }
        }

        try {
            bw.close();
        } catch (IOException e) {
        }

        HashSet<HashMap<String, HashSet<String>>> nonUniqueAttributes2 = new HashSet<>();
        HashMap<HashMap<String, HashSet<String>>, String> attributes2 = new HashMap<>();
        File dataFile2 = new File("D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW\\ExperimentData\\BAGELLolsResults.txt");
        try (BufferedReader br = new BufferedReader(new FileReader(dataFile2))) {
            String readLine;

            while ((readLine = br.readLine()) != null) {
                int a1 = readLine.indexOf("MR@@");
                int b1 = readLine.indexOf("@@", a1 + "MR@@".length());

                String line = readLine.substring(a1 + "MR@@".length(), b1);

                HashMap<String, String> names = new HashMap<>();
                int s = line.indexOf('"');
                int a = 0;
                while (s != -1) {
                    int e = line.indexOf('"', s + 1);

                    String name = line.substring(s, e + 1);
                    line = line.replace(name, "x" + a);
                    names.put("x" + a, name);
                    a++;

                    s = line.indexOf('"');
                }

                HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
                String[] args = line.split(",");

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
                        value = "x" + index;
                    }
                    attributeValues.get(attr).add(value.toLowerCase());
                }
                if (attributes2.containsKey(attributeValues)) {
                    nonUniqueAttributes2.add(attributeValues);
                }
                attributes2.put(attributeValues, readLine);
            }
        } catch (FileNotFoundException ex) {
        } catch (IOException ex) {
        }

        bw = null;
        f = null;
        try {
            f = new File("D:\\Users\\Black Fox\\Dropbox\\TO-DO\\JAROW\\ExperimentData\\BAGELLolsResultsUNIQUE.txt");
        } catch (NullPointerException e) {
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
        } catch (FileNotFoundException e) {
        }

        for (HashMap<String, HashSet<String>> attr : attributes.keySet()) {
            try {
                if (!nonUniqueAttributes.contains(attr)) {
                    bw.write(attributes2.get(attr));
                    bw.write("\n");
                }

            } catch (IOException e) {
            }
        }

        try {
            bw.close();
        } catch (IOException e) {
        }
    }
}
