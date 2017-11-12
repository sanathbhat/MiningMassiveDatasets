
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class NBTrain {
	public static int bufferCapacity = 1000;
	static HashMap<String, Integer> wordCountBuffer = new HashMap<String, Integer>();
	
	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		try (/* For debugging only 
				BufferedReader br = new BufferedReader(new FileReader("bin/RCV1.very_small_train.txt")) */
				BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			
			String documentWithLabel = "";
			while ((documentWithLabel = br.readLine()) != null) {
				List<String> catLabels = extractCATLabels(documentWithLabel);
				// skip document if it has no *CAT label
				if (catLabels.isEmpty())
					continue;

				// get document text by stripping off the labels
				String document = documentWithLabel.substring(documentWithLabel.indexOf('\t'));
				// get all words of the document
				List<String> documentWords = tokenizeDoc(document);

				//print messages for each label of the document
				for (String catLabel : catLabels) {
					//Update C(Y=catLabel)
					String key = catLabel.toUpperCase();
					if(bufferCapacity>0) {						
						addToBuffer(key);
					}
					else
						System.out.println(key+",1");
				}
				
				//print messages for each occurrence of a word with a label 
				for (String word : documentWords) {
					//To update C(-word) for tracking vocabulary
					String vocabKey = "-"+word.toUpperCase();
					addToBuffer(vocabKey);
					
					for (String catLabel : catLabels) {						
						//To update C(X=word ^ Y=catLabel)
						String key1 = (catLabel + "^" + word).toUpperCase();
						//To update C(X=ANY ^ Y=catLabel)
						String key2 = (catLabel + "^*").toUpperCase();
						
						if(bufferCapacity>0){														
							addToBuffer(key1);
							addToBuffer(key2);
						}
						else {
							System.out.println(key1+",1");
							System.out.println(key2+",1");
						}
					}
					// if buffer full, then flush messages out to sysout
					if (bufferCapacity>0 && bufferOverflow()) {
						flushBufferToSysOut();
					}
				}

			}
			flushBufferToSysOut();
			
			System.out.println("AAAAAAAAVocabCounterSeparatorDummy");			
			System.out.println("ZZStartTime:" + startTime);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static boolean bufferOverflow() {
		return wordCountBuffer.size() >= bufferCapacity;
	}

	private static List<String> extractCATLabels(String documentWithLabel) {
		String allLabels[] = documentWithLabel.substring(0, documentWithLabel.indexOf('\t')).split(",");
		List<String> catLabels = new ArrayList<String>();
		for (String aLabel : allLabels) {
			if (aLabel.contains("CAT")) {
				catLabels.add(aLabel);
			}
		}
		return catLabels;
	}
	
	private static void addToBuffer(String key) {
		if (wordCountBuffer.containsKey(key))
			wordCountBuffer.replace(key, wordCountBuffer.get(key) + 1);
		else
			wordCountBuffer.put(key, 1);
	}

	private static void flushBufferToSysOut() {
		try {
			//FileWriter fw = new FileWriter("BufferDumpF10000.txt", true);
			for (String message : wordCountBuffer.keySet()) {
				// convert message to upper case as part of data cleaning
				System.out.println(message + "," + wordCountBuffer.get(message));	
				//fw.write(message + "," + wordCountBuffer.get(message)+"\n");			//Writing to buffer dump
			}
			wordCountBuffer.clear();
			//fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static List<String> tokenizeDoc(String document) {
		String[] words = document.split("\\s+");
		List<String> tokens = new ArrayList<String>();
		for (int i = 0; i < words.length; i++) {
			words[i] = words[i].replaceAll("\\W", "");
			//removing underscores as well
			words[i] = words[i].replaceAll("_", "");
			if (words[i].length() > 0) {
				tokens.add(words[i]);
			}
		}
		return tokens;
	}
}
