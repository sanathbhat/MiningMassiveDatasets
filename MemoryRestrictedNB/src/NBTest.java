import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class NBTest {
	private static HashMap<String, Integer> allCounters = new HashMap<>();
	private static List<String> labelDomain = new ArrayList<>();
	
	public static void main(String[] args) {
		String timeStampInfo = "";
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			String message = "";
						
			while((message = br.readLine())!= null && message.contains(",")) {
				String counterValue[] = message.split(",");
				String counter = counterValue[0];
				int value = Integer.parseInt(counterValue[1]);
				if(value>1)
					allCounters.put(counter, value);
				
//				String counter = message.substring(0, message.indexOf(","));
//				int value = Integer.parseInt(message.substring(message.indexOf(",") + 1, message.length()));
//				allCounters.put(counter, value);
				//System.out.println("HashMap Keys:"+allCounters.size());
			}	
			
			timeStampInfo = message;
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//read and attempt to classify a document
		labelDomain.add("CCAT");labelDomain.add("ECAT");labelDomain.add("GCAT");labelDomain.add("MCAT");
		
		try (BufferedReader br = new BufferedReader(new FileReader(args[0]))){
			String documentWithLabel = "";
			
			long nDocuments = 0;
			long nCorrectPredictions = 0;
			while ((documentWithLabel = br.readLine()) != null) {
				List<String> catLabels = extractCATLabels(documentWithLabel);
				// skip document if it has no *CAT label
				if (catLabels.isEmpty())
					continue;
				
				// get document text by stripping off the labels
				String document = documentWithLabel.substring(documentWithLabel.indexOf('\t'));
				// get all words of the document
				List<String> documentWords = tokenizeDoc(document);
				
				String bestLabel = "";
				double highestlogPr = Double.NEGATIVE_INFINITY;
				
				//for each label in domain, compute logPr
				for (String catLabel : labelDomain) {
					//compute log Pr(catLabel, documentWords)
					double logPr = 0;
					for (String word : documentWords) {						
						double numerator=0, denominator = 0;	//to represent C(X=word^Y=catLabel) and C(X=ANY^Y=catLabel) respectively
						
						String numeratorKey = (catLabel + "^" + word).toUpperCase();
						if(allCounters.containsKey(numeratorKey)) {
							numerator += allCounters.get(numeratorKey);
						}
/*Laplacian*/			numerator += 1;
/*Dirichlet*///			numerator += Math.pow(allCounters.get("VocabularySize"), -1);
						
						String denominatorKey = (catLabel + "^*").toUpperCase();
						if(allCounters.containsKey(denominatorKey)) {
							denominator += allCounters.get(denominatorKey);
						}
/*Laplacian*/			denominator += (allCounters.get("VocabularySize") + 1);
/*Dirichlet*///			denominator += 1;
						
						logPr += Math.log(numerator/denominator);
					}
					
					double outerNumerator=0, outerDenominator = 0;	//to represent C(Y=catLabel) and C(Y=ANY) respectively
					outerNumerator = allCounters.get(catLabel.toUpperCase());
/*Laplacian*/		outerNumerator += 1;
/*Dirichlet*///		outerNumerator += Math.pow(allCounters.get("VocabularySize"), -1);
					
					for (String catLabelForSummation : labelDomain) {
						outerDenominator += (allCounters.get(catLabelForSummation.toUpperCase()));
					}
/*Laplacian*/		outerDenominator += (allCounters.get("VocabularySize") + 1);
/*Dirichlet*///		outerDenominator += 1;
					
					logPr += Math.log(outerNumerator/outerDenominator);
					System.out.print(catLabel + " : "+ logPr + ", ");
					
					//if logPr greater than before, set current catLabel as best label
					if(logPr>highestlogPr){
						highestlogPr = logPr;
						bestLabel = catLabel;
					}										
				}
				System.out.println();
				
				//predicted vs actual display
				System.out.print("Doc#" + nDocuments + ": Actual = ");
				catLabels.forEach(l->System.out.print(l + " "));
				System.out.println(" | Predicted =" + bestLabel);
				System.out.println();
				if(catLabels.contains(bestLabel)) {
					//correct prediction
					nCorrectPredictions++;
				}
				nDocuments++;				
			}
			
			System.out.println("Accuracy = "+nCorrectPredictions+"/"+nDocuments+"=" +nCorrectPredictions*100.0/nDocuments);
			//Time calculations;
			String times[] = timeStampInfo.split(";");
			long startTime = Long.parseLong(times[0].split(":")[1]);
			long trainingEndTime = Long.parseLong(times[1].split(":")[1]);
			long classificnEndTime = System.currentTimeMillis();
			
			System.out.println("Total training time = "+getTimeSpanInSeconds(startTime, trainingEndTime));
			System.out.println("Total classification time = "+getTimeSpanInSeconds(trainingEndTime, classificnEndTime));
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	private static double getTimeSpanInSeconds(long startTime, long trainingEndTime) {
		return (int)(trainingEndTime-startTime)/1000.0;
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
