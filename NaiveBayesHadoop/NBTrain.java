import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;

import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class NBTrain {
	static enum COUNTERS{VOCAB_SIZE, NLABELS, NDOCS}
	static long vocabSize, nLabels, nDocs;
	static final HashSet<String> allLabels = new HashSet<String>();
	static long correctPredictions=0;
	static long totalDocsClassified = 0;
	
	//This hashmap stores values of C(Y=y) and C(X=* and Y=y)	There are only 2*|dom(Y)| of these counters in all.
	static HashMap<String, Integer> smallCounterSet = new HashMap<String, Integer>();
	
	/*Mapper-Reducer 1*/
	public static class WordWiseMapper extends Mapper<Object, Text, Text, Text> {
		
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			//get the labels of a document
			String documentWithLabel = value.toString();
			List<String> catLabels = extractCATLabels(documentWithLabel);
			
			//get the words of the document
			String document = documentWithLabel.substring(documentWithLabel.indexOf('\t'));
			List<String> documentWords = tokenizeDoc(document);
			
			//write as word, label pair. This helps accumulating everything w.r.t a word
			//i.e. Map (X=x && Y=y)
			for (String catLabel : catLabels) {
				for (String word : documentWords) {
					context.write(new Text(word.toUpperCase()), new Text(catLabel));
				}
			}
		}
	}

	public static class WordWiseReducer extends Reducer<Text, Text, Text, Text> {

		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			//each key-value pair corresponds to a unique word, hence increment the vocabulary count
			context.getCounter(COUNTERS.VOCAB_SIZE).increment(1);
			
			//parameter 'values' contains all labels of occurrences of the word 'key'. Accumulate the label counts for each label
			HashMap<String, Integer> labelCounts = new HashMap<String, Integer>();
			
			//identity for testing only
//			for (Text text : values) {
//				context.write(key, text);
//			}
			
			for (Text label : values) {
				String labelString = label.toString();
				if(labelCounts.containsKey(labelString))
					labelCounts.put(labelString, labelCounts.get(labelString).intValue()+1);
				else
					labelCounts.put(labelString, 1);
			}
			//Output C(X=x && Y=y)
			for (String label : labelCounts.keySet()) {
				context.write(new Text(key.toString() + "^" + label), new Text(labelCounts.get(label).toString()));
			}
		}
	}
	/*End Mapper-Reducer 1*/
	/**********************************************************************************************************************************************************/
	/*Mapper-Reducer 2*/
	public static class LabelWiseMapper extends Mapper<Object, Text, Text, Text> {
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			//get the labels of a document
			String documentWithLabel = value.toString();
			List<String> catLabels = extractCATLabels(documentWithLabel);
			
			//get the words of the document
			String document = documentWithLabel.substring(documentWithLabel.indexOf('\t'));
			List<String> documentWords = tokenizeDoc(document);
			
			//write as label, word pair. This helps accumulating everything w.r.t a label
			for (String catLabel : catLabels) {
				for (String word : documentWords) {
					//Map (Y=y && X=*)
					context.write(new Text(catLabel), new Text(word.toUpperCase()));
				}
			}
		}
	}
	
	public static class LabelWiseReducer extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			//each key-value pair corresponds to a unique label, hence increment the label count
			context.getCounter(COUNTERS.NLABELS).increment(1);
			
			//parameter 'values' contains all words with duplicates seen under the label 'key'. 
			//Here we just need the count of all the words for the current label C(Y=y && X=*)
			int wordCount = 0;
			while (values.iterator().hasNext()) {
				wordCount++;
				values.iterator().next();
			}
			//Output (Y=y && X=*)
			key = new Text("y*:"+key.toString());
			context.write(key, new Text(String.valueOf(wordCount)));
			smallCounterSet.put(key.toString(), wordCount);			
		}
	}
	
	/*End Mapper-Reducer 2*/	
	/**********************************************************************************************************************************************************/
	/*Mapper-Reducer 3*/
	public static class DocumentLabelMapper extends Mapper<Object, Text, Text, Text> {
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			//get the labels of a document
			String documentWithLabel = value.toString();
			List<String> catLabels = extractCATLabels(documentWithLabel);
			
			//write as label, 1 . This helps in accumulating the count of a label
			for (String catLabel : catLabels) {
				//A label counts as a document. Since documents may have multiple labels, each label for a document counts as a document i.e. C(Y=*)=1
				context.getCounter(COUNTERS.NDOCS).increment(1);
				//Map (Y=y)
				context.write(new Text(catLabel), new Text("1"));
			}
		}
	}
	
	public static class DocumentLabelReducer extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			allLabels.add(key.toString());
			//parameter 'values' contains all words with duplicates seen under the label 'key'. 
			//Here we just need the count of all the words for the current label C(Y=y && X=*)
			int labelCount = 0;
			for (Text labelOccurence : values) {
				labelCount++;
			}
			//Output (Y=y && X=*)
			key = new Text("y:"+key.toString());
			context.write(key, new Text(String.valueOf(labelCount)));
			smallCounterSet.put(key.toString(), labelCount);
		}
	}
	/*End Mapper-Reducer 3*/
	/**********************************************************************************************************************************************************/
	/*Mapper-Mapper-Reducer 4*/
	//Reads Model and maps as a <word>, <label:count> pair
	public static class WordModelMapper extends Mapper<Object, Text, Text, Text> {
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String wordLabelCount[] = value.toString().split("\\^");
			String labelCount[] = wordLabelCount[1].split("\t");
			context.write(new Text(wordLabelCount[0]), new Text(labelCount[0] + ":" +labelCount[1]));
		}
	}
	//Maps test data
	public static class TestDataMapper extends Mapper<Object, Text, Text, Text> {
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			//get DocumentID
			String documentId = String.valueOf(((LongWritable)key).get());
			
			//get the words of the document and its correct label(for accuracy calculation later)
			String documentWithLabel = value.toString();
			List<String> labels = extractCATLabels(documentWithLabel);
			if(labels.size()>0) {	//ignore documents without CAT labels
				List<String> documentWords = tokenizeDoc(documentWithLabel.substring(documentWithLabel.indexOf('\t')));
				String labelString = convertToHashTagSeparatedString(labels);
				//write as word, doc pair. This helps accumulating everything w.r.t a word
				for (String word : documentWords) {
					context.write(new Text(word.toUpperCase()), new Text(documentId + labelString));
				}
			}
		}

		private String convertToHashTagSeparatedString(List<String> labels) {
			StringBuilder labelString = new StringBuilder();
			for (String label : labels) {
				labelString.append("#" + label);
			}
			labelString.append("\b");
			return labelString.toString();
		}
	}
	//Join reducer
	public static class JoinReducer extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			String valuesString="";
			for (Text text : values) {
				valuesString+=(text.toString()+",");
			}
			valuesString = valuesString.substring(0, valuesString.length()-1);
			context.write(key, new Text(valuesString));
		}
	}	
	/*End Mapper-Reducer 4*/
	/**********************************************************************************************************************************************************/
	/*Mapper-Reducer 5*/
	public static class PreClassifierMapper extends Mapper<Object, Text, Text, Text> {
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String wordData[] = value.toString().split("\t");
			String data[] = wordData[1].split(",");
			
			HashMap<String, Long> labelCount = new HashMap<String, Long>();
			ArrayList<String> documentList = new ArrayList<String>();
			for (String part : data) {
				if(part.contains(":")) {
					int separator = part.indexOf(":");
					labelCount.put(part.substring(0, separator), Long.parseLong(part.substring(separator+1)));
				}
				else {
					documentList.add(part);
				}					
			}
			//Throw away those words that aren't in any of the documents. They aren't.toString() useful for classification.
			if(documentList.size()>0){
				for (String document : documentList) {
					for (String label : allLabels) {
						double logTerm = labelCount.containsKey(label) ? labelCount.get(label) : 0;
						//Use Laplacian smoothing
						logTerm += 1.0/vocabSize;
						
						logTerm /= (smallCounterSet.get("y*:"+label)+1);
						logTerm = Math.log(logTerm);
						context.write(new Text(document + ":" + label), new Text(String.valueOf(logTerm)));
					}
				}
			}			
		}
	}
	
	public static class PreClassifierReducer extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			double sumLogTerms = 0;
			for (Text term  : values) {
				sumLogTerms+=Double.parseDouble(term.toString());
			}
			double logLabelPriorProb = smallCounterSet.get("y:"+key.toString().substring(key.toString().indexOf(":")+1));
			logLabelPriorProb += 1.0/nLabels;
			
			logLabelPriorProb /= (nDocs+1);
			
			logLabelPriorProb = Math.log(logLabelPriorProb);
			context.write(key, new Text(String.valueOf(sumLogTerms+logLabelPriorProb)));
		}
	}
	/*End Mapper-Reducer 5*/
	/**********************************************************************************************************************************************************/
	/*Mapper-Reducer 6*/
	public static class ClassifierMapper extends Mapper<Object, Text, Text, Text> {
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String line = value.toString();
			String docId = line.substring(0, line.indexOf(":"));
			String labelProb = line.substring(line.indexOf(":")+1);
			context.write(new Text(docId), new Text(labelProb));
		}
	}
	
	public static class ClassifierReducer extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			double highestProb = Double.NEGATIVE_INFINITY;
			String bestLabel = "";
			for (Text labelProb : values) {
				String labelProbString = labelProb.toString();
				String label = labelProbString.substring(0, labelProbString.indexOf("\t"));
				double prob = Double.parseDouble(labelProbString.substring(labelProbString.indexOf("\t")+1));
				if(prob>highestProb) {
					highestProb = prob;
					bestLabel = label;
				}
			}
			//for accuracy calculation
			if(key.toString().contains(bestLabel))
				correctPredictions++;
			totalDocsClassified++;
			
			String docId = key.toString().substring(0, key.toString().indexOf("#"));
			context.write(new Text(key), new Text(bestLabel+"\t"+highestProb));
		}
	}
	/*End Mapper-Reducer 6*/
	
	
	
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

	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args)
				.getRemainingArgs();
		if (otherArgs.length < 1) {
			System.err.println("Usage: NBTrain <input1> [<input2>...] <output>");
			System.exit(2);
		}
		
		//Job 1: For Mapper-Reducer 1:
		Job wordLabelVocabJob = Job.getInstance(conf, "word-label and vocab counter");
		wordLabelVocabJob.setJarByClass(NBTrain.class);
		wordLabelVocabJob.setMapperClass(WordWiseMapper.class);
		//job.setCombinerClass(WordWiseReducer.class);
		wordLabelVocabJob.setReducerClass(WordWiseReducer.class);
		
		wordLabelVocabJob.setMapOutputKeyClass(Text.class);
		wordLabelVocabJob.setMapOutputValueClass(Text.class);
		
		wordLabelVocabJob.setOutputKeyClass(Text.class);
		wordLabelVocabJob.setOutputValueClass(Text.class);
		
		wordLabelVocabJob.getConfiguration().set("mapreduce.output.basename", "output");
		
		FileInputFormat.addInputPath(wordLabelVocabJob, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(wordLabelVocabJob, new Path("WordStatsModel"));
		
		
		//Job 2: For Mapper-Reducer 2:
		Job labelStatsJob = Job.getInstance(conf, "label-words and label counter");
		labelStatsJob.setJarByClass(NBTrain.class);
		labelStatsJob.setMapperClass(LabelWiseMapper.class);
		//job.setCombinerClass(LabelWiseReducer.class);
		labelStatsJob.setReducerClass(LabelWiseReducer.class);
		
		labelStatsJob.setMapOutputKeyClass(Text.class);
		labelStatsJob.setMapOutputValueClass(Text.class);
		
		labelStatsJob.setOutputKeyClass(Text.class);
		labelStatsJob.setOutputValueClass(Text.class);
		
		labelStatsJob.getConfiguration().set("mapreduce.output.basename", "output");
		
		FileInputFormat.addInputPath(labelStatsJob, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(labelStatsJob, new Path("LabelStatsModel"));
		
		
		//Job 3: For Mapper-Reducer 3:
		Job documentLabelJob = Job.getInstance(conf, "document-label and document counter");
		documentLabelJob.setJarByClass(NBTrain.class);
		documentLabelJob.setMapperClass(DocumentLabelMapper.class);
		//job.setCombinerClass(DocumentLabelReducer.class);
		documentLabelJob.setReducerClass(DocumentLabelReducer.class);
		
		documentLabelJob.setMapOutputKeyClass(Text.class);
		documentLabelJob.setMapOutputValueClass(Text.class);
		
		documentLabelJob.setOutputKeyClass(Text.class);
		documentLabelJob.setOutputValueClass(Text.class);
		
		documentLabelJob.getConfiguration().set("mapreduce.output.basename", "output");
		
		FileInputFormat.addInputPath(documentLabelJob, new Path(otherArgs[0]));

		FileOutputFormat.setOutputPath(documentLabelJob, new Path("DocumentStatsModel"));

		
		//Job 4: For Mapper-Mapper-Reducer joiner:
		Job testModelJoinJob = Job.getInstance(conf, "test and model joiner");
		testModelJoinJob.setJarByClass(NBTrain.class);
		
		MultipleInputs.addInputPath(testModelJoinJob, new Path("WordStatsModel/output-r-00000"), TextInputFormat.class, WordModelMapper.class);
		MultipleInputs.addInputPath(testModelJoinJob, new Path(otherArgs[1]), TextInputFormat.class, TestDataMapper.class);		
		
		//job.setCombinerClass(WordLabelReducer.class);
		testModelJoinJob.setReducerClass(JoinReducer.class);
		
		testModelJoinJob.setMapOutputKeyClass(Text.class);
		testModelJoinJob.setMapOutputValueClass(Text.class);
		
		testModelJoinJob.setOutputKeyClass(Text.class);
		testModelJoinJob.setOutputValueClass(Text.class);
		
		testModelJoinJob.getConfiguration().set("mapreduce.output.basename", "output");
		
		FileOutputFormat.setOutputPath(testModelJoinJob, new Path("TestModelJoinedResults"));
		
		
		//Job 5: For Mapper-Reducer 5:
		Job preClassificationJob = Job.getInstance(conf, "pre-classification");
		preClassificationJob.setJarByClass(NBTrain.class);
		preClassificationJob.setMapperClass(PreClassifierMapper.class);
		//job.setCombinerClass(DocumentLabelReducer.class);
		preClassificationJob.setReducerClass(PreClassifierReducer.class);
		
		preClassificationJob.setMapOutputKeyClass(Text.class);
		preClassificationJob.setMapOutputValueClass(Text.class);
		
		preClassificationJob.setOutputKeyClass(Text.class);
		preClassificationJob.setOutputValueClass(Text.class);
		
		preClassificationJob.getConfiguration().set("mapreduce.output.basename", "output");
		
		FileInputFormat.addInputPath(preClassificationJob, new Path("TestModelJoinedResults/output-r-00000"));

		FileOutputFormat.setOutputPath(preClassificationJob, new Path("PreClassificationResults"));

		
		//Job 5: For Mapper-Reducer 6:
		Job classificationJob = Job.getInstance(conf, "classification");
		classificationJob.setJarByClass(NBTrain.class);
		classificationJob.setMapperClass(ClassifierMapper.class);
		//job.setCombinerClass(DocumentLabelReducer.class);
		classificationJob.setReducerClass(ClassifierReducer.class);
		
		classificationJob.setMapOutputKeyClass(Text.class);
		classificationJob.setMapOutputValueClass(Text.class);
		
		classificationJob.setOutputKeyClass(Text.class);
		classificationJob.setOutputValueClass(Text.class);
		
		classificationJob.getConfiguration().set("mapreduce.output.basename", "output");
		
		FileInputFormat.addInputPath(classificationJob, new Path("PreClassificationResults/output-r-00000"));

		FileOutputFormat.setOutputPath(classificationJob, new Path("ClassificationResults"));
		
		//Execute the jobs		
		boolean success1 = wordLabelVocabJob.waitForCompletion(true);
		vocabSize = wordLabelVocabJob.getCounters().findCounter(COUNTERS.VOCAB_SIZE).getValue();
		
		boolean success2 = labelStatsJob.waitForCompletion(true);
		nLabels = labelStatsJob.getCounters().findCounter(COUNTERS.NLABELS).getValue();
		
		boolean success3 = documentLabelJob.waitForCompletion(true);
		nDocs = documentLabelJob.getCounters().findCounter(COUNTERS.NDOCS).getValue();
		
		boolean success4 = testModelJoinJob.waitForCompletion(true);
		
		boolean success5 = preClassificationJob.waitForCompletion(true);
		
		boolean success6 = classificationJob.waitForCompletion(true);
		
		showCounterValues();
		
		System.out.println("Accuracy: "+correctPredictions+"/"+totalDocsClassified + "="+correctPredictions*100.0/totalDocsClassified);
		
		System.exit(success1 && success2 && success3 && success4 && success5 && success6? 0: 1);
	}

	private static void showCounterValues() {
		System.out.println("Vocab Size = "+vocabSize);
		System.out.println("Unique Labels = "+nLabels);
		System.out.println("Total Documents = "+nDocs);
		
		for (String key : smallCounterSet.keySet()) {
			System.out.println(key + " : " + smallCounterSet.get(key));
		}
	}
	
}
