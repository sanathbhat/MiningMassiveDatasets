import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class MergeCounts {
	private static String previousEvent = "";
	private static int eventCounter = 0;
	private static long vocabularyCounter = 0;
	
	public static void main(String[] args) {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			
			String message = "";
			String timeStampInfo = "";
			
			while((message = br.readLine()).startsWith("-")) {
				//accumulate all words to get vocabulary size
				//ignore duplicates
				String eventAndDelta[] = message.split(",");
				String event = eventAndDelta[0];
				
				if(!event.equals(previousEvent))	{
					//new event, hence, increment vocabulary count
					vocabularyCounter++;
					previousEvent = event;					
				}
			}
			System.out.println("VocabularySize,"+vocabularyCounter);
			
			previousEvent = "";			//clear previous history
			while((message = br.readLine()) != null) {
//				System.out.println(message);
				//if start time-stamp from NBTrain output is encountered
				if(message.startsWith("ZZ")) {
					timeStampInfo = message + (";TrainingEndTime:"+System.currentTimeMillis());
					break;
				}
				
				String eventAndDelta[] = message.split(",");
				String event = eventAndDelta[0];
				int deltaCount = Integer.parseInt(eventAndDelta[1]);
				
				if(event.equals(previousEvent))	//if same event as last seen event
					eventCounter += deltaCount;		//add to the counter
				else {
					//new event, hence, flush current event with its final counter to Sysout
					flushPreviousEventToSysOut();
					previousEvent = event;
					eventCounter = deltaCount;					
				}				
			}
			flushPreviousEventToSysOut();
			System.out.println(timeStampInfo);
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void flushPreviousEventToSysOut() {
		if(!previousEvent.equals(""))	
			System.out.println(previousEvent+","+eventCounter);		
	}

}
