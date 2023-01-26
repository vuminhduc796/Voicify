package com.research.activityinvoker.services;

import android.Manifest;
import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.text.TextUtils;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityManager;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;
import androidx.appcompat.content.res.AppCompatResources;
import androidx.core.app.ActivityCompat;
import androidx.core.graphics.drawable.DrawableCompat;

import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.research.activityinvoker.JsonApi;
import com.research.activityinvoker.R;
import com.research.activityinvoker.SettingsActivity;
import com.research.activityinvoker.model.LabelFoundNode;
import com.research.activityinvoker.model.PackageDataObject;
import com.research.activityinvoker.model.ResponseObject;
import com.research.activityinvoker.model.TooltipRequiredNode;

import org.apache.commons.compress.compressors.gzip.GzipUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Collections;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

@RequiresApi(api = Build.VERSION_CODES.R)
public class ActionFulfilment extends AccessibilityService implements View.OnTouchListener{
    final String FILE_NAME = "voicify";
    final String ALL_COMMANDS = "all_commands";
    int width,height;
    Button listenButton;
    //TextView textMsg;
    TextView inputTxt;
    Retrofit retrofit;
    JsonApi jsonApi;

    ArrayList<String> outputArr = new ArrayList<>();
    ArrayList<AccessibilityNodeInfo> editableNodes = new ArrayList<>();
    ArrayList<String> uiElements = new ArrayList<String>();
    ArrayList<String> previousUIElements = new ArrayList<String>();
    ArrayList<String> appNames = new ArrayList<String>();
    SharedPreferences  mPrefs;
    SharedPreferences sharedPreferences;
    ArrayList<String> predefinedCommands = new ArrayList<>();
    ArrayList<String> currentSequence = new ArrayList<String>();
    AccessibilityNodeInfo currentSource;
    AccessibilityNodeInfo previousSource;
    ArrayList<AccessibilityNodeInfo> scrollableNodes = new ArrayList<AccessibilityNodeInfo>();

    FrameLayout mLayout;
    ArrayList<LabelFoundNode> foundLabeledNodes = new ArrayList<>();
    ArrayList<TooltipRequiredNode> tooltipRequiredNodes = new ArrayList<>();

    boolean isVoiceCommandConnected = false;

    int noOfLabels = 0;
    boolean hasExecuted = true;
    boolean skipPreviousRTECheck = false;
    SpeechRecognizer speechRecognizer;                      // declaring speech recognition var
    Intent speechRecognizerIntent;
    String debugLogTag = "FIT4003_VOICIFY";                  // use this tag for all log tags.
    ArrayList<String> launchTriggers = new ArrayList<String>(Arrays.asList("load","launch","execute","open"));

    // Defining window manager for overlay elements and switchBar
    WindowManager wm;
    WindowManager.LayoutParams switchBar; // stores layout parameters for movable switchBar

    long currentTime;

    // variable for switch bar coordinates
    private int initialX;
    private int initialY;
    private float initialTouchX;
    private float initialTouchY;

    ArrayList<String> savedCommands = new ArrayList<>();
    private int currentTooltipCount = 1;
    boolean isRecording = false;
    boolean isPlaying = false;

    // new code
    ArrayList<PackageDataObject> packageDataObjects = new ArrayList<>();
    int currentIntentIndex = 0;
    String currentOpeningPackage;
    String currentOpeningFeature;
    String currentAppName;
    String currentIntentIdentifier;
    ArrayList<String> matchedIntents = new ArrayList<>();
    ArrayList<String> matchedDeeplinks = new ArrayList<>();

    String[] tooltipColorSpinnerItems = new String[]{"#64b5f6","#2b2b2b","#ff4040"};
    int[] tooltipSizeSpinnerItems = new int[]{14,18,22};
    int[] tooltipOpacitySpinnerItems = new int[]{250,220,170,120};
    int[] buttonOpacitySpinnerItems = new int[]{250,220,170,120};
    boolean[] buttonRecordItems = new boolean[]{false,true};
    boolean[] buttonAlgoItems = new boolean[]{true,false};

    int tooltipColor = 0 ;
    int tooltipSize = 0;
    int tooltipOpacity = 0;
    int buttonOpacity = 0;
    int buttonRecordTxt = 0;
    int buttonAlgoTxt = 0;
    String previousEventCode = "";
    public Word2Vec model;

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        /**
         * This function will be invoked when defined type of event occurs
         * param: event is an instance that capture every information about the event
         */

        // basic checks for null safety
        AccessibilityNodeInfo source = event.getSource();
        Log.d("dddd", event.getEventType() + "  " + event.getSource());
        if (source == null && event.getEventType() == AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED) {
            return;
        }

        if(previousSource!= null &&  source != null && (previousSource.hashCode() == source.hashCode() || previousSource.equals(source))){
            Log.d(debugLogTag, "Blocked event");

            return;
        }

        if(previousEventCode.equals("32") && previousSource!= null && event.getEventType() == AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED){
            Log.d(debugLogTag, "Overlapped event");
            return;
        }

        previousSource = source;
        previousEventCode = event.getEventType() + "";

        if(isNotBlockedEvent()) {
            checkSettingsChanged();
            uiElements.clear();
            noOfLabels = 0;
            removeAllTooltips();    // remove all old  tooltip when screen changed
            currentSource = getRootInActiveWindow(); // update the current root node

            if(isVoiceCommandConnected && currentSource != null)  {
                printOutAllClickableElement(getRootInActiveWindow(), 0, event); // call function for root node
                uiElements.remove("blocked numbers storage");
            }
            Log.d("test", uiElements.toString());
            this.executeCommand("");
        }
    }
    public void checkSettingsChanged(){
        sharedPreferences = getSharedPreferences(FILE_NAME,0);
        tooltipColor = sharedPreferences.getInt(SettingsActivity.TOOLTIP_COLOR,0);
        tooltipSize = sharedPreferences.getInt(SettingsActivity.TOOLTIP_SIZE,0);
        tooltipOpacity = sharedPreferences.getInt(SettingsActivity.TOOLTIP_OPACITY,0);
        int previousBtnOpacity = buttonOpacity;
        int previousBtnRecord = buttonRecordTxt;
        int previousBtnAlgo = buttonAlgoTxt;
        buttonOpacity = sharedPreferences.getInt(SettingsActivity.BUTTON_OPACITY,0);
        buttonRecordTxt = sharedPreferences.getInt(SettingsActivity.BUTTON_RECORD,0);
        buttonAlgoTxt = sharedPreferences.getInt(SettingsActivity.BUTTON_ALGO,0);
        if(previousBtnOpacity != buttonOpacity ||buttonRecordTxt != previousBtnRecord){
            wm.removeView(mLayout);
            createSwitch();
        }
    }

    final Handler autoReloadHandler = new Handler();
    final Runnable runnable = new Runnable() {
        public void run() {
            autoReload();
            autoReloadHandler.postDelayed(this, 3000);

        }
    };



    public boolean isNotBlockedEvent(){
        Date date = new Date();
        long time = date.getTime();
        if (time - currentTime > 750){
            currentTime = time;
            return true;
        }
        return false;
    }

    public void autoReload(){
        AccessibilityManager manager = (AccessibilityManager) getSystemService(Context.ACCESSIBILITY_SERVICE);
        if (manager.isEnabled()) {

            AccessibilityEvent e = AccessibilityEvent.obtain();
            e.setEventType(AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED);
            e.setClassName(getClass().getName());
                   //Log.d(debugLogTag, "Auto Reloaded");
        }
    }
    public void printOutAllClickableElement(AccessibilityNodeInfo nodeInfo, int depth, AccessibilityEvent event){
        /**
         * This function will print out all clickable element, storing the data it has or number for those
         * clickable elements.
         *
         */



        if (nodeInfo == null) {
            return;
        }
        if(nodeInfo.isClickable()){
            String label = "";
            Rect rectTest = new Rect();                     //  to get the coordinate of the UI element
            nodeInfo.getBoundsInScreen(rectTest);           //  store data of the node
            if(rectTest.right -100 < width && rectTest.bottom-100<height && rectTest.left+100 > 0 && rectTest.top+100 >0){
                if (nodeInfo.getText() != null) {   // check if node has a corresponding text
                    label += nodeInfo.getText();
                    uiElements.add(label.toLowerCase());
                    return;
                } else {
                    // no information about node or event (Tags to be assigned!)
                    String foundLabel  = searchForTextView(nodeInfo,"");
                    String[] texts = foundLabel.split(" ");
                    int end = texts.length;
                    Set<String> cleanTexts = new HashSet<String>();

                    for(int i = 0; i < end; i++){
                        cleanTexts.add(texts[i]);
                    }
                    String finalText = "";
                    for (String cleanText : cleanTexts) {
                        finalText += cleanText + " ";
                    }

                    if (!foundLabel.equals("") && noOfLabels < 15 && cleanTexts.size() < 10){
                        foundLabeledNodes.add(new LabelFoundNode(nodeInfo,foundLabel.toLowerCase()));
                        uiElements.add(foundLabel.toLowerCase());

                        noOfLabels+=1;
                    } else if (currentTooltipCount < 20){
                         inflateTooltip((rectTest.right+rectTest.left + 15) /2, (rectTest.bottom +rectTest.top - 70) /2, nodeInfo);    // call function to create number tooltips
                    }
                }
            }

            //clickableNodes.add(new Pair<>(label,nodeInfo));
             //Log.d(debugLogTag,"Available commands: " + label);
        }
        for (int i = 0; i < nodeInfo.getChildCount(); ++i) {
            printOutAllClickableElement(nodeInfo.getChild(i), depth + 1, event);    // recursive call
        }
    }

    public String filterDuplicateWord(String inputString){
        String[] allWords;
        String result = "";

        // Split the given sentence to get each word as String array
        allWords = inputString.split(" ");
        // Use for loop to remove duplicate words
        for(int i=0; i<allWords.length; i++) {
            for(int j=i+1; j<allWords.length; j++) {
                if(allWords[i].equals(allWords[j])) {
                    allWords[j] = "$##@$@#";
                }
            }
        }
        // Convert to String
        for(String word: allWords) {
            if(!word.equals("$##@$@#")) {
                result = result + word + " ";
            }
        }

        return result;
    }

    private void inflateTooltip(int x, int y, AccessibilityNodeInfo nodeInfo){
        /**
         * This function will configure each of the tooltip on the screen, so this function will be
         * called for each of the tooltip on the screen.
         * param: x is the location in x axis
         * param: y is the location in y axis
         */
        FrameLayout tooltipLayout = new FrameLayout(this);      // create new layout for each tooltip
        WindowManager.LayoutParams tooltipLayoutParams = new WindowManager.LayoutParams();
        tooltipLayoutParams.type = WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY;
        tooltipLayoutParams.format = PixelFormat.TRANSLUCENT;
        tooltipLayoutParams.flags |= WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE;
        tooltipLayoutParams.width = WindowManager.LayoutParams.WRAP_CONTENT;
        tooltipLayoutParams.height = WindowManager.LayoutParams.WRAP_CONTENT;
        tooltipLayoutParams.gravity = Gravity.TOP|Gravity.START;     // reset the (0,0) to the top left screen
        tooltipLayoutParams.x = x + 15;       // x location
        tooltipLayoutParams.y = y + 40;       // y location
        LayoutInflater inflater = LayoutInflater.from(this);
        inflater.inflate(R.layout.tooltip_number, tooltipLayout);   // inflate the view to the screen
        wm.addView(tooltipLayout, tooltipLayoutParams);

        TextView tooltip = tooltipLayout.findViewById(R.id.tooltip);    // set the count based on current count
        tooltip.setText(currentTooltipCount + "");
        tooltip.setTextSize(tooltipSizeSpinnerItems[tooltipSize]);
        tooltip.setBackgroundResource(R.drawable.tooltip_shape);  //drawable id
        GradientDrawable gd = (GradientDrawable) tooltip.getBackground().getCurrent();
        gd.setColor(Color.parseColor(tooltipColorSpinnerItems[tooltipColor])); //set color
        gd.setAlpha(tooltipOpacitySpinnerItems[tooltipOpacity]);        // add to the list to retrieve later
        gd.setSize(tooltipSizeSpinnerItems[tooltipSize] + 40,tooltipSizeSpinnerItems[tooltipSize]+ 5);
        tooltipRequiredNodes.add(new TooltipRequiredNode(nodeInfo,currentTooltipCount,tooltipLayout));
       //change
        //uiElements.add(Integer.toString(currentTooltipCount));
        currentTooltipCount += 1;

    }



    private void initializeSpeechRecognition() {
        /**
         * This function performs all the steps required for speech recognition initialisation
         */

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        speechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        // also available: LANGUAGE_MODEL_WEB_SEARCH
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
        // setting the limit for the service to listen as an requirement for API30
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 100000);

        speechRecognizer.setRecognitionListener(new RecognitionListener() {

            @Override
            public void onReadyForSpeech(Bundle params) {
            //    Log.d("ss", "onReady");
               // textMsg.setText("Ready...");
               // textMsg.setBackgroundResource(R.color.green);
                // Called when the endpointer is ready for the user to start speaking.
            }

            @Override
            public void onBeginningOfSpeech() {
            //    Log.d("ss", "onBeginning");
                // The user has started to speak.
               // textMsg.setText("Listening...");
               // textMsg.setBackgroundResource(R.color.green);
            }

            @Override
            public void onRmsChanged(float rmsdB) {
                // The sound level in the audio stream has changed.
            }

            @Override
            public void onBufferReceived(byte[] buffer) {
                //Log.d("ss", "buffer");
                // More sound has been received.
            }

            @Override
            public void onEndOfSpeech() {
               // Log.d("ss", "onEndOfSpeech");
                // Called after the user stops speaking
            }

            @Override
            public void onError(int error) {
                //Log.d("ss", "onError: " + error);
                Log.d("hihi", error + " error");
                //Toast.makeText(MainActivity.this, "An error has occurred. Code: " + Integer.toString(error), Toast.LENGTH_SHORT).show();
                if(error == 8 || error == 7) {
                    speechRecognizer.cancel();
                    if(isVoiceCommandConnected) {
                        speechRecognizer.startListening(speechRecognizerIntent);
                    }
                }
            }

            @Override
            public void onResults(Bundle results) {
                // Called when recognition results are ready.

             //   textMsg.setText("Processing.");
             //   textMsg.setBackgroundResource(R.color.yellow);
                if(!isVoiceCommandConnected) {
                    return;
                }
                ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);

                if (matches!=null && matches.size() > 0) {
                    String match = matches.get(0);
  //                  inputTxt.setText(match);
                    Log.d(debugLogTag,match);
                    if(match.contains("show me")) {
                        Log.d("duc","sdds");
                        Intent intent = new Intent(Intent.ACTION_VIEW);
                        intent.setComponent(new ComponentName("com.cookware.worldcusinerecipes", "com.cookware.worldcusinerecipes.MyRecipesGridActivity"));
                        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                        startActivity(intent);
                    } else{
                        executeCommand(match);
                    }



                }

                speechRecognizer.startListening(speechRecognizerIntent);
            }


            @Override
            public void onPartialResults(Bundle partialResults) {
            //    textMsg.setText("Processing.");
            //    textMsg.setBackgroundResource(R.color.yellow);

                ArrayList<String> matches = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches!=null && matches.size() > 0) {
                    String match = matches.get(0);
//                    inputTxt.setText(match);
                    Log.d(debugLogTag,match);
                    if(match.contains("show me")) {
                        Log.d("duc","sdds");
                        Intent intent = new Intent(Intent.ACTION_VIEW);
                        intent.setComponent(new ComponentName("com.cookware.worldcusinerecipes", "com.cookware.worldcusinerecipes.MyRecipesGridActivity"));
                        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                        startActivity(intent);
                    } else{
                        executeCommand(match);
                    }



                }
            }

            @Override
            public void onEvent(int eventType, Bundle params) {
                // reserved by android for future events
            }
        });
    }

    private String currentNumberOutput(int currentNumberTooltip) {
        String output = "";
        for (int i = 1; i <= currentNumberTooltip; i ++) {
            output += i + ", ";
        }
        return output;
    }

    private String formatNumberInCommand(String sentence){
        String[] numbers = {" zero", " one", " two"," three"," four"," five"," six", " seven", " eight"," nine"," ten"," eleven"," twelve"," thirteen"," fourteen"," fifteen"," sixteen"," seventeen"," eighteen"," nineteen"};
        sentence = sentence.replaceAll(" number","").replaceAll("click to","click two").replaceAll("press to","click two").replaceAll("for", "four").replaceAll("want", "one").replaceAll("sex","six");
        sentence = sentence.replaceAll("pressed","press").replaceAll("clique","click").replaceAll("quick","click").replaceAll("Preston", "press");
        for(int i = 0; i< numbers.length;i++){
            if(sentence.contains(numbers[i])){
                sentence = sentence.replaceAll(numbers[i]," " +String.valueOf(i));
            }
        }
        return sentence;
    }

    private void executeCommand(String sentence){
        sentence = formatNumberInCommand(sentence);
        sentence = sentence.trim();


        String currentCommand = "";
        if(!hasExecuted) {
            return;
        }


        if(sentence.equals("") && currentSequence.size() == 0) {
            skipPreviousRTECheck = true;
            return;
        }

        if(!sentence.equals("")){
            skipPreviousRTECheck = true;
            String[] commands = sentence.split("and");
            ArrayList<String> outputModelCommand = new ArrayList<String>();
            Collections.addAll(currentSequence, commands);
        }

        if(!skipPreviousRTECheck){
            if(uiElements.equals(previousUIElements)) {
                Log.d(debugLogTag, "No change in RTE: " + uiElements.toString() + ", " + previousUIElements.toString());
                return;
            }
        }

        String[] words = sentence.split(" ");
        previousUIElements.clear();
        previousUIElements.addAll(uiElements);
        skipPreviousRTECheck = false;
        boolean isExecuted = false;

        if(currentSequence.size() != 0) {
            //extract command
            currentCommand = currentSequence.remove(0);

            if (currentCommand.contains("back")) {
                performGlobalAction(GLOBAL_ACTION_BACK);

            } else if (currentCommand.contains("home")) {
                performGlobalAction(GLOBAL_ACTION_HOME);
            }


            ArrayList<String> clickPhrase = new ArrayList<>(Arrays.asList("click", "tap", "touch", "press"));
            ArrayList<String> scrollPhrase = new ArrayList<>(Arrays.asList("swipe", "scroll"));
            ArrayList<String> enterPhrase = new ArrayList<>(Arrays.asList("enter", "type", "input"));
            ArrayList<String> openPhrase = new ArrayList<>(Arrays.asList("open"));
            String parameters = "";

            Log.d("duc", Arrays.toString(words) + "");
            if (words.length == 1) {
                isExecuted = clickButtonByText(words[0].trim());
            } else if (words.length > 1) {
                for (int i = 1; i < words.length; i++) {
                    parameters += words[i].toLowerCase(Locale.ROOT) + " ";
                }
                parameters = parameters.trim();
                if (clickPhrase.contains(words[0].toLowerCase(Locale.ROOT))) {
                    isExecuted = clickButtonByText(parameters);
                    skipPreviousRTECheck = true;
                } else if (scrollPhrase.contains(words[0].toLowerCase(Locale.ROOT))) {
                    if (words[0].toLowerCase(Locale.ROOT).equals("scroll")){
                        isExecuted = scrollingActivity(parameters);
                        skipPreviousRTECheck = true;
                    } else {
                        switch(parameters){
                            case "down":
                                isExecuted = scrollingActivity("up");
                                break;
                            case "up":
                                isExecuted = scrollingActivity("down");
                                break;
                            case "left":
                                isExecuted = scrollingActivity("right");
                                break;
                            case "right":
                                isExecuted = scrollingActivity("left");
                                break;
                        }

                        skipPreviousRTECheck = true;
                    }

                } else if (openPhrase.contains(words[0].toLowerCase(Locale.ROOT))) {
                    isExecuted = openApp(parameters);
                    skipPreviousRTECheck = true;
                } else if (enterPhrase.contains(words[0].toLowerCase(Locale.ROOT))) {
                    setTextForAllSubNode(currentSource, 0, parameters);
                    skipPreviousRTECheck = true;
                    isExecuted = true;
                }
            }

            if (isExecuted) {
                Log.d("ss", "locally executed");
                return;
            }

            String command = currentCommand + " ||| " + uiElements.toString().replace("[","").replace("]","");
            Log.d("hihi",command);
            Call<ResponseObject> call = jsonApi.getData(command);
            call.enqueue(new Callback<ResponseObject>() {
                @Override
                public void onResponse(Call<ResponseObject> call, Response<ResponseObject> response) {
                    if (!response.isSuccessful()) {
                        assert response.errorBody() != null;
                        Log.e("myErrTag", "not successful");
                        return;
                    }
                    ResponseObject data = response.body();
                    if (data != null && data.hypotheses != null && data.hypotheses.size() > 0) {

                        Log.d("ggg",data.hypotheses.get(0).value);
                        String rawData = data.hypotheses.get(0).value.replace(" ","");
                        String[] modelOutput = rawData.substring(1,rawData.length() - 1).split(",");
                        if (modelOutput.length == 2){
                            String action = modelOutput[0].trim();
                            String target = modelOutput[1].trim().replace("_"," ");

                            if (target.startsWith("app:")) {
                                target = target.substring(4);
                            }
                            switch (action) {
                                case "PRESS":
                                    clickButtonByText(target);
                                    break;
                                case "SWIPE":
                                    scrollingActivity(target);
                                    break;
                                case "OPEN":
                                    openApp(target);
                                    break;
                                case "ENTER":
                                    setTextForAllSubNode(currentSource,0, target);
                                    break;
                                default:
                                    Log.d("duc", action + "/" + target);
                            }
                        } else if (modelOutput.length == 3) {
                            String action = modelOutput[0].trim();
                            String component = modelOutput[1].trim();
                            String app =  modelOutput[2].trim();

                            if (app.startsWith("app:")) {
                                app = app.substring(4);
                            }

                            if (component.startsWith("component:")) {
                                component = component.substring(10);
                            }
                            invokeComponent(app,component);

                        }

                    }
                }
                @Override
                public void onFailure(Call<ResponseObject> call, Throwable t) {
                    Log.e("myErrTag", t.getMessage());
                }
            });


            try{
                java.util.concurrent.TimeUnit.SECONDS.sleep(1); // emulates thread synchronisation
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public void getScrollableNode(AccessibilityNodeInfo currentNode){
        /**
         * Get all the scrollable node in the current screen.
         * @param: currentNode: the current node that is being checked ( start from root node and recursively for all node)
         */
        if (currentNode == null) return;
        if(currentNode.isClickable()){
            scrollableNodes.add(currentNode);
        }
        for (int i = 0; i < currentNode.getChildCount(); ++i) {
            getScrollableNode(currentNode.getChild(i));    // recursive call
        }
    }

    public String searchForTextView(AccessibilityNodeInfo currentNode, String allTexts){
        String concatenatedString = allTexts;
        if(currentNode == null || concatenatedString.split(" ").length > 5) return concatenatedString;

        if( currentNode.getClassName() != null && currentNode.getClassName().equals("android.widget.TextView") && currentNode.getText() != null){
            concatenatedString += currentNode.getText().toString() + " ";
        } else {
            for (int i = 0; i < currentNode.getChildCount(); ++i) {
                concatenatedString += searchForTextView(currentNode.getChild(i),concatenatedString);    // recursive call
            }
        }

        return concatenatedString;
    }


    public void setTextForAllSubNode(AccessibilityNodeInfo nodeInfo, int depth, String text)  {
        /**
         * This function will set text for all sub-node ( all element on the screen)
         * @param: nodeInfo : current node that this function is called on ( will start from root node)
         * @param: depth : the current level of leaf
         * @param: text: the passed in text for writing in the edit text field.
         */
        if (nodeInfo == null) return;   // null check
        if(nodeInfo.isEditable()){      // check if the node has editable field
            setGivenText(nodeInfo,text);       // call a method to put in the text
            Log.d("duc",nodeInfo + "");
        }
        for (int i = 0; i < nodeInfo.getChildCount(); ++i) {    // recursive call to reach all nested nodes/leaves
            setTextForAllSubNode(nodeInfo.getChild(i), depth + 1, text);
        }
    }

    public boolean scrollingActivity(String command){
        /**
         * This function will work as scrolling the screen for user on invocation.
         * @param: a string - can be up or down specifying the scrolling direction.
         */
        boolean returnVal = false;
        getScrollableNode(currentSource);   // get all scrollable not within current screen
        if(scrollableNodes.size() == 0) {
            Log.d(debugLogTag,"Can't find item to scroll");
            return false;
        }
        else {      // if there exist item to be scrolled.
            for(AccessibilityNodeInfo node: scrollableNodes){
                DisplayMetrics displayMetrics = getResources().getDisplayMetrics();
                // scrolling using gesture builder.
                final int height = displayMetrics.heightPixels;
                final int top = (int) (height * .25);
                final int mid = (int) (height * .5);
                final int bottom = (int) (height * .75);
                final int midX = displayMetrics.widthPixels / 2;
                final int width = displayMetrics.widthPixels;
                final int left = (int) (width * 0.25);
                final int right = (int) (width * 0.75);

                GestureDescription.Builder gestureBuilder = new GestureDescription.Builder();
                Path path = new Path();
                command = command.toLowerCase().trim();

                // Scroll up
                if (command.contains("up")) {
                    path.moveTo(midX, mid);
                    path.lineTo(midX, bottom);
                    returnVal = true;
                    // Scroll down
                } else if (command.contains("down")){
                    path.moveTo(midX, mid);
                    path.lineTo(midX, top);
                    returnVal = true;
                } else if (command.contains("right")) {
                    path.moveTo(right, mid);
                    path.lineTo(left, mid);
                    returnVal = true;
                } else if (command.contains("left")) {
                    path.moveTo(left, mid);
                    path.lineTo(right, mid);
                    returnVal = true;
                }

                gestureBuilder.addStroke(new GestureDescription.StrokeDescription(path, 100, 300));
                dispatchGesture(gestureBuilder.build(), new GestureResultCallback() {
                    @Override
                    public void onCompleted(GestureDescription gestureDescription) { // gesture execution
                        //Log.d(debugLogTag,"Gesture Completed");
                        super.onCompleted(gestureDescription);
                    }
                }, null);
            }
        }
        return returnVal;
    }

    @Override
    public void onInterrupt() {
        Log.d("Service Test","Service Disconnected");
    }

    @Override
    protected void onServiceConnected() {
        /**
         * This function is invoked after the accessibility service has been stared by the user. this
         * function inflates the layout and draws the floating UI for the service. It also initialises
         * speech recognition & checks audio permissions.
         *
         * @param: None
         * @return: None
         * @post-cond: A button floating on top of the screen can be used to control the service
         *             by the user if the app have all the permissions it needs. Else opens settings
         *             page with the app's details.
         * */

        super.onServiceConnected();
        wm  = (WindowManager) getSystemService(WINDOW_SERVICE);
        Date date = new Date();
        currentTime = date.getTime();
        createSwitch();
        initializeSpeechRecognition();                      // Checking permissions & initialising speech recognition
        loadData();
        loadAPIConnection();
        getDisplayMetrics();
        Log.d(debugLogTag,"Service Connected");
        loadAppNames();
        //createText2VecModel();

        autoReloadHandler.post(runnable);
    }

    void loadAPIConnection(){
        Gson gson = new GsonBuilder()
                .setLenient()
                .create();
        retrofit = new Retrofit.Builder()
                .baseUrl("http://localhost:8099/")
                .addConverterFactory(GsonConverterFactory.create(gson))
                .build();
        jsonApi = retrofit.create(JsonApi.class);
    }
    void loadData(){

        mPrefs = getSharedPreferences(FILE_NAME,0);
        Gson gson = new Gson();
        String json = mPrefs.getString("packageDataObjects", "");
        Type type = new TypeToken<ArrayList<PackageDataObject>>(){}.getType();
        packageDataObjects = gson.fromJson(json, type);
    }

    void deleteIntent(){
        Log.d("duc",currentOpeningFeature +"// "+currentOpeningPackage);
        if(packageDataObjects.size() >0){
            for(PackageDataObject packageDataObject: packageDataObjects){
                if(packageDataObject.packageName.equals(currentOpeningPackage)){
                    String[] dataArrIntent = new String[matchedIntents.size()];
                    dataArrIntent = matchedIntents.toArray(dataArrIntent);
                    String currentConcatenatedIntent = dataArrIntent[currentIntentIndex];
                    String[] intentComponent = currentConcatenatedIntent.split("  ");

                    ArrayList<String> intentActionList = packageDataObject.intentsByActivity.get(intentComponent[1]);
                    assert intentActionList != null;
                    boolean remove = intentActionList.remove(intentComponent[0]);
                    Log.d("duc",currentConcatenatedIntent);

                    packageDataObject.intentsByActivity.put(intentComponent[1],intentActionList);
                    if(remove){
                        Log.d("duc","Removed from the intent database");
                    }
                    SharedPreferences.Editor prefsEditor = mPrefs.edit();
                    Gson gson = new Gson();
                    String json = gson.toJson(packageDataObjects);
                    prefsEditor.putString("packageDataObjects", json);
                    prefsEditor.commit();
                    invokeComponent(currentAppName,currentOpeningFeature);
                    break;
                }
            }
        }

    }
    private void getDisplayMetrics() {
        DisplayMetrics metrics = getApplicationContext().getResources().getDisplayMetrics();
        width = metrics.widthPixels;
        height = metrics.heightPixels;
    }



    private void createSwitch(){
        /**
         * This code will create a layout for the switch. This code is called whenever service is
         * connected and will be gone when service is shutdown
         *
         */

        // Check for permissions
        int LAYOUT_FLAG;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            LAYOUT_FLAG = WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY;
        } else {
            LAYOUT_FLAG = WindowManager.LayoutParams.TYPE_PHONE;
        }
        mLayout = new FrameLayout(this);

        // Create layout for switchBar
        switchBar = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                LAYOUT_FLAG,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT);
        switchBar.gravity = Gravity.TOP;  // stick it to the top
        //WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN | WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL |


        LayoutInflater inflater = LayoutInflater.from(this);
        View actionBar = inflater.inflate(R.layout.action_bar, mLayout);
        wm.addView(mLayout, switchBar);       // add it to the screen



        listenButton = mLayout.findViewById(R.id.listenBtn);
       // textMsg = mLayout.findViewById(R.id.msg);
      //  inputTxt = mLayout.findViewById(R.id.inputTxt);
      //  inputTxt.setBackgroundResource(R.color.black);
        listenButton.setBackgroundResource(R.drawable.start_btn);
        configureListenButton();

    }



    // This method is responsible for updating the switchBar coordniates upon touch and updating the view
    @Override
    public boolean onTouch(View view1, MotionEvent motionEvent){

        switch(motionEvent.getAction()){

            case MotionEvent.ACTION_DOWN:
                initialX = switchBar.x;
                initialY = switchBar.y;
                initialTouchX = motionEvent.getRawX();
                initialTouchY = motionEvent.getRawY();
                break;

            case MotionEvent.ACTION_UP:
                break;

            case MotionEvent.ACTION_MOVE:
                switchBar.x = initialX + (int) (motionEvent.getRawX() - initialTouchX);
                switchBar.y = initialY + (int) (motionEvent.getRawY() - initialTouchY);
                wm.updateViewLayout(mLayout, switchBar);
                break;
        }
        return false;
    }

    private void removeAllTooltips(){
        /**
         * This function will be called when something changed on the screen, reset all tooltips.
         *
         */
        for (TooltipRequiredNode tooltip: tooltipRequiredNodes){    // remove the list of current tooltips
            if(tooltip.tooltipLayout != null)
                wm.removeView(tooltip.tooltipLayout);   // remove them from the screen
        }
        // reset all variables when changing to new screen.
        currentTooltipCount = 1;
        tooltipRequiredNodes.clear();
        foundLabeledNodes.clear();
    }

    private void configureListenButton() {
        /**
         * This function is called after the service has been connected. This function binds
         * functionality to the master button which can be used to turn on/off the tool.
         *
         * @param: None
         * @return: None
         * @post-cond: functionality has been added to the inflated button
         * */

        listenButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (listenButton.getText().toString().equalsIgnoreCase("start")){
                    listenButton.setText("Stop");
                    listenButton.setBackgroundResource(R.drawable.stop_btn);
                    isVoiceCommandConnected = true;
                    speechRecognizer.startListening(speechRecognizerIntent);       // on click listener to start listening audio
                } else {
                    listenButton.setText("Start");

                    listenButton.setBackgroundResource(R.color.transparent);
                    isVoiceCommandConnected = false;
                    listenButton.setBackgroundResource(R.drawable.start_btn);
                    speechRecognizer.stopListening();           // on click listener to stop listening & processing data
           //         textMsg.setText("");
                }
            }
        });
    }


    private void configureConfirmButton(Button listenBtn) {

        listenBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                deleteIntent();
            }
        });
    }
    private boolean openApp(String inputName) {
        /**
         * This function is used to check if the given string matches with any applications that the
         * user may have installed. It launches apps that have matched. Current matching algorithm is
         * trivial. (WIP: Improved Matching Algorithm)
         *
         * @param: inputName — This is a String that is supposed to be checked for app name matching
         * @return: None
         * @post-cond: Apps that match with the given string are launched and presented on the
         *             foreground adding them to the system backstack if multiple apps are launched.
         * */

        final PackageManager pm = getPackageManager();
        List<ApplicationInfo> packages = pm.getInstalledApplications(PackageManager.GET_META_DATA); // getting meta data of all installed apps

        for (ApplicationInfo packageInfo : packages) {          // checking if the input has a match with app name
            try {
                ApplicationInfo info = pm.getApplicationInfo(packageInfo.packageName, PackageManager.GET_META_DATA);
                String appName = (String) pm.getApplicationLabel(info).toString().toLowerCase();
                if(appName.equals(inputName)){
                    Intent mIntent = getPackageManager().getLaunchIntentForPackage(
                            packageInfo.packageName);
                    if (mIntent != null) {
                        try{
                            startActivity(mIntent);
                            return true;
                            // Adding some text-to-speech feedback for opening apps based on input
                            // Text-to-speech feedback if app not found);
//                            speakerTask(speechPrompt.get("open") + inputName);

                        } catch (ActivityNotFoundException err) {
                            // Text-to-speech feedback if app not found
//                            speakerTask(speechPrompt.get("noMatch") + inputName);

                            // Render toast message on screen
                            Toast t = Toast.makeText(getApplicationContext(),
                                    "APP NOT FOUND", Toast.LENGTH_SHORT);
                            t.show();
                        }
                    }
                }
            } catch (PackageManager.NameNotFoundException e) {
                e.printStackTrace();                // handling app not found exception
//                speakerTask(speechPrompt.get("noMatch") + inputName);
            }
        }

        return false;
    }

    private void loadAppNames() {
        final PackageManager pm = getPackageManager();
        List<ApplicationInfo> packages = pm.getInstalledApplications(PackageManager.GET_META_DATA); // getting meta data of all installed apps

        for (ApplicationInfo packageInfo : packages) {          // checking if the input has a match with app name
            try {
                ApplicationInfo info = pm.getApplicationInfo(packageInfo.packageName, PackageManager.GET_META_DATA);
                String appName = (String) pm.getApplicationLabel(info).toString().toLowerCase();
                this.appNames.add(appName);
            } catch (PackageManager.NameNotFoundException e) {
                e.printStackTrace();                // handling app not found exception
            }
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        int result = super.onStartCommand(intent, flags, startId);
        if(intent != null){
            String appName = intent.getStringExtra("app_name");
            String componentName = intent.getStringExtra("component_name");
            Log.d("duc",appName + " " +componentName);
            invokeComponent(appName,componentName);

        }

        return result ;
    }

    public void speakerTask(String toSpeak) {
        /**
         * Use this method to call out to TTSService (Text-To-speech service) to speak out message
         * param: a string to be spoken by the Text-to-speech service
         */
        Intent i = new Intent(this, TTSService.class);
        i.putExtra("message", toSpeak);
        // starts service for intent
        startService(i);
    }

    boolean clickButtonByText(String word) {
        /**
         * This function will click a button (anything thats clickable) with provided information
         * param: word: a string to store data about what to click
         */
        // Processes input first to determine if number label was called
        // More efficient number label processing? Skips iterating through array of numbers and assumes the array is numerical order if input is a Digit
        Log.d("dddf",word);
        if(word.equals("")){
            return false;
        }

        // check
        Log.d("dsds", foundLabeledNodes + "");
        for(LabelFoundNode node : foundLabeledNodes){
            if(node.label.contains("burger")){
                node.nodeInfo.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                return true;
            }
        }

        if (TextUtils.isDigitsOnly(word)) {
            //Log.d(debugLogTag,word);
            if (Integer.parseInt(word) <= currentTooltipCount - 1){
                if (tooltipRequiredNodes.size() >= Integer.parseInt(word) && tooltipRequiredNodes.get(Integer.parseInt(word)-1).nodeInfo.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                    //Log.d(debugLogTag, "Clicked number: " + word);    // log the information
                    return true;
                }

            }
        }

//        LabelFoundNode bestNode = null;
//        double maxScore = 0.5;
//        for(LabelFoundNode foundLabeledNode: foundLabeledNodes){
//            double concatenatedScore = 0;
//            for (String spokenWord : word.split(" ")) {
//                double bestMatch = 0;
//                for(String labelWord: foundLabeledNode.label.split(" ")){
//                    double currentScore = model.similarity(spokenWord,labelWord);
//                    if ((currentScore > bestMatch) && (currentScore > 0.85)) {
//                        bestMatch = currentScore;
//                    }
//                }
//                concatenatedScore += bestMatch;
//            }
//           if (concatenatedScore > maxScore) {
//               Log.d("ddd", concatenatedScore + " "+ maxScore + " " + foundLabeledNode.label + " " + word);
//               maxScore = concatenatedScore;
//               bestNode = foundLabeledNode;
//           }
//        }
//        if(bestNode != null){
//            if (bestNode.nodeInfo.performAction(AccessibilityNodeInfo.ACTION_CLICK)){
//                Log.d("ddd", "Press " + bestNode.label);
//                //Log.d(debugLogTag, "Clicked on description:" + word);
//                return true;
//            }
//            // return once clicked
//        }

        if (currentSource == null) {return false;}

        for(LabelFoundNode foundLabeledNode: foundLabeledNodes){
            Log.d("hehe",foundLabeledNode.label);
            if(foundLabeledNode.label.contains(word)){

                if (foundLabeledNode.nodeInfo.performAction(AccessibilityNodeInfo.ACTION_CLICK)){
                    //Log.d(debugLogTag, "Clicked on description:" + word);
                    return true;
                }
                // return once clicked
            }
        }

        //Find ALL of the nodes that match the "text" argument.
        List<AccessibilityNodeInfo> list = currentSource.findAccessibilityNodeInfosByText(word);    // find the node by text
        for (final AccessibilityNodeInfo node : list) { // go through each node to see if action can be performed
            if (node.performAction(AccessibilityNodeInfo.ACTION_CLICK)){
                return true;     // return once clicked
            }

        }
        // for some element that named with first capital word
        String camelCaseWord = word.substring(0, 1).toUpperCase() + word.substring(1);
        list = currentSource.findAccessibilityNodeInfosByText(camelCaseWord);    // find the node by text
        for (final AccessibilityNodeInfo node : list) { // go through each node to see if action can be performed
            if (node.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                return true;     // return once clicked
            }

        }


        return false;
    }

    public boolean commandExecution(String match){
        /**
         * This function will be called on user voice input as a string of command
         * @param: match : the user command interpreted into a string
         */
        String[] words = match.split(" ");
        ArrayList<String> trimmedWords = new ArrayList<String>();
        // ["open","uber"...]
        //Log.d(debugLogTag,match);


        for (int index=0; index< words.length; index++) {
            String word = words[index].toLowerCase().trim();
            trimmedWords.add(word);
        }
        boolean isActionInvoked = false;
        String initialWord = trimmedWords.get(0) ; // first word from the command
        if (initialWord.equals("back")){
            performGlobalAction(GLOBAL_ACTION_BACK);
            isActionInvoked = true;
        } else if (initialWord.equals("home")){
            performGlobalAction(GLOBAL_ACTION_HOME);
            isActionInvoked = true;
        } else if(launchTriggers.contains(initialWord)){
            isActionInvoked = true;
            openApp(match);
        } else if (initialWord.equals("enter")) {
            String textToSet = match.substring(6);
            setTextForAllSubNode(currentSource,0,textToSet);
            isActionInvoked = true;
        }
        else if (initialWord.equals("scroll")){
            scrollingActivity(trimmedWords.get(1));
            isActionInvoked = true;
        } else if(!trimmedWords.get(0).equals("")){
            if (clickButtonByText(trimmedWords.get(0))) {
                isActionInvoked = true;
            }
        }
        if (isActionInvoked){
            if(isRecording){
                savedCommands.add(match);
            }
            AccessibilityManager manager = (AccessibilityManager) getSystemService(Context.ACCESSIBILITY_SERVICE);
            if (manager.isEnabled()) {
                AccessibilityEvent e = AccessibilityEvent.obtain();
                e.setEventType(AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED);
                e.setClassName(getClass().getName());
                e.getText().add("User interaction invoked this event");
                manager.sendAccessibilityEvent(e);
            }
            return true;
        }
        return false;
    }

    // Enter action
    public void setGivenText(AccessibilityNodeInfo currentNode, String text){
        /**
         * This function will set the text for a given node
         * @param: currentNode: the node to store information about object that will be inserted the text.
         * @param: text: the customized passed in text to be written in the field.
         */
        Bundle arguments = new Bundle();
        arguments.putCharSequence(AccessibilityNodeInfo
                .ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,text);
        currentNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments);
    }

// new code
    String getAppComponent(PackageDataObject packageDataObject, String referencedName){
        ArrayList<String> commonWords = new ArrayList<>(Arrays.asList("http", "deeplink", "com", "android","intent", "action", "google"));
        List<String> appWords = Arrays.asList(packageDataObject.packageName.split("\\."));
        ArrayList<String> components = new ArrayList<>();
        for(String deeplink: packageDataObject.deepLinks){
            String[] words = deeplink.split("[^\\w']+");
            String prefix = words[words.length -1];
            if(!appWords.contains(prefix) && !commonWords.contains(prefix)){
                components.add(prefix);
                if(referencedName.toLowerCase(Locale.ROOT).equals(prefix)){
                    return deeplink;
                }
            }
        }
        for(String intent: packageDataObject.getQuerySearch("")) {
            String[] words = intent.split("[^\\w']+");
            ArrayList<String> keywords = new ArrayList<>();
            for(String word: words){
                if(!appWords.contains(word) && !commonWords.contains(word)){
                    if (word.endsWith("Activity")){
                        //Log.d("nani", word);
                        keywords.add(word.replace("Activity",""));
                    } else if (word.endsWith("Launcher")){
                        //Log.d("nani", word);
                        keywords.add(word.replace("Launcher",""));
                    } else if (word.contains("_")){
                        //Log.d("nani", word);
                        word.replace("_"," ");
                    } else {
                        keywords.add(word);
                    }

                }
            }
            ArrayList<String> finalSet = new ArrayList<>();
            for (String keyword: keywords){
                String[] singleWords = keyword.split("(?<!(^|[A-Z]))(?=[A-Z])|(?<!^)(?=[A-Z][a-z])");
                String element = String.join(" ",singleWords);
                finalSet.add(element);
            }
            String componentName =String.join(" ",finalSet);
            components.add(componentName);
            if (componentName.toLowerCase(Locale.ROOT).equals(referencedName)){
                return intent;
            }
        }
        Log.d("Test",packageDataObject.name +": " + components);
        return "!@#!@#@!#!@#";
    }

    public void invokeComponent(String appSearchString, String featureSearchString) {
        appSearchString = appSearchString.replace("_"," ");
        featureSearchString = featureSearchString.replace("_", " ");
        boolean isAppFound = false;
        String componentOutput = "";
        PackageDataObject currentPackage = null;
        for(PackageDataObject packageDataObject: packageDataObjects){

            if(packageDataObject.name.replace("!","").equals(appSearchString)){
                componentOutput = getAppComponent(packageDataObject,featureSearchString);
                isAppFound = true;
                currentPackage = packageDataObject;
                break;
            }


        }
        if(isAppFound) {
            if (currentPackage.getDeepLinks().contains(componentOutput)) {
                Intent intent = new Intent(Intent.ACTION_VIEW);
                intent.setData(Uri.parse(componentOutput));
                try {
                    startActivity(intent);
                } catch (ActivityNotFoundException e) {
                    // Define what your app should do if no activity can handle the intent.

                    Log.d("Invocation Errors", e.getMessage());
                }
            } else if (currentPackage.getQuerySearch("").contains(componentOutput)) {
                String[] intents = componentOutput.split("  ");
                Intent intent = new Intent(intents[0]);
                intent.setComponent(new ComponentName(currentPackage.packageName, intents[1]));
                intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
//                try {
                    PackageManager packageManager = getPackageManager();

                    if (intent.resolveActivity(packageManager) != null) {
                        startActivity(intent);
//                        intentValidationHandler.postDelayed(intentExecutionValidation, 4000);
//                        Log.d("duc", componentOutput + "1111" + currentOpeningPackage + "11111" + intents[1]);
                    }
//                    else {
//                        Log.d("Duc", "No Intent available to handle action");
//                        currentIntentIndex += 1;
//                        invokeComponent(appSearchString, featureSearchString);
//                    }


//                } catch (Exception e) {
//                    currentIntentIndex += 1;
//                    // Define what your app should do if no activity can handle the intent
//                    Log.d("Duc", e.getMessage());
//                    invokeComponent(appSearchString, featureSearchString);
//                }
            } else {
                Intent mIntent = getPackageManager().getLaunchIntentForPackage(
                        currentPackage.packageName);

                if (mIntent != null) {
                    try{
                        Log.d("Duc", "Open app launcher screen instead");
                        startActivity(mIntent);
                        speakerTask("Component not found, opened the app instead");
                        // Adding some text-to-speech feedback for opening apps based on input
                        // Text-to-speech feedback if app not found);
        //                            speakerTask(speechPrompt.get("open") + inputName);

                    } catch (ActivityNotFoundException err) {
                        // Text-to-speech feedback if app not found
        //                            speakerTask(speechPrompt.get("noMatch") + inputName);

                        // Render toast message on screen
                        Toast t = Toast.makeText(getApplicationContext(),
                                "APP NOT FOUND", Toast.LENGTH_SHORT);
                        speakerTask("Cannot open the app");
                        t.show();
                    }
//                    currentIntentIndex = 0;
                }
            }
        }


    }

    public boolean isTransitionToNewApp(String packageName){

        Log.d("duc",  packageName + '/' +getRootInActiveWindow().getPackageName() );
        return packageName == currentSource.getPackageName();

    }


    Handler intentValidationHandler = new Handler();
    final Runnable intentExecutionValidation = new Runnable() {
        public void run() {
            Log.d("duc","Handler "+ getRootInActiveWindow().getPackageName());
            Log.d("duc","Handler "+ currentOpeningPackage);
            if(currentOpeningPackage.equals(getRootInActiveWindow().getPackageName())){

                Log.d("duc", "success");
                speakerTask("Opened the component");
            } else {
                currentIntentIndex +=1;
                currentIntentIndex +=1;
                Log.d("duc", "handler reinvoke");
                invokeComponent(currentAppName,currentOpeningFeature);
            }

        }
    };

    public void createText2VecModel() {
        try {
            InputStream file = getApplicationContext().getAssets().open("model.bin");
            File model_path = File.createTempFile( "model", ".bin" );
            FileUtils.copyToFile( file, model_path );
            model = loadGoogleBinaryModel(model_path, false);
            Log.d("ddd", "Done");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void verifyStoragePermission(Activity activity) {
        final int REQUEST_EXTERNAL_STORAGE = 1;
        String[] PERMISSIONS_STORAGE = {
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
        };
        // Get permission status
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission we request it
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }
    public static Word2Vec loadGoogleBinaryModel(File modelFile, boolean lineBreaks) throws IOException {
        return readBinaryModel(modelFile, lineBreaks, true);
    }

    private static Word2Vec readBinaryModel(File modelFile, boolean linebreaks, boolean normalize)
            throws NumberFormatException, IOException {
        InMemoryLookupTable<VocabWord> lookupTable;
        VocabCache<VocabWord> cache;
        INDArray syn0;
        int words, size;

        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();

        if (originalPeriodic)
            Nd4j.getMemoryManager().togglePeriodicGc(false);

        Nd4j.getMemoryManager().setOccasionalGcFrequency(50000);

        try (BufferedInputStream bis = new BufferedInputStream(GzipUtils.isCompressedFilename(modelFile.getName())
                ? new GZIPInputStream(new FileInputStream(modelFile)) : new FileInputStream(modelFile));
             DataInputStream dis = new DataInputStream(bis)) {
            words = Integer.parseInt(WordVectorSerializer.readString(dis));
            size = Integer.parseInt(WordVectorSerializer.readString(dis));
            syn0 = Nd4j.create(words, size);
            cache = new AbstractCache<>();

            WordVectorSerializer.printOutProjectedMemoryUse(words, size, 1);

            lookupTable = (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>().cache(cache)
                    .useHierarchicSoftmax(false).vectorLength(size).build();

            String word;
            float[] vector = new float[size];
            for (int i = 0; i < words; i++) {

                word = WordVectorSerializer.readString(dis);
                //Log.d("tagg","Loading " + word + " with word " + i);

                for (int j = 0; j < size; j++) {
                    vector[j] = WordVectorSerializer.readFloat(dis);
                }

                syn0.putRow(i, normalize ? Transforms.unitVec(Nd4j.create(vector)) : Nd4j.create(vector));

                // FIXME There was an empty string in my test model ......
                if (StringUtils.isNotEmpty(word)) {
                    VocabWord vw = new VocabWord(1.0, word);
                    vw.setIndex(cache.numWords());

                    cache.addToken(vw);
                    cache.addWordToIndex(vw.getIndex(), vw.getLabel());

                    cache.putVocabWord(word);
                }

                if (linebreaks) {
                    dis.readByte(); // line break
                }

                Nd4j.getMemoryManager().invokeGcOccasionally();
            }
        } finally {
            if (originalPeriodic)
                Nd4j.getMemoryManager().togglePeriodicGc(true);

            Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);
        }

        lookupTable.setSyn0(syn0);

        Word2Vec ret = new Word2Vec.Builder().useHierarchicSoftmax(false).resetModel(false).layerSize(syn0.columns())
                .allowParallelTokenization(true).elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .learningRate(0.025).windowSize(5).workers(1).build();

        ret.setVocab(cache);
        ret.setLookupTable(lookupTable);

        return ret;
    }


}


/*
* 1) Button for switching between AutoML & Custom Action-Target Matching algorithm
* 2) Function in VoiceToActionService to execute action on a target.
*       param - ArrayList<HashMap<Action,String>>
                eg: [{OPEN: "chrome"}]
  3)
* */

//    private void executeCommand(String sentence){
//
//        sentence = formatNumberInCommand(sentence);
//        Log.d("duc", sentence);
//
//
//        String currentCommand = "";
//        if(!hasExecuted) {
//            return;
//        }
//
//        if(predefinedCommands.contains(sentence.toLowerCase().trim())){
//
//            String commands  = sharedPreferences.getString(sentence.toLowerCase().trim(),null);
//            sentence = commands.replaceAll(";"," and ");
//            Log.d(debugLogTag, "Loaded predefined commands: "+ sentence);
//        }
//
//        Log.d(debugLogTag, "Current EXEC SEQ: " + currentSequence.toString());
//
//        if(sentence.equals("") && currentSequence.size() == 0) {
//            skipPreviousRTECheck = true;
//            return;
//        }
//
//        if(!sentence.equals("")){
//            skipPreviousRTECheck = true;
//            String[] commands = sentence.split("and");
////            for(int i = 0; i < commands.length; i ++){
////                String command = commands[i];
////                commands[i] = command;
////            }
//            Collections.addAll(currentSequence, commands);
//            Log.d(debugLogTag, "New EXEC SEQ: " + currentSequence.toString());
//        }
//
//        if(!skipPreviousRTECheck){
//            if(uiElements.equals(previousUIElements)) {
//                Log.d(debugLogTag, "No change in RTE: " + uiElements.toString() + ", " + previousUIElements.toString());
//                return;
//            }
//        }
//
//        if(currentSequence.size() != 0) {
//            ActionTargetMatcher currentCommandMatcher;
//            currentCommand = currentSequence.remove(0);
//            if (currentCommand.contains("back")){
//                performGlobalAction(GLOBAL_ACTION_BACK);
//
//            } else if (currentCommand.contains("home")) {
//                performGlobalAction(GLOBAL_ACTION_HOME);
//            } else if (currentCommand.replace(" ","").contains("shutdownservice")){
//                Button listenBtn = mLayout.findViewById(R.id.listenBtn);
//                listenBtn.setText("Start");
//                isVoiceCommandConnected = false;
//                speechRecognizer.stopListening();           // on click listener to stop listening & processing data
//            } else {
//                currentCommandMatcher = new ActionTargetMatcher(currentCommand, uiElements, appNames);
//                previousUIElements.clear();
//                previousUIElements.addAll(uiElements);
//                skipPreviousRTECheck = false;
////                Log.d(debugLogTag, "Executing: " + currentCommand + " RTEnv: " + uiElements.toString());
//                ArrayList<HashMap<ActionTargetMatcher.Action, ArrayList<String>>> actionTargetPairs = currentCommandMatcher.getCommandActionTargets();
//                Log.d(debugLogTag, "AT Pairs: " + actionTargetPairs.toString());
//
//                if (actionTargetPairs.size() == 0 && !sentence.contains("match not found") && !sentence.contains("did you mean")) {
//                    skipPreviousRTECheck = true;
//                    //this.speakerTask("Target match not found. ");
//                    //this.executeCommand("");
//                    currentSequence.clear();
//                }
//                for (HashMap<ActionTargetMatcher.Action, ArrayList<String>> actionTargetPair : actionTargetPairs) {
//                    //Log.d(debugLogTag, actionTargetPair.toString());
//                    List<ActionTargetMatcher.Action> actionList = new ArrayList<ActionTargetMatcher.Action>(actionTargetPair.keySet());
//                    ActionTargetMatcher.Action currentAction = actionList.get(0);
//                    //                   if(actionTargetPair.get(currentAction).size() > 1){
////                        this.speakerTask("Exact match not found. Did you mean "+ actionTargetPair.get(currentAction).get(0) + " or " +actionTargetPair.get(currentAction).get(1) + "?");
////                        skipPreviousRTECheck = true;
////                        //this.executeCommand("");
////                        currentSequence.clear();
////                        return;
////                        String currentTarget = actionTargetPair.get(currentAction).get(0);
////                        actionTargetPair.get(currentAction).clear();
////                        actionTargetPair. currentTarget;
//
//
////                    }
//
//                    hasExecuted = false;
//                    boolean isExecuted = false;
//                    if (currentAction == ActionTargetMatcher.Action.CLICK) {
//                        Log.d("dddf",actionTargetPair.get(currentAction).get(0));
//                        clickButtonByText(actionTargetPair.get(currentAction).get(0));
//                        isExecuted = true;
//                    } else if (currentAction == ActionTargetMatcher.Action.SCROLL) {
//                        scrollingActivity(actionTargetPair.get(currentAction).get(0));
//                        skipPreviousRTECheck = true;
//                        isExecuted = true;
//                    } else if (currentAction == ActionTargetMatcher.Action.OPEN) {
//                        openApp(actionTargetPair.get(currentAction).get(0));
//                        isExecuted = true;
//                    } else if (currentAction == ActionTargetMatcher.Action.ENTER) {
//                        setTextForAllSubNode(currentSource, 0, actionTargetPair.get(currentAction).get(0));
//                        skipPreviousRTECheck = true;
//                        isExecuted = true;
//                    }
//                    if(isRecording && isExecuted){
//                        Log.d(debugLogTag,"Duc" + currentCommand);
//                        savedCommands.add(currentCommand);
//                    }
//                }
//            }
//            hasExecuted = true;
//            try{
//                java.util.concurrent.TimeUnit.SECONDS.sleep(3); // emulates thread synchronisation
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//        }
//    }
//public void invokeComponent(String appSearchString, String featureSearchString) {
//    appSearchString = appSearchString.replace("_"," ");
//    featureSearchString = featureSearchString.replace("_", " ");
//    Log.d("duc", String.valueOf(packageDataObjects.size()));
//    boolean isAppFound = false;
//    String componentOutput = "";
//    for(PackageDataObject packageDataObject: packageDataObjects){
//
//        if(packageDataObject.name.replace("!","").equals(appSearchString)){
//            componentOutput = getAppComponent(packageDataObject,featureSearchString);
//            matchedDeeplinks = packageDataObject.getDeeplinkSearch(featureSearchString);
//            matchedIntents = packageDataObject.getQuerySearch(featureSearchString);
//            currentOpeningPackage = packageDataObject.packageName;;
//            currentOpeningFeature = featureSearchString;
//            currentAppName = appSearchString;
//            isAppFound = true;
//            break;
//        }
//    }
//
//    String[] dataArrDeeplink = new String[matchedDeeplinks.size()];
//    dataArrDeeplink = matchedDeeplinks.toArray(dataArrDeeplink);
//    String[] dataArrIntent = new String[matchedIntents.size()];
//    dataArrIntent = matchedIntents.toArray(dataArrIntent);
//
//
//    if (dataArrIntent.length != 0 && currentIntentIndex < dataArrIntent.length) {
//
//        String intentName = dataArrIntent[currentIntentIndex];
//        String[] intents = intentName.split("  ");
//        Intent intent = new Intent (intents[0]);
//        intent.setComponent(new ComponentName(currentOpeningPackage,intents[1]));
//        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
//        try {
//            PackageManager packageManager = getPackageManager();
//
//            if (intent.resolveActivity(packageManager) != null) {
//                startActivity(intent);
//                intentValidationHandler.postDelayed(intentExecutionValidation,4000);
//                Log.d("duc", intentName+ "1111" + currentOpeningPackage +"11111" + intents[1]);
//            } else {
//                Log.d("Duc", "No Intent available to handle action");
//                currentIntentIndex +=1;
//                invokeComponent(appSearchString,featureSearchString);
//            }
//
//
//        } catch (Exception e) {
//            currentIntentIndex +=1;
//            // Define what your app should do if no activity can handle the intent
//            Log.d("Duc", e.getMessage());
//            invokeComponent(appSearchString,featureSearchString);
//        }
//
//
//    }
//    else if(dataArrDeeplink.length != 0){
//        Intent intent = new Intent (Intent.ACTION_VIEW);
//        intent.setData (Uri.parse( dataArrDeeplink[0]));
//        try {
//            startActivity(intent);
//        } catch (ActivityNotFoundException e) {
//            // Define what your app should do if no activity can handle the intent.
//
//            Log.d("Invocation Errors", e.getMessage());
//        }
//
//    }
//    // open the app instead
//    else {
//        Intent mIntent = getPackageManager().getLaunchIntentForPackage(
//                currentOpeningPackage);
//
//        if (mIntent != null) {
//            try{
//                Log.d("Duc", "Open app launcher screen instead");
//                startActivity(mIntent);
//                speakerTask("Component not found, opened the app instead");
//                isPlaying = true;
//                // Adding some text-to-speech feedback for opening apps based on input
//                // Text-to-speech feedback if app not found);
////                            speakerTask(speechPrompt.get("open") + inputName);
//
//            } catch (ActivityNotFoundException err) {
//                // Text-to-speech feedback if app not found
////                            speakerTask(speechPrompt.get("noMatch") + inputName);
//
//                // Render toast message on screen
//                Toast t = Toast.makeText(getApplicationContext(),
//                        "APP NOT FOUND", Toast.LENGTH_SHORT);
//                speakerTask("Cannot open the app");
//                t.show();
//            }
//            currentIntentIndex = 0;
//        }
//    }
//}