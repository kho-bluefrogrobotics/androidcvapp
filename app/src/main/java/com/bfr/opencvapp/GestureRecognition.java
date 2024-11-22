package com.bfr.opencvapp;

import static com.bfr.opencvapp.MainActivity.IMG_HEIGHT;
import static com.bfr.opencvapp.MainActivity.IMG_WIDTH;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.INDEX;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.MIDDLE;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.PINKIE;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.RING;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.THUMB;
import static com.bfr.opencvapp.utils.HumanPoseLandmarks.*;

import android.util.Log;

import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.objdetect.Detection;
import com.bfr.opencvapp.utils.Gesture;
import com.bfr.opencvapp.utils.GestureMotionDetect;
import com.bfr.opencvapp.utils.HandPose;
import com.bfr.opencvapp.utils.HandPoseEstimator;
import com.bfr.opencvapp.utils.HumanPose;
import com.bfr.opencvapp.utils.HumanPoseEstimator;
import com.bfr.opencvapp.utils.IGestureRsp;
import com.bfr.opencvapp.utils.MotionDetector;
import com.bfr.opencvapp.utils.MultiDetector;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoWriter;

import java.io.File;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class GestureRecognition {

    public GestureRecognition(String mname, com.bfr.opencvapp.utils.MultiDetector multiDetector, HandPoseEstimator handPoseEstimator, GestureMotionDetect gestureMotionDetect,
                              MotionDetector motionDetector, HumanPoseEstimator humanPoseEstimator) {
        this.name = mname;

        this.multiDetector = multiDetector;
        this.handPoseEstimator = handPoseEstimator;
        this.gestureMotionDetect = gestureMotionDetect;
        this.motionDetector = motionDetector;

        this.humanPoseEstimator = humanPoseEstimator;
        this.handROI = new Rect(0,0,0,0);
        this.armROI = new Rect(0,0,0,0);
    }

    String name = "";

    // detectors
    MultiDetector multiDetector;

    HandPoseEstimator handPoseEstimator;
    public HandPose handPose = null;
    MotionDetector motionDetector;

    int stabilizationFrames =0;
    int numofLowLevelMotion =0;
    int accumulatedMotion =0;


    HumanPoseEstimator humanPoseEstimator;
    public HumanPose humanPose = null;

    GestureMotionDetect gestureMotionDetect;

    // recognition sequence vars (steps,...)
    public int step_num =0;
    private int previous_step = 0;
    public boolean go = true;


    // coords of the detected hand bbox
    public int left, right, top, bottom;
    // margin to crop the hand in pixel
    final int MARGIN = 50;
    public Rect armROI, handROI;
    Mat handMat; // subimage at hand roi in original image containing the crop of the hand


    //Thres for minimum size of hand to analyse, in % of the image area
    // Suggestion 0.2 for narrow-angle camera, 0.15 for wide-angle camera
    float THRES_HAND_AREA = 0.02f ;
    // id of the first largest hand visible
    int handID=-1;

    //dims of the input image
    int rows, cols;

    // index of image to record for optical flow
    int imNum=0;

    // motion
//    boolean motion = false;
    float optFlow = 0.0f;
    // thres for optical flow
//    float THRES_OPT_FLOW_COUCOU = 15.f;
    float THRES_OPT_FLOW_COUCOU = 2.0f;
    float THRES_PROPORTIONAL_OPT_FLOW_COUCOU = 0.015f;
//    float THRES_OPT_FLOW_COME_HERE = 10.0f;
    float THRES_OPT_FLOW_COME_HERE = 2.0f;
    public String result = "";

    int TRIALS_TO_FIND_MOTION = 3;
    int NUM_OF_LOW_MOTION_LEVEL = 2;
    int TRIALS = 5;
    int numofTry = 0;
    //
    public Mat displaymat;

    // callback
    private IGestureRsp gestureRsp;
    private Gesture gesture = new Gesture();

    int signingHand = -1;

    float prevWristPos = 999.0f;
    float currWristPos = 0.0f;
    /** Recognised gestures*/
    public enum GESTURE{
        COUCOU,
        COME_HERE,
        STOP,
        POSITIVE,
        NEGATIVE,
        POINTING,
        ROCKNROLL,
        FYOU,
        ALLO
    }

    double tLasDisplay = 0;

    LocalDateTime myDateObj = LocalDateTime.now();
    DateTimeFormatter myFormatObj = DateTimeFormatter.ofPattern("yyMMddHHmmss");
    VideoWriter videoWriter;
    String formattedDate = myDateObj.format(myFormatObj);
    String debugFileName = "/storage/emulated/0/Download/" + formattedDate + "_trackingDebug.avi" ;
    int fourcc =-1;
    List<Mat> listOfMat = new ArrayList<>();
    int imgIdx = 0;
    int NUM_OF_IMG = 15;


    public void registerGestureRecog(IGestureRsp gestureRsp)
    {
        this.gestureRsp = gestureRsp;
    }

    public void recognize(Mat input)  {
        Mat frame = input.clone();

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
        rows = frame.rows();
        cols = frame.cols();


            // if step changed
            if( !(step_num == previous_step)) {
                // display current step
                Log.i(name, "current step: " + step_num + "  (previous step: " + previous_step + ")");
                // update
                previous_step = step_num;
            } // end if step = same


            // which grafcet step?

            /***/if(step_num==0) { // Wait for go
                    //wait until check box
                    if (go) {
                        // go to next step
                        step_num = 5;
                    }
                    return;
                }


        /** Detect Human and wait for a signing hand*/
        if(step_num==5) {

            //reset
            left = 1;
            top = 1;
            right = 1;
            bottom = 1;
            result = "";
            gesture.result = result;
            gesture.orientation = 0;

            humanPose = humanPoseEstimator.recognizeImage(frame);

            //if human detection
            if (humanPose != null) {

                //display for debug only
//
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(LEFT_WRIST).x()*IMG_WIDTH, humanPose.landmarks.get(LEFT_WRIST).y()*IMG_HEIGHT),
//                        5, new Scalar(0,255,0), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(LEFT_SHOULDER).x()*IMG_WIDTH, humanPose.landmarks.get(LEFT_SHOULDER).y()*IMG_HEIGHT),
//                        5, new Scalar(0,150,150), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(LEFT_HIP).x()*IMG_WIDTH, humanPose.landmarks.get(LEFT_HIP).y()*IMG_HEIGHT),
//                        5, new Scalar(0,150,150), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(LEFT_ELBOW).x()*IMG_WIDTH, humanPose.landmarks.get(LEFT_ELBOW).y()*IMG_HEIGHT),
//                        5, new Scalar(0,255,250), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(LEFT_INDEX).x()*IMG_WIDTH, humanPose.landmarks.get(LEFT_INDEX).y()*IMG_HEIGHT),
//                        5, new Scalar(0,255,250), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(LEFT_THUMB).x()*IMG_WIDTH, humanPose.landmarks.get(LEFT_THUMB).y()*IMG_HEIGHT),
//                        5, new Scalar(0,255,250), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(LEFT_PINKY).x()*IMG_WIDTH, humanPose.landmarks.get(LEFT_PINKY).y()*IMG_HEIGHT),
//                        5, new Scalar(0,255,250), 10);
//
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(RIGHT_WRIST).x()*IMG_WIDTH, humanPose.landmarks.get(RIGHT_WRIST).y()*IMG_HEIGHT),
//                        5, new Scalar(255,0,0), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(RIGHT_ELBOW).x()*IMG_WIDTH, humanPose.landmarks.get(RIGHT_ELBOW).y()*IMG_HEIGHT),
//                        5, new Scalar(255,255,0), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(RIGHT_HIP).x()*IMG_WIDTH, humanPose.landmarks.get(RIGHT_HIP).y()*IMG_HEIGHT),
//                        5, new Scalar(255,255,0), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(RIGHT_INDEX).x()*IMG_WIDTH, humanPose.landmarks.get(RIGHT_INDEX).y()*IMG_HEIGHT),
//                        5, new Scalar(255,255,0), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(RIGHT_THUMB).x()*IMG_WIDTH, humanPose.landmarks.get(RIGHT_THUMB).y()*IMG_HEIGHT),
//                        5, new Scalar(255,255,0), 10);
//                Imgproc.circle(input, new Point(humanPose.landmarks.get(RIGHT_PINKY).x()*IMG_WIDTH, humanPose.landmarks.get(RIGHT_PINKY).y()*IMG_HEIGHT),
//                        5, new Scalar(255,255,0), 10);

//                Log.i(name, "forearm position " + humanPose.landmarks.get(RIGHT_WRIST).y() + "  " + humanPose.landmarks.get(RIGHT_ELBOW).y());

                //reset
                signingHand = -1;
                //if wrist visible and hand up (above elbow)
                signingHand = humanPose.isSigning();

                int handH, handW, handX =0, handX2, handY=0, handY2;
                handH = (int) (IMG_HEIGHT/1.2);
                handW = (int) (IMG_WIDTH/1.7);

                if(signingHand>-1)
                {
                    // Display only
                    if ( signingHand==LEFT_WRIST ) {
                        handX = Math.max( (int) (humanPose.landmarks.get(LEFT_INDEX).x()*IMG_WIDTH - handW/2), 2);
                        handY = Math.max( (int) (humanPose.landmarks.get(LEFT_WRIST).y()*IMG_HEIGHT - handH/2), 2);
                    }
                    else if (signingHand == RIGHT_WRIST) {
                        handX = Math.max( (int) (humanPose.landmarks.get(RIGHT_INDEX).x()*IMG_WIDTH - handW/2), 2);
                        handY = Math.max( (int) (humanPose.landmarks.get(RIGHT_WRIST).y()*IMG_HEIGHT - handH/2), 2);
                    } //end if left or right hand

                    handX2 = Math.min( handX + handW, IMG_WIDTH-2);
                    handY2 = Math.min(handY+handH, IMG_HEIGHT-2);
                    armROI.x = handX;
                    armROI.y = handY;
                    armROI.height = Math.abs(handY2-handY);
                    armROI.width = Math.abs(handX2-handX);

//                        Imgproc.rectangle(input,new Point(handX, handY), new Point(handX2, handY2),
//                                new Scalar(0,220, 0), 4);
//                        Imgcodecs.imwrite("/sdcard/Download/"+System.currentTimeMillis()+"_armROI.jpg", input);

                    //reset
                    stabilizationFrames = 0;
                    numofLowLevelMotion = 0;
                    prevWristPos = 999.0f;
                    currWristPos = 0.0f;
                    //next step
                    step_num = 10;

                } //end if hand is signing

            } else {
                // periodic display
                if(System.currentTimeMillis()-tLasDisplay>2000) {
                    Log.d(name, "No human found");
                    tLasDisplay = System.currentTimeMillis();
                }
                return;
            } //end if human pose found

        } //end if step_num

        /** Human is signing : stabilization of the Wrist*/
        /***/if(step_num==10) {

            // human pose estimation
            humanPose = humanPoseEstimator.recognizeImage(frame);

            // wrist position
            if ( signingHand==LEFT_WRIST )
                currWristPos = humanPose.landmarks.get(LEFT_WRIST).y();
            else
                currWristPos = humanPose.landmarks.get(RIGHT_WRIST).y();

            // if small variation of position
            if(Math.abs(currWristPos-prevWristPos)<=0.1){
                motionDetector.reset();
                step_num = 11;
            }
            else{
                //
                Log.i(name, "current step: " + step_num + "  " + Math.abs(currWristPos-prevWristPos ) );
                prevWristPos = currWristPos;
                return;
            }

        }


        // creation of videoWriter
        if(step_num==11){
            //hand pose estimation
            handPose = handPoseEstimator.recognizeImage(frame.submat(armROI), signingHand);

            //reset if needed
            if (numofTry>TRIALS)
                numofTry=0;

            // if no more hand, exit after a timeout
            if (handPose == null){
                //
                if(numofTry<TRIALS){
                    Log.i(name, "Lost detected hand -> retry");
                    numofTry+=1;
                }
                else { //cancel and start from the begining
                    Log.i(name, "no more hand -> restart");

                    step_num = 5;
                }
                return;
            }

            // set hand ROI
            int HAND_ROI_MARGIN = 60;
            double width = Math.abs( (handPose.landmarks.get(rightLandmark(handPose.landmarks)).x() - handPose.landmarks.get(leftLandmark(handPose.landmarks)).x() )
                    *frame.submat(armROI).cols());
            double height = Math.abs( (handPose.landmarks.get(topLandmark(handPose.landmarks)).y() - handPose.landmarks.get(bottomLandmark(handPose.landmarks)).y() )
                    *frame.submat(armROI).rows()) ;

            int x1 = (int)(handPose.landmarks.get(leftLandmark(handPose.landmarks)).x()*frame.submat(armROI).cols()) + armROI.x;
            int y1 = (int)(handPose.landmarks.get(topLandmark(handPose.landmarks)).y()*frame.submat(armROI).rows()) + armROI.y;
            int x2 = x1 + (int)width;
            int y2 = y1 + (int)height;

            left = Math.max(2, x1 - HAND_ROI_MARGIN - (int)(width/3) ) ;
            top = Math.max(2,y1 -HAND_ROI_MARGIN - (int)(height/4) );
            right = Math.min(frame.cols()-2, x2 + HAND_ROI_MARGIN + (int)(0.3*width) );
            bottom = Math.min(frame.rows()-2, y2 + HAND_ROI_MARGIN + (int)(0.25*height) );

            handROI.x =  left;
            handROI.y = top;
            handROI.height = bottom - top;
            handROI.width = right -left;

//
//            videoWriter = new VideoWriter(debugFileName, fourcc,
//                    13, new Size(1024, 768));
//            videoWriter.open(debugFileName, fourcc,
//                    13, new Size(1024, 768));

            //reset
            listOfMat.clear();
            imgIdx = 0;
            step_num = 12;
        }

        // record frames
        if(step_num == 12){

            if(imgIdx<NUM_OF_IMG){
                listOfMat.add(frame.submat(handROI));
                Log.i(name, "                     recording " + imgIdx);
                imgIdx+=1;
                return;
            }
            else{
                myDateObj = LocalDateTime.now();
                formattedDate = myDateObj.format(myFormatObj);
                debugFileName = "/storage/emulated/0/Download/" + formattedDate + "_trackingDebug.avi" ;
                fourcc = VideoWriter.fourcc('M','J','P','G');
                Log.i(name, "Ready to save video " + handROI.width+"x"+handROI.height);
                videoWriter = new VideoWriter(debugFileName, fourcc,
                        13, new Size(handROI.width, handROI.height));

                step_num =13;
            }
        }

        // add frames to video
        if(step_num==13){

            for (int u=0; u<NUM_OF_IMG; u++){
                videoWriter.write(listOfMat.get(u));
            }
            //save
            videoWriter.release();

            //next step
            step_num = 15;
        }

        /** Human is signing : start motion detection*/
        /***/if(step_num==15) { // Pose estimation

            Log.i(name, "Human is signing");

            //hand pose estimation
            handPose = handPoseEstimator.recognizeImage(frame.submat(armROI), signingHand);

            //reset if needed
            if (numofTry>TRIALS)
                numofTry=0;

            // if no more hand, exit after a timeout
            if (handPose == null){
                //
                if(numofTry<TRIALS){
                    Log.i(name, "Lost detected hand -> retry");
                    numofTry+=1;
                }
                else { //cancel and start from the begining
                    Log.i(name, "no more hand -> restart");

                    step_num = 5;
                }
                return;
            }



            /*** debug*/
//            handPose.isFront();
//            Log.d(name, "Finger status : thumb:" + handPose.isOpen(THUMB) + " index:" + handPose.isOpen(INDEX) + " mid:" + handPose.isOpen(MIDDLE) + " ring:" + handPose.isOpen(RING) + " pinkie:" + handPose.isOpen(PINKIE));


            if(true)
                return;

            // set hand ROI
            int HAND_ROI_MARGIN = 50;
            double width = Math.abs( (handPose.landmarks.get(rightLandmark(handPose.landmarks)).x() - handPose.landmarks.get(leftLandmark(handPose.landmarks)).x() )
                    *frame.submat(armROI).cols());
            double height = Math.abs( (handPose.landmarks.get(topLandmark(handPose.landmarks)).y() - handPose.landmarks.get(bottomLandmark(handPose.landmarks)).y() )
                    *frame.submat(armROI).rows()) ;


            int x1 = (int)(handPose.landmarks.get(leftLandmark(handPose.landmarks)).x()*frame.submat(armROI).cols()) + armROI.x;
            int y1 = (int)(handPose.landmarks.get(topLandmark(handPose.landmarks)).y()*frame.submat(armROI).rows()) + armROI.y;
            int x2 = x1 + (int)width;
            int y2 = y1 + (int)height;

            left = Math.max(2, x1 - HAND_ROI_MARGIN - (int)(width/3) ) ;
            top = Math.max(2,y1 -HAND_ROI_MARGIN - (int)(height/4) );

            right = Math.min(frame.cols()-2, x2 + HAND_ROI_MARGIN + (int)(0.3*width) );
            bottom = Math.min(frame.rows()-2, y2 + HAND_ROI_MARGIN + (int)(0.25*height) );

            handROI.x =  left;
            handROI.y = top;
            handROI.height = bottom - top;
            handROI.width = right -left;


            Log.i(name, "current step: " + step_num + "\n"
                    + handROI.x + "," + handROI.y + ","+handROI.height + "," + handROI.width);

//            handMat = frame.submat(handROI);

//            Imgcodecs.imwrite("/sdcard/Download/"+System.currentTimeMillis()+"_handROI.jpg", handMat );

            motionDetector.reset();


            stabilizationFrames =0;
            numofLowLevelMotion = 0;
            accumulatedMotion = 0;

//            if (false)
//                step_num = 17;
//            else
                step_num = 90;
        }


        /***Motion detected or not*/
        if(step_num==17)
        {

//            if(handPose.isFront())
//                Log.i("rearfront", "FRONT");
//            else {
//                Log.i("rearfront", "BACK");
//            }

            // handROI hasbeen found the step before
            handMat = frame.submat(handROI); // subimage at hand roi in original image containing the crop of the hand

            motionDetector.detectMotion(handMat, 320,240, false);

            accumulatedMotion += motionDetector.motionOptFlow;
//            boolean criteriaOfMotion = (handROI.height>=300 && motionDetector.motionOptFlow >= 30) || (handROI.height<300 && motionDetector.motionOptFlow >= 20);
            boolean criteriaOfMotion = (accumulatedMotion>50);
            if (criteriaOfMotion)
            {
                Log.i(name, "Optical flow detected -> 30");
                step_num = 30;
            }
            else // no motion detected
            {

                //give it another chance
                if(stabilizationFrames<TRIALS_TO_FIND_MOTION){
                    Log.i("coucmotion", "Still looking for motion -> staying in 17 " + stabilizationFrames
                            + "\na=" + motionDetector.motionOptFlow + "  b=" + handROI.height+" c="+handROI.width );
                    // if low level of motion everal times in a row
                    if(motionDetector.motionOptFlow<5) {
                        if(numofLowLevelMotion>NUM_OF_LOW_MOTION_LEVEL) {
                            Log.i(name, "Low level of motion-> 90");
                            step_num = 90;
                        }
                        numofLowLevelMotion += 1;
                        stabilizationFrames+=1;
                        return;
                    }
                    else{ // motion level not low but not enough either

                        stabilizationFrames+=1;
                        return;
                    }


                }
                else{
                    Log.i(name, "No mvt -> 90");
                    Log.w("coucmotion", "STOP: a=" + motionDetector.motionOptFlow + "  b=" + handROI.height+" c="+handROI.width );
                    step_num = 90;
                }
            }


        }

        /***Motion detected : gesture identification according to Front or back hand*/
        if(step_num == 30){
            Log.i(name, "step 30 : Motion detected ");

                if(handPose.isFront()){
                    Log.i(name, "Front hand  ");
                    debugSaveImg("Coucou", frame.submat(armROI));

                    result = "COUCOU";
                    gesture.result = result;
                    gesture.orientation = 0;
                    gestureRsp.onSuccess(gesture);
                    debugSaveImgMotion(result, handMat);
                    BuddySDK.Speech.startSpeaking("Coucou");
                }
                else {
                    Log.i(name, "Not the front hand  ");
                    result = "Come Here";
                    gesture.result = result;
                    gesture.orientation = 0;
                    gestureRsp.onSuccess(gesture);
                    debugSaveImgMotion(result, frame.submat(armROI));
                    BuddySDK.Speech.startSpeaking("J'arrive");
                }


            step_num =300;
        }













            /***No Motion detected: id static hand pose*/
            /***/if(step_num==90) { // Pose estimation

                if(handPose==null){
                    Log.d(name, "Handpose null");
                    step_num =5;
                    return;
                }
                Log.d(name, "Finger status : thumb:" + handPose.isOpen(THUMB) + " index:" + handPose.isOpen(INDEX) + " mid:" + handPose.isOpen(MIDDLE) + " ring:" + handPose.isOpen(RING) + " pinkie:" + handPose.isOpen(PINKIE));
                // Only checking index and thumb for more robustness
                //Thumbs up or down
                if (!handPose.isOpen(INDEX) && handPose.isOpen(THUMB) && !handPose.isOpen(MIDDLE)  && !handPose.isOpen(PINKIE))
                {
                    Log.d(name, "Thumbs open -> 200 : ");
                    step_num = 200;
                }
                else{
                    // double checking index and thumb for more robustness
                    if (handPose.isOpen(INDEX) && handPose.isOpen(MIDDLE) && handPose.isOpen(RING) && handPose.isOpen(PINKIE)) // hand is open
                    {
                        // openhand
                        step_num = 190;
                    }
//                // Thumbs open
                else if (!handPose.isOpen(INDEX) && !handPose.isOpen(MIDDLE) && !handPose.isOpen(RING) && handPose.isOpen(THUMB) && !handPose.isOpen(PINKIE)) // all fingers closed beside thumb
                {
                    Log.d(name, "Thumbs open -> 200 : ");
                    step_num = 200;
                }
                    // index, thumb and pinkie open
                    else if(handPose.isOpen(INDEX) && !handPose.isOpen(MIDDLE) && !handPose.isOpen(RING) && handPose.isOpen(PINKIE))
                    {
                        Log.d(name, "Rock'n roll ->250 : ");
                        step_num = 250;
                    }
                    // just thumb and pinkie open
                    else if(handPose.isOpen(THUMB) && !handPose.isOpen(INDEX) && !handPose.isOpen(MIDDLE) && !handPose.isOpen(RING) && handPose.isOpen(PINKIE))
                    {
                        Log.d(name, "Allo -> 260 : ");
                        step_num = 260;
                    }
                    // allfingers closed except middle finger
                    else if(!handPose.isOpen(INDEX) && handPose.isOpen(MIDDLE) && !handPose.isOpen(RING) && !handPose.isOpen(PINKIE))
                    {
                        Log.d(name, "F*** -> 270 : ");
                        step_num = 270;
                    }
                    // index and middle finger open
                    else if(handPose.isOpen(INDEX) && handPose.isOpen(MIDDLE) && !handPose.isOpen(RING) && !handPose.isOpen(PINKIE))
                    {
                        Log.d(name, "Peace -> 270 : ");
                        step_num = 280;
                    }
                    // only Index open
                    else if(handPose.isOpen(INDEX) && !handPose.isOpen(MIDDLE) && !handPose.isOpen(RING) && !handPose.isOpen(PINKIE))
                    {
                        Log.d(name, "Pointing -> 280 : ");
                        step_num = 290;
                    }
                    else {
                        Log.d(name, "ELSE : Finger status OTHER ");
                        Log.d(name, "OTHER");
                        result = "????";

                        gesture.result = result;
                        gesture.orientation = 0;
                        gestureRsp.onSuccess(gesture);

                        debugSaveImg(result, frame.submat(armROI));

                        step_num = 300;

                    } //endif hand is open
                } //end if thumbs up or down
                



            }


        /***/if(step_num==190) { // hand open

            // double check palm facing the camera
            if(handPose.isFront()) {
                result = "STOP";

                gesture.result = result;
                gesture.orientation = 0;
                gestureRsp.onSuccess(gesture);

                debugSaveImg(result, frame.submat(armROI));
                debugSaveImgMotion(result, frame.submat(armROI));

            }
            else // we assume we see the back hand for a slow come here movement
            {
                Log.i(name, "Not the front hand  -> abort");
            }
            step_num = 300;
        }


        /***/if(step_num==200) { // Thumb up down

                int thumbAngle = handPose.fingerOrientation(THUMB);
                if (thumbAngle >= 0) {
                    Log.d(name, "POSITIVE");
                    result = "POSITIVE";

                    gesture.result = result ;
                    gesture.orientation = 0;
                    gestureRsp.onSuccess(gesture);

//                    debugSaveImg(result, frame);
                    debugSaveImg(result, frame.submat(armROI));

                    step_num = 300;
                } else {
                    Log.d(name, "NEGATIVE");
                    result = "NEGATIVE";

                    gesture.result = result;
                    gesture.orientation = 0;
                    gestureRsp.onSuccess(gesture);

//                    debugSaveImg(result, frame);
                    debugSaveImg(result, frame.submat(armROI));

                    step_num = 300;
                }

                return;
            }

            /***/if(step_num==250) { // Rock'n roll

                    Log.d(name, "RockNRoll");
                    result = "RockNRoll";

                gesture.result = result;
                gesture.orientation = 0;
                gestureRsp.onSuccess(gesture);

//                debugSaveImg(result, frame);
                debugSaveImg(result, frame.submat(armROI));

                    step_num = 300;

                return;
            }

            /***/if(step_num==260) { // Allo

                Log.d(name, "ALLO");
                result = "ALLO";

                gesture.result = result;
                gesture.orientation = 0;
                gestureRsp.onSuccess(gesture);

                debugSaveImg(result, frame.submat(armROI));

                step_num = 300;

                return;
            }
            /***/if(step_num==270) { // F You

                Log.d(name, "F*** You");
                result = "F*** YOU";

                gesture.result = result;
                gesture.orientation = 0;
                gestureRsp.onSuccess(gesture);

                debugSaveImg(result, frame.submat(armROI));

                step_num = 300;

                return;
            }
            /***/if(step_num==280) { // Peace

                Log.d(name, "Peace");
                result = "PEACE";

                gesture.result = result;
                gesture.orientation = 0;
                gestureRsp.onSuccess(gesture);

//                debugSaveImg(result, frame);
                debugSaveImg(result, frame.submat(armROI));

                step_num = 300;

                return;
            }
            /***/if(step_num==290) { // Pointing

                Log.d(name, "Pointing");
                int fingerAngle = handPose.fingerOrientation(INDEX);
                result = "POINTING"  ;

                gesture.result = result;
                gesture.orientation = fingerAngle;
                gestureRsp.onSuccess(gesture);

                debugSaveImg(result, frame.submat(armROI));

                step_num = 300;

                return;
            }

        if (step_num==300) {
            Log.i(name, "step 300 :reset motion detector ");
            //reset
            motionDetector.reset();
            step_num = 305;
            return;
        }

        if(step_num==305){ // wait for change (motion)
            Log.i(name, "step 305 : wait for motion change ");
            // handROI hasbeen found the step before
            handMat = frame.submat(handROI); // subimage at hand roi in original image containing the crop of the hand
            motionDetector.detectMotion(handMat, 320, 240, false);
            if (motionDetector.motionOptFlow>20){
                //go to stabilization step
                step_num = 310;
            }
            else //double check  if huma is signing
            {
                humanPose = humanPoseEstimator.recognizeImage(frame);
                // if no more signing with hand
                if (!humanPose.isSigning(signingHand)){
                    result = "";
                    gesture.result = result;
                    gesture.orientation = 0;
                    gestureRsp.onSuccess(gesture);
                    step_num = 5;
                }
            }
            return;

        }

        /***Human is still signing : waiting for stabilization for motion*/
        if(step_num==310)
        {
            Log.i(name, "User is changing sign : stabilizing... ");
            //give it another chance
            try {
                Thread.sleep(600);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            result = ""  ;
            gesture.result = result;
            gestureRsp.onSuccess(gesture);
            step_num = 15;
        }



    }

    int topLandmark(List<NormalizedLandmark> landmarks)
    {
        int id = -1;
        float tmpValue =9999.0f;

        for (int i=0; i<landmarks.size();i++)
        {
            if(Float.compare(landmarks.get(i).y(), tmpValue)<0){
                tmpValue = landmarks.get(i).y();
                id = i;
            }
        }
        return id;
    }

    int bottomLandmark(List<NormalizedLandmark> landmarks)
    {
        int id = -1;
        float tmpValue =-1.0f;

        for (int i=0; i<landmarks.size();i++)
        {
            if(Float.compare(landmarks.get(i).y(), tmpValue)>0) {
                tmpValue = landmarks.get(i).y();
                id = i;
            }
        }
        return id;
    }

    int leftLandmark(List<NormalizedLandmark> landmarks)
    {
        int id = -1;
        float tmpValue =9999.0f;

        for (int i=0; i<landmarks.size();i++)
        {
            if(Float.compare(landmarks.get(i).x(), tmpValue)<0){
                tmpValue = landmarks.get(i).x();
                id = i;
            }
        }
        return id;
    }

    int rightLandmark(List<NormalizedLandmark> landmarks)
    {
        int id = -1;
        float tmpValue =-1.0f;

        for (int i=0; i<landmarks.size();i++)
        {
            if(Float.compare(landmarks.get(i).x(), tmpValue)>0){
                tmpValue = landmarks.get(i).x();
                id = i;
            }
        }
        return id;
    }

    void debugRecord(String folder)
    {
        ArrayList<Mat> matArray = GestureMotionDetect.matArray;

        Date date = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyMMddHHmmssSSS");
        String strDate= formatter.format(date);
        // create folder if doesn't exist
        File saveDir = new File("", "/sdcard/Download/"+ folder + "/" + strDate);
        if(!saveDir.exists()) {
            // create folder
            saveDir.mkdirs();
        }

        for (int i = 0; i<matArray.size(); i++)
        {
//            Log.d(name, "Saving image " + i);
            Imgcodecs.imwrite("/sdcard/Download/"+ folder + "/" + strDate+"/" + String.format("%02d", i) + "_gestRecog.jpg", matArray.get(i));
        }

    } // end record debug


    void debugSaveImg(String folder, Mat img)
    {

        Date date = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyMMddHHmmssSSS");
        String strDate= formatter.format(date);
        // create folder if doesn't exist
        File saveDir = new File("", "/sdcard/Download/"+ folder + "/" + strDate);
        if(!saveDir.exists()) {
            // create folder
            saveDir.mkdirs();
        }


        Mat toSave = img.clone();
        for (int i=0; i<handPose.landmarks.size(); i++){
            int fingerTipX = (int)(handPose.landmarks.get(i).x()*img.cols()) ;
            int fingerTipY = (int)(handPose.landmarks.get(i).y()*img.rows()) ;
            Imgproc.circle(toSave, new Point(fingerTipX, fingerTipY), 2, new Scalar(255,255,0), 3);
            Imgcodecs.imwrite("/sdcard/Download/"+ folder + "/" + strDate+"/_gestRecog.jpg", toSave);
        }



    } // end record debug


    void debugSaveImgMotion(String folder, Mat img)
    {

        Date date = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyMMddHHmmssSSS");
        String strDate= formatter.format(date);
        // create folder if doesn't exist
        File saveDir = new File("", "/sdcard/Download/"+ folder + "/" + strDate);
        if(!saveDir.exists()) {
            // create folder
            saveDir.mkdirs();
        }
        Imgcodecs.imwrite("/sdcard/Download/"+ folder + "/" + strDate+"/_gestRecog_"+ motionDetector.motionOptFlow+".jpg", img);


    } // end record debug



    }
