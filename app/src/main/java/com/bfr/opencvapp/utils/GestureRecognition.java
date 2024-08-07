package com.bfr.opencvapp.utils;


import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.INDEX;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.MIDDLE;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.PINKIE;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.RING;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.THUMB;
import static com.bfr.opencvapp.utils.HumanPoseLandmarks.*;

import static org.opencv.core.CvType.CV_8UC3;

import android.util.Log;

import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.objdetect.Detection;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class GestureRecognition {

    public GestureRecognition(String mname, MultiDetector multiDetector, HandPoseEstimator handPoseEstimator, GestureMotionDetect gestureMotionDetect,
    MotionDetector motionDetector, HumanPoseEstimator humanPoseEstimator) {
        this.name = mname;

        this.multiDetector = multiDetector;
        this.handPoseEstimator = handPoseEstimator;
        this.gestureMotionDetect = gestureMotionDetect;
        this.motionDetector = motionDetector;

        this.humanPoseEstimator = humanPoseEstimator;
    }

    String name = "";

    // detectors
    MultiDetector multiDetector;
    ArrayList<Detection> detections = new ArrayList<Detection>();

    HandPoseEstimator handPoseEstimator;
    public HandPoseEstimator.HandPose handPose = null;
    MotionDetector motionDetector;


    HumanPoseEstimator humanPoseEstimator;
    public HumanPoseEstimator.HumanPose humanPose = null;


    // index of img to reset
    int resetImNb = 0;

    // number of frames for optical flow
    final int NUMOFFRAMES= 10;
    // buffer to store the sequence of frame for optical flow analysis
    ArrayList<Mat> matArray = new ArrayList<Mat>();

    GestureMotionDetect gestureMotionDetect;

    public boolean isStarted = false;

    // recognition sequence vars (steps,...)
    public int step_num =0;
    private int previous_step = 0;
    public boolean go = true;


    // coords of the detected hand bbox
    public int left, right, top, bottom;
    // margin to crop the hand in pixel
    final int MARGIN = 50;
    public Rect handROI;
    // black background
    Mat black;
    Mat roiInBlack; // subimage at hand roi in black image
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

    //
    public Mat displaymat;

    // callback
    private IGestureRsp gestureRsp;
    private Gesture gesture = new Gesture();

    public void registerGestureRecog(IGestureRsp gestureRsp)
    {
        this.gestureRsp = gestureRsp;
    }

    public void recognize(Mat input)
    {
        Mat frame = input.clone();

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
        rows = frame.rows();
        cols = frame.cols();

        try{
            // if step changed
            if( !(step_num == previous_step)) {
                // display current step
                Log.i(name, "current step: " + step_num + "  ");
                // update
                previous_step = step_num;
            } // end if step = same


            // which grafcet step?

            /***/if(step_num==0) { // Wait for go
                    //wait until check box
                    if (go) {
                        // go to next step

                        handROI = new Rect(1,1, 1020, 765);

                        step_num = 10;
                    }
                    return;
                }



            /***/if(step_num==5) { // hands detection

                step_num = 10;
                if(true)
                return;
                humanPose = humanPoseEstimator.recognizeImage(frame);

                //if human detection
                if (humanPose != null) {

                    if (humanPose.landmarks.get(LEFT_WRIST).visibility().get() > 0.7
                            || humanPose.landmarks.get(RIGHT_WRIST).visibility().get() > 0.7) {
                        //**** Crop hand image
                        // ROI of hand
                        handROI = new Rect(humanPose.x(LEFT_WRIST), 1, humanPose.x(LEFT_WRIST)+MARGIN, (rows - 1));
                        gestureMotionDetect.handROI = handROI;
                        // start motion detection
                        Log.w(name, "Req for motion detection ");
                        gestureMotionDetect.go = true;
                        //next step
                        step_num = 10;
//                                step_num =6;
                        Log.i(name, "current step: " + step_num + "  ");
                    }
                } else {
                    Log.d(name, "No human found");
                    return;
                }

                if (true)
                    return;

//                //detecting hands only
//                detections = multiDetector.recognizeImage(frame, 99.0f, 99.0f, 0.4f, 0.0f, false);
//
//                if (detections.size() > 0) {
//
////                    Log.d(name, "detected size : " + (detections.get(0).right - detections.get(0).left) * (detections.get(0).bottom - detections.get(0).top));
//
//                        // reset
//                        handID = -1;
//
//                        // for each detection
//                        for(int h=0; h<detections.size();h++)
//                        {
//                            // if detected object is big enough
//                            if( (detections.get(h).right - detections.get(h).left) * (detections.get(h).bottom - detections.get(h).top)> THRES_HAND_AREA)
//                            {
//                                Log.w(name, "detected size : " + (detections.get(h).right - detections.get(h).left) * (detections.get(h).bottom - detections.get(h).top));
//                                //remember the index
//                                handID = h;
//
//
//                                left =  (int) (detections.get(handID).left * cols );
//                                int leftWMargin = Math.max(1, (int) (detections.get(handID).left * cols )- MARGIN);
//                    top = Math.max(1, (int) (detections.get(handID).top * rows) -MARGIN);
////                                top = 1;
//                                right = (int) (detections.get(handID).right * cols);
//                                int rightWMargin = Math.min(frame.cols() - 1, right + MARGIN);
//                    bottom = Math.min(frame.rows() - 1, (int) (detections.get(handID).bottom * rows) +MARGIN);
////                                bottom = rows;
//
//                                //**** Crop hand image
//                                // ROI of hand
//                                handROI = new Rect( leftWMargin, 1, (rightWMargin-leftWMargin), (rows-1));
//                                gestureMotionDetect.handROI =handROI;
//                                // start motion detection
//                                Log.w(name, "Req for motion detection ");
//                                gestureMotionDetect.go = true;
//                                //next step
//                                step_num =10;
////                                step_num =6;
//                                Log.i(name, "current step: " + step_num + "  ");
//                                // interrupt
//                                break;
//                            } //end if obj big enough
//                        } // next object
//                } //end if obj. detected
//                else
//                {
//                    // no obj. detected -> end
////                    result = "";
//                    return;
//                }
//
//                //if object detected but not big enough -> end
//                if (handID<0)
//                    return;
//            }

            } //end if step_num

//            if(step_num == 6)
//            {
//                Log.d(name, "Hand pose Estimation");
//
//                left = Math.max(1, (int) (detections.get(handID).left * cols )- MARGIN);
//                top = Math.max(1, (int) (detections.get(handID).top * rows) -MARGIN);
//                right = Math.min(frame.cols() - 1, (int) (detections.get(handID).right * cols) + MARGIN);
//                bottom = Math.min(frame.rows() - 1, (int) (detections.get(handID).bottom * rows) +MARGIN);
//
//                //**** Crop hand image
//                // ROI of hand
//                handROI = new Rect( left, top, (right-left), (bottom-top));
//                // black background
//                black = new Mat(rows,cols, CV_8UC3, new Scalar(0, 0, 0));
//                roiInBlack = black.submat(handROI); // subimage at hand roi in black image
//                handMat = frame.submat(handROI); // subimage at hand roi in original image containing the crop of the hand
//                // copy hand crop to black background
//                handMat.copyTo(roiInBlack);
//
//                frame = black.clone();
//
//                // to keep to debug
//                // displaymat = black.clone();
//
//                //hand pose estimation
//                handPose = handPoseEstimator.recognizeImage(frame);
//                if(handPose!=null)
//                handPose.handOrientation();
//                step_num = 5;
//                return;
//            }

            /***/if(step_num==10) { // Pose estimation

                try {
                    Log.d(name, "Hand pose Estimation");


//                    // black background
//                    black = new Mat(rows,cols, CV_8UC3, new Scalar(0, 0, 0));
//                    roiInBlack = black.submat(handROI); // subimage at hand roi in black image
//                    handMat = frame.submat(handROI); // subimage at hand roi in original image containing the crop of the hand
//                    // copy hand crop to black background
//                    handMat.copyTo(roiInBlack);
//
//                    frame = black.clone();


                    // to keep to debug
                    // displaymat = black.clone();

                    //hand pose estimation
                    handPose = handPoseEstimator.recognizeImage(frame);

                    if (handPose == null)
                    {
//                        Log.d(name, "NO HAND for POSE ESTIMATION");
//                        debugSaveImg("NOHAND", frame);
//                        step_num = 5;
                        return;
                    }

                    // start motion detection
                    Log.w(name, "Req for motion detection ");

                    // set hand ROI

                    int HAND_ROI_MARGIN = 50;
                    left = Math.max(2, (int)(handPose.landmarks.get(leftLandmark(handPose.landmarks)).x()*1024) - HAND_ROI_MARGIN);
                    top = Math.max(2,(int)(handPose.landmarks.get(topLandmark(handPose.landmarks)).y()*768)-HAND_ROI_MARGIN);
                    right = Math.min(frame.cols()-2, (int)(handPose.landmarks.get(rightLandmark(handPose.landmarks)).x()*1024)+HAND_ROI_MARGIN);
                    bottom = Math.min(frame.rows()-2,  (int)(handPose.landmarks.get(bottomLandmark(handPose.landmarks)).y()*768)+HAND_ROI_MARGIN);


                    handROI.x =  left;
                    handROI.y = top;
                    handROI.height = bottom - top;
                    handROI.width = right -top;

                    gestureMotionDetect.setHandROI(handROI);


                    Log.d("coucou2", "GestureRecog " + left + " " + top + " " + right + " " + bottom);

                    gestureMotionDetect.go = true;


                } catch (Exception e) {
                    e.printStackTrace();
                    step_num = 5;
                    return;
                }


                Log.d(name, "Finger status : " + handPose.isOpen(THUMB) + " " + handPose.isOpen(INDEX) + " " + handPose.isOpen(MIDDLE) + " " + handPose.isOpen(RING) + " " + handPose.isOpen(PINKIE));
                //
                if (handPose.isOpen(INDEX) && handPose.isOpen(MIDDLE) && handPose.isOpen(RING) && handPose.isOpen(PINKIE)) // hand is open
                {
                    // init frame index for buffer recording
                    imNum = 0;
                    step_num = 100;
                    Log.d(name, "Hand is open -> 100 : ");
                }
                // Thumbs open
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
                    if (handPose.isFront() && handPose.handOrientation()<=40 ) {
                        Log.d(name, "FRONT -> 900 : ");
                        result = "STOP";
                        gesture.result = result;
                        gesture.orientation = 0;
                        gestureRsp.onSuccess(gesture);
                        Log.d(name, "STOP");
                        debugSaveImg(result, frame);
                        step_num = 5; // wait for no hands in the image
                    } else {
                        Log.d(name, "Else back -> 5 : ");
                        step_num = 5;
                        return;
                    }

                } //endif hand is open

            }

            /***/if(step_num==100) { // wait for end of motion detection
                if(!gestureMotionDetect.go) {
                    step_num = 120;
                    Log.i(name, "current step: " + step_num + "  ");
                }
            }



            /***/if(step_num==120) { // motion result

                int widthcrop = right-left;
                int heightcrop = bottom - top;
                int area = widthcrop*heightcrop;
                float proportionh = gestureMotionDetect.optFlow/(float)widthcrop;
                float proportionv = gestureMotionDetect.optFlow/(float)heightcrop;

                Log.d(name, "Calculating is front or not" );
                // if seeing palm
                if (handPose.isFront()) {
                    if (gestureMotionDetect.optFlow > THRES_OPT_FLOW_COUCOU)
//                    if (proportionv > THRES_PROPORTIONAL_OPT_FLOW_COUCOU)
                    {
                        Log.w(name, "COUCOU Measured opt flow="+ gestureMotionDetect.optFlow + "length=" + widthcrop+
                                "\nproportionh=" + proportionh + " proportionv=" + proportionv);
                        result = "COUCOU";
                        gesture.result = result;
                        gesture.orientation = 0;
                        gestureRsp.onSuccess(gesture);
                        BuddySDK.Speech.startSpeaking("Coucou");
                        debugRecord("coucou");
                    }
                    else // palm and not moving
                    {
                        result = "STOP";

                        Log.w(name, "STOP Measured opt flow="+ gestureMotionDetect.optFlow + "length=" + widthcrop+
                                "\nproportionh=" + proportionh + " proportionv=" + proportionv);gesture.result = result;
                        gesture.orientation = 0;
                        gestureRsp.onSuccess(gesture);
                        BuddySDK.Speech.startSpeaking("STOP");
                        debugRecord("stop");

                    }

                }
                else // back of the hand
                {
                    //fingers upward
                    if( handPose.fingerOrientation(INDEX) >0) {
                        if (gestureMotionDetect.optFlow > THRES_OPT_FLOW_COME_HERE) {
//                        if (proportionv > THRES_PROPORTIONAL_OPT_FLOW_COUCOU) {

                            Log.d(name, "COME HERE");
                            result = "COME HERE";

                            gesture.result = result;
                            gesture.orientation = 0;
                            gestureRsp.onSuccess(gesture);

//                        BuddySDK.Vision.stopCamera(new IVisionRsp.Stub() {
//                            @Override
//                            public void onSuccess(String s) throws RemoteException {
//
//                            }
//
//                            @Override
//                            public void onFailed(String s) throws RemoteException {
//
//                            }
//                        });
                            BuddySDK.Speech.startSpeaking("J'arrive");
//                        BuddySDK.Companion.raiseEvent("startFollow");
                            debugRecord("comehere");
                        } //end if motion
                        else //back of hand and no motion
                        {
                            step_num = 5;
                            return;
                        }
                    }
                    else // fingers downward
                    {
                        if (gestureMotionDetect.optFlow > THRES_OPT_FLOW_COME_HERE) {
//                        if (proportionv > THRES_PROPORTIONAL_OPT_FLOW_COUCOU) {

                            Log.d(name, "GO AWAY" + gestureMotionDetect.optFlow);
                            result = "GO AWAY";

                            gesture.result = result;
                            gesture.orientation = 0;
                            gestureRsp.onSuccess(gesture);

//                        BuddySDK.Vision.stopCamera(new IVisionRsp.Stub() {
//                            @Override
//                            public void onSuccess(String s) throws RemoteException {
//
//                            }
//
//                            @Override
//                            public void onFailed(String s) throws RemoteException {
//
//                            }
//                        });
                            BuddySDK.Speech.startSpeaking("Je m'en vais");
//                        BuddySDK.Companion.raiseEvent("startFollow");
                            debugRecord("goaway");
                        } //end if motion
                        else //back of hand and no motion
                        {
                            Log.d(name, "Back hand and no motion = " + gestureMotionDetect.optFlow);
                            step_num = 5;
                            return;
                        }
                    }


                } //end if front or back of hand

                //reset
                resetImNb = 0;

                step_num = 900;

                return;

            }

            /***/if(step_num==200) { // Thumb up down


                if (handPose.fingerOrientation(THUMB) >= 0) {
                    Log.d(name, "POSITIVE");
                    result = "POSITIVE";

                    gesture.result = result;
                    gesture.orientation = 0;
                    gestureRsp.onSuccess(gesture);

                    debugSaveImg(result, frame);

                    step_num = 5;
                } else {
                    Log.d(name, "NEGATIVE");
                    result = "NEGATIVE";

                    gesture.result = result;
                    gesture.orientation = 0;
                    gestureRsp.onSuccess(gesture);

                    debugSaveImg(result, frame);

                    step_num = 5;
                }

                return;
            }

            /***/if(step_num==250) { // Rock'n roll

                    Log.d(name, "RockNRoll");
                    result = "RockNRoll";

                gesture.result = result;
                gesture.orientation = 0;
                gestureRsp.onSuccess(gesture);

                debugSaveImg(result, frame);

                    step_num = 5;

                return;
            }

            /***/if(step_num==260) { // Allo

                Log.d(name, "ALLO");
                result = "ALLO";

                gesture.result = result;
                gesture.orientation = 0;
                gestureRsp.onSuccess(gesture);

                debugSaveImg(result, frame);

                step_num = 5;

                return;
            }
            /***/if(step_num==270) { // F You

                Log.d(name, "F*** You");
                result = "F*** YOU";

                gesture.result = result;
                gesture.orientation = 0;
                gestureRsp.onSuccess(gesture);

                debugSaveImg(result, frame);

                step_num = 5;

                return;
            }
            /***/if(step_num==280) { // Peace

                Log.d(name, "Peace");
                result = "PEACE";

                gesture.result = result;
                gesture.orientation = 0;
                gestureRsp.onSuccess(gesture);

                debugSaveImg(result, frame);

                step_num = 5;

                return;
            }
            /***/if(step_num==290) { // Pointing

                Log.d(name, "Pointing");
                int fingerAngle = handPose.fingerOrientation(INDEX);
                result = "POINTING"  ;

                gesture.result = result;
                gesture.orientation = fingerAngle;
                gestureRsp.onSuccess(gesture);

                debugSaveImg(result, frame);

                step_num = 5;

                return;
            }

            /***/if(step_num==900) { //wait for no hands in region of analysis

                try{
                    // black background
                    black = new Mat(rows,cols, CV_8UC3, new Scalar(0, 0, 0));
                    roiInBlack = black.submat(handROI); // subimage at hand roi in black image
                    handMat = frame.submat(handROI); // subimage at hand roi in original image containing the crop of the hand
                    // copy hand crop to black background
                    handMat.copyTo(roiInBlack);

                    frame = black.clone();
                    handPose = handPoseEstimator.recognizeImage(frame);
                } catch (Exception e) {
                    step_num = 10;
                    return;
                }


                // No more detected hand
                if (handPose == null) {
                    Log.d(name, "No more hand");
//                    Imgcodecs.imwrite("/sdcard/Download/nomorehand"+ System.currentTimeMillis()+".jpg", frame);
                    step_num = 5;
                    return;
                }

                //if previously detected a COME HERE
                if(gesture.result.toUpperCase().contains("COME"))
                {
                    //reset if hand not open
                    if ( !handPose.isOpen(INDEX)
                            || !handPose.isOpen(MIDDLE)
                            || !handPose.isOpen(RING)
                            || !handPose.isOpen(PINKIE)
                            || handPose.fingerOrientation(INDEX) <=0 //or hand not upwards (= GO AWAY)
                            || handPose.isFront()) // or hand front(COUCOU or STOP)
                    {
                        Log.d(name, "Hand pose changed : "
                               + handPose.isOpen(INDEX) + " "
                                + handPose.isOpen(MIDDLE) + " "
                                + handPose.isOpen(RING) + " "
                                + handPose.isOpen(PINKIE) + " "
                                + handPose.isFront() + " -> 10 ") ;

                        step_num = 901;
                    } //end if hand changed

                }
                else if(gesture.result.toUpperCase().contains("AWAY")) {

                    //reset if hand not open
                    if ( !handPose.isOpen(INDEX)
                            || !handPose.isOpen(MIDDLE)
                            || !handPose.isOpen(RING)
                            || !handPose.isOpen(PINKIE)
                            || handPose.fingerOrientation(INDEX) >0 //or hand upwards (= COME Here)
                            || handPose.isFront()) // or hand front(COUCOU or STOP)
                    {
                        step_num = 901;
                    } //end if hand changed

                }
                else if(gesture.result.toUpperCase().contains("COUCOU")) {
                    //reset if hand not open
                    if (!handPose.isOpen(INDEX)
                            || !handPose.isOpen(MIDDLE)
                            || !handPose.isOpen(RING)
                            || !handPose.isOpen(PINKIE)
                            || !handPose.isFront()) // or hand Back(COME HERE or GO AWAY)
                    {
                        Log.d(name, "Hand pose changed : "
                                + handPose.isOpen(INDEX) + " "
                                + handPose.isOpen(MIDDLE) + " "
                                + handPose.isOpen(RING) + " "
                                + handPose.isOpen(PINKIE) + " "
                                + handPose.isFront() + " -> 10 ") ;

                        step_num = 901;

                    } //end if hand changed
                }
                else if(gesture.result.toUpperCase().contains("STOP")) {
                    //reset if hand not open
                    if (!handPose.isOpen(INDEX)
                            || !handPose.isOpen(MIDDLE)
                            || !handPose.isOpen(RING)
                            || !handPose.isOpen(PINKIE)
                            || !handPose.isFront()) // or hand Back(COME HERE or GO AWAY)
                    {
                        Log.d(name, "Hand pose changed : "
                                + handPose.isOpen(INDEX) + " "
                                + handPose.isOpen(MIDDLE) + " "
                                + handPose.isOpen(RING) + " "
                                + handPose.isOpen(PINKIE) + " "
                                + handPose.isFront() + " -> 10 ") ;

                        step_num = 901;

                    } //end if hand changed

                    // detect motion
                    motionDetector.detectMotion(frame, false);

                    resetImNb += 1;
                    // take a few images for optical flow
                    if(resetImNb<=2){
                        Log.d(name, "Optical flow on im: " + resetImNb);
                        return;
                    }
                    else //reset and loop if enough images
                        resetImNb =0;

                    if(motionDetector.motionOptFlow > THRES_OPT_FLOW_COUCOU)  // Presence of mvt => COUCOU?
                    {
                        Log.d(name, "Previously detected STOP and motion detected ("+motionDetector.motionOptFlow+") -> reset to 10");
                        step_num = 10;
                        return;
                    }

                }
                else{
                    Log.d(name, "Nothing recognized ? -> reset to 5");
                    // reset
                    step_num = 5;
                    return;
                } //end if result = COME or else

                //
                if (detections.size() == 0)
                {
                    Log.d(name, "No more hand -> reset to step 5");
                    step_num = 5;
                }
                return;
            }

            /***/if (step_num==901) // stabilization
            {
                Thread.sleep(1000);
                step_num = 10;
                return;
            }

        } catch (Exception e) {
            Log.e(name, "ERROR :" + Log.getStackTraceString(e));
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
//            Log.d("coucou3", "Bottom:" + i +" "+ landmarks.get(i).y() + "("+tmpValue+")");
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
            Imgcodecs.imwrite("/sdcard/Download/"+ folder + "/" + strDate+"/_gestRecog.jpg", img);


    } // end record debug



    }
