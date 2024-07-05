package com.bfr.opencvapp.utils;


import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.INDEX;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.MIDDLE;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.PINKIE;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.RING;
import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.THUMB;

import static org.opencv.core.CvType.CV_8UC3;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.vision.shared.IVisionRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.objdetect.Detection;

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

public class GestureRecognition {

    public GestureRecognition(String mname, MultiDetector multiDetector, HandPoseEstimator handPoseEstimator, MotionDetector motionDetector) {
        this.name = mname;

        this.multiDetector = multiDetector;
        this.handPoseEstimator = handPoseEstimator;
        this.motionDetector = motionDetector;
    }

    String name = "";

    // detectors
    MultiDetector multiDetector = new MultiDetector();
    ArrayList<Detection> detections = new ArrayList<Detection>();

    HandPoseEstimator handPoseEstimator;
    public HandPoseEstimator.HandPose handPose = null;
    MotionDetector motionDetector = new MotionDetector();

    // number of frames for optical flow
    final int NUMOFFRAMES= 10;
    // buffer to store the sequence of frame for optical flow analysis
    ArrayList<Mat> matArray = new ArrayList<Mat>();

    // input frame; reminder the arguements are passed as reference in java
    Mat frame;



    public boolean isStarted = false;

    // recognition sequence vars (steps,...)
    public int step_num =0;
    private int previous_step = 0;
    public boolean go = true;


    // coords of the detected hand bbox
    int left, right, top, bottom;
    //Thres for minimum size of hand to analyse, in % of the image
    // Suggestion 0.2 for narrow-angle camera, 0.15 for wide-angle camera
    float THRES_HAND_WIDTH = 0.20f;
    // id of the first largest hand visible
    int handID=-1;

    //dims of the input image
    int rows, cols;

    // index of image to record for optical flow
    int imNum=0;

    // motion
//    boolean motion = false;
    float optFlow = 0.0f;
    float THRES_OPT_FLOW = 15.f; // Thres for decting motion
    public String result = "";

    //
    public Mat displaymat;

    public void init(Mat frame)
    {
        this.frame = frame;
    }

    public void start()
    {
        isStarted = true;
        try{
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void stop()
    {
        isStarted = false;
        try{

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public void recognize(Mat frame)
    {
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
                        step_num = 5;
                    }
                    return;
                }


            /***/if(step_num==5) { // hands detection

                //detecting hands only
                detections = multiDetector.recognizeImage(frame, 99.0f, 99.0f, 0.7f, 0.0f, false);

                if (detections.size() > 0) {
//                    Log.d(name, "detected objs : " + detections.size() + "   id=" + detections.get(0).getDetectedClass() + " ; " + detections.get(0).getConfidence());
                        Log.d(name, "detected size : " + (detections.get(0).right - detections.get(0).left));

                        // reset
                        handID = -1;

                        // for each detection
                        for(int h=0; h<detections.size();h++)
                        {
                            // if detected object is big enough
                            if( (detections.get(h).right - detections.get(h).left)>THRES_HAND_WIDTH )
                            {
                                //remember the index
                                handID = h;
                                //next step
                                step_num =10;
                                // interrupt
                                break;
                            } //end if obj big enough
                        } // next object
                } //end if obj. detected
                else
                    // no obj. detected -> end
                    return;

                //if object detected but not big enough -> end
                if (handID<0)
                    return;
            }

            /***/if(step_num==10) { // Pose estimation

                try {
                    Log.d(name, "Hand pose Estimation");

                    left = Math.max(1, (int) (detections.get(handID).left * cols ));
                    top = Math.max(1, (int) (detections.get(handID).top * rows));
                    right = Math.min(frame.cols() - 1, (int) (detections.get(handID).right * cols));
                    bottom = Math.min(frame.rows() - 1, (int) (detections.get(handID).bottom * rows));

                    //**** Crop hand image
                    // ROI of hand
                    Rect handROI = new Rect( left, top, (right-left), (bottom-top));
                    // black background
                    Mat black = new Mat(rows,cols, CV_8UC3, new Scalar(0, 0, 0));
                    Mat roiInBlack = black.submat(handROI); // subimage at hand roi in black image
                    Mat handMat = frame.submat(handROI); // subimage at hand roi in original image containing the crop of the hand
                    // copy hand crop to black background
                    handMat.copyTo(roiInBlack);

                    frame = black.clone();

                    // to keep to debug
                    // displaymat = black.clone();

                    //hand pose estimation
                    handPose = handPoseEstimator.recognizeImage(frame);

                    Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                            new Scalar(0, 255, 0), 3);

//                        int x, y;
//                        for (int l=0; l<20; l++)
//                        {
//                            x = (int) (handPose.landmarks.get(l).x()* frame.cols());
//                            y = (int) (handPose.landmarks.get(l).y()* frame.rows());
//                            Imgproc.circle(frame, new Point(x,y), 5, new Scalar(0,255,0), 5);
//                        }

                } catch (Exception e) {
                    e.printStackTrace();
                    step_num = 5;
                    return;
                }


                Log.d(name, "Finger status : " + handPose.isOpen(THUMB) + " " + handPose.isOpen(INDEX) + " " + handPose.isOpen(MIDDLE) + " " + handPose.isOpen(PINKIE) + " ");
                //
                if (handPose.isOpen(THUMB) && handPose.isOpen(INDEX) && handPose.isOpen(MIDDLE) && handPose.isOpen(RING)) // hand is open
                {
                    Log.d(name, "Hand is open -> 100 : ");

                    // init frame index for buffer recording
                    imNum = 0;
                    step_num = 100;
                } else if (!handPose.isOpen(INDEX) && !handPose.isOpen(RING) && handPose.isOpen(THUMB)) // all fingers closed beside thumb
                {
                    Log.d(name, "Thumbs open -> 200 : ");
                    step_num = 200;
                } else {
                    Log.d(name, "ELSE : ");
                    if (handPose.isFront()) {
                        Log.d(name, "FRONT -> 900 : ");
                        result = "STOP";
                        Log.d(name, "STOP");
                        step_num = 900; // wait for no hands in the image
                    } else {
                        Log.d(name, "Else back -> 5 : ");
                        step_num = 5;
                        return;
                    }

                }
            }

            /***/if(step_num==100) { // Open hand start record video for optical flow
                Log.d(name, "Recording for optical flow : ");

                //add at the end if needed
                if (matArray.size() <= imNum) {
//                        Log.d(name, "adding:" + imNum +" to " + matArray.size() );
                    matArray.add(frame.clone());
                } else // record Mat
                {
//                        Log.d(name, "setting:" + imNum +" to " + matArray.size() );
                    matArray.set(imNum, frame.clone());
                }


                //increment index
                imNum += 1;

                // next step if recording complete
                if (imNum >= NUMOFFRAMES)
                    step_num = 110;
                return;
            }

            /***/if(step_num==110) { //end of record video

                Log.d(name, "End of recording : ");

                step_num = 115;
                //break;
            }

            /***/if(step_num==115) { // optical flow analysis
                Log.d(name, "Optical flow estimation");

                // init motion deteciton
                motionDetector.frameCount = 0;
                optFlow = 0.0f;

                // Analyse from n-th frame to waith for hand stabilization
                for (int i = 4; i < NUMOFFRAMES; i++) {
//                        Mat img = Imgcodecs.imread("/sdcard/Download/" + String.format("%02d", i)  + "_gestRecog.jpg");

                    // get frame from recorded buffer
                    Mat img = matArray.get(i);
                    motionDetector.detectMotion(img.clone(), false);
                    Imgcodecs.imwrite("/sdcard/Download/" + String.format("%02d", i) + "_gestRecog.jpg", matArray.get(i));
                    Log.w(name, "Measured opt flow="+ motionDetector.motionOptFlow);
                    //record if motion or not at this frame
//                    motion = motion || motionDetector.detectedMotion;

                    if(motionDetector.motionOptFlow > optFlow)
                        optFlow = motionDetector.motionOptFlow;
                }

                step_num = 120;
                return;
            }

            /***/if(step_num==120) { // motion result

                // if motion detected
                if (optFlow > THRES_OPT_FLOW) {

                    Log.d(name, "Motion optical flow = " + optFlow);

                    if (handPose.isFront()) {
                        Log.d(name, "COUCOU");
                        result = "COUCOU";
                        BuddySDK.Speech.startSpeaking("Coucou");
                        debugRecord("coucou");
//                        step_num = 5;
                    } else {
                        Log.d(name, "COME HERE");
                        result = "COME HERE";
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
//                        step_num = 5;
                    }
                } else {
                    result = "STOP";
                    Log.d(name, "STOP");
                    BuddySDK.Speech.startSpeaking("STOP");
                    debugRecord("stop");

                }

                step_num = 900;

            }

            /***/if(step_num==200) { // close hands

                handPose.fingerOrientation(THUMB);

                if (handPose.angle >= 0) {
                    Log.d(name, "POSITIVE");
                    result = "POSITIVE";
                    step_num = 5;
                } else {
                    Log.d(name, "NEGATIVE");
                    result = "NEGATIVE";
                    step_num = 5;
                }

                return;
            }

            /***/if(step_num==900) { //wait for no hands in region of analysis
                detections = multiDetector.recognizeImage(frame, 99.0f, 99.0f, 0.7f, 0.0f, false);

                if (detections.size() == 0)
                {
                    Log.d(name, "No more hand -> reset to step 5");
                    step_num = 5;
                }
                return;
            }


        } catch (Exception e) {
            Log.e(name, "ERROR :" + Log.getStackTraceString(e));
        }

    }


    void debugRecord(String folder)
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

        for (int i = 0; i<matArray.size(); i++)
        {
//            Log.d(name, "Saving image " + i);
            Imgcodecs.imwrite("/sdcard/Download/"+ folder + "/" + strDate+"/" + String.format("%02d", i) + "_gestRecog.jpg", matArray.get(i));
        }
    } // end record debug
}
