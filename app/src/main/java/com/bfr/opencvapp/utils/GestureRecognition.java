package com.bfr.opencvapp.utils;


import static com.bfr.opencvapp.utils.HandPoseEstimator.FINGER.*;

import static org.opencv.core.CvType.CV_8UC3;

import android.content.Context;
import android.util.Log;

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

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = true;
    final static int INTERVAL_MIN = 350;


    private int previous_step = 0;


    // coords of the detected hand bbox
    int left, right, top, bottom;

    int rows, cols;

    int imNum=0;

    // motion
    boolean motion = false;
    public String result = "";

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
            switch (step_num) {
                case 0: // Wait for checkbox
                    //wait until check box
                    if (go) {
                        // go to next step
                        step_num = 5;
                    }
                    break;

                case 5 : // Human face hand detection

                    detections = multiDetector.recognizeImage(frame, 99.0f, 99.0f, 0.7f, 0.0f, false);

                    if(detections.size()>0)
                    {
                        Log.d(name, "detected objs : " + detections.size() + "   id=" + detections.get(0).getDetectedClass() + " ; " + detections.get(0).getConfidence());
                        step_num = 10;
                    }
                    else
                        break;

//                    break;

                case 10: // Pose estimation

                    try{
                        Log.d(name, "Hand pose Estimation");
                        left = Math.max( 1, (int)(detections.get(0).left * cols));
                        top = Math.max(1, (int)(detections.get(0).top * rows));
                        right = Math.min(frame.cols()-1, (int)(detections.get(0).right * cols));
                        bottom = Math.min(frame.rows()-1, (int)(detections.get(0).bottom* rows));


//                        Mat black = new Mat(rows,cols, CV_8UC3, new Scalar(0, 0, 0));
//                        Rect handROI = new Rect( left, top, (right-left), (bottom-top));
//                        Mat handMat = frame.submat(handROI);
//                        black.copyTo(handMat);

                        handPose =  handPoseEstimator.recognizeImage(frame);

                        Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                                new Scalar(0, 255, 0), 3);

                        handPose.fingerOrientation(THUMB);



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
                    }




                    step_num = 10;
                    break;





//
//                    Log.d(name, "Finger status : " + handPose.isOpen(THUMB) + " "+ handPose.isOpen(INDEX) + " "+ handPose.isOpen(MIDDLE) + " "+ handPose.isOpen(PINKIE) + " ");
//                    //
//                    if (handPose.isOpen(THUMB) && handPose.isOpen(INDEX) && handPose.isOpen(MIDDLE) && handPose.isOpen(RING)) // hand is open
//                    {
//                        // init frame index for buffer recording
//                        imNum =0;
//                        step_num = 100;
//                    }
//                    else if (!handPose.isOpen(INDEX) && !handPose.isOpen(RING) && handPose.isOpen(THUMB) ) // all fingers closed beside thumb
//                    {
//                        step_num = 200;
//                    }
//                    else {
//                        if(handPose.isFront())
//                        {
//                            result = "STOP";
//                            Log.d(name, "STOP");
//                            step_num = 900; // wait for no hands in the image
//                        }
//                        else
//                            step_num = 5;
//                        break;
//                    }

                case 100: // Open hand start record video for optical flow
//                    Log.d(name, "recording for optical flow im num:" + imNum +" to " + matArray.size() );
//                    Imgcodecs.imwrite("/sdcard/Download/" + String.format("%02d", imNum) + "_gestRecog.jpg", frame);

                    //add at the end if needed
                    if (matArray.size()<=imNum)
                    {
//                        Log.d(name, "adding:" + imNum +" to " + matArray.size() );
                        matArray.add(frame.clone());
                    }
                    else // record Mat
                    {
//                        Log.d(name, "setting:" + imNum +" to " + matArray.size() );
                        matArray.set(imNum, frame.clone());
                    }


                    //increment index
                    imNum +=1;

                    // next step if recording complete
                    if (imNum>=NUMOFFRAMES)
                        step_num = 110;
                    break;

                case 110: //end of record video

                    // init motion deteciton
                    motionDetector.frameCount = 0;
                    motion = false;
                    step_num = 115;
                    //break;

                case 115: // optical flow analysis
                    Log.d(name, "Optical flow estimation");

                    // Analyse from n-th frame to waith for hand stabilization
                    for(int i=5; i<NUMOFFRAMES;i++) {
//                        Mat img = Imgcodecs.imread("/sdcard/Download/" + String.format("%02d", i)  + "_gestRecog.jpg");

                        // get frame from recorded buffer
                        Mat img = matArray.get(i);
                        motionDetector.detectMotion(img.clone(), false);
                        //record if motion or not at this frame
                        motion = motion || motionDetector.detectedMotion ;
                    }

                    step_num = 120;
                    break;
                case 120 : // motion result

                    if (motion)
                    {
                        if (handPose.isFront())
                        {
                            Log.d(name, "COUCOU");
                            result = "COUCOU";
                            debugRecord("coucou");
                            step_num = 5;
                        }
                        else
                        {
                            Log.d(name, "COME HERE");
                            result = "COME HERE";
                            debugRecord("comehere");
                            step_num = 5;
                        }
                    }
                    else
                    {
                        result = "STOP";
                        Log.d(name, "STOP");
                        debugRecord("stop");
                        step_num = 900;
                    }


                    break;

                case 200 : // close hands

                    handPose.fingerOrientation(THUMB);

                    if (handPose.angle >= 0)
                    {
                        Log.d(name, "POSITIVE");
                        result = "POSITIVE";
                        step_num = 5;
                    }
                    else {
                        Log.d(name, "NEGATIVE");
                        result = "NEGATIVE";
                        step_num = 5;
                    }

                    break;
                case 900 : //wait for no hands
                    detections = multiDetector.recognizeImage(frame, 99.0f, 99.0f, 0.7f, 0.0f, false);

                    if(detections.size()==0)
                        step_num = 5;
                    break;

                default :
                    // go to next step
                    step_num = 0;
                    break;
            } //End switch

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
