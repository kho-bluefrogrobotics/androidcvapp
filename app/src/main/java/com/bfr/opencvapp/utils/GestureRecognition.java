package com.bfr.opencvapp.utils;


import android.os.RemoteException;
import android.util.Log;

import com.bfr.opencvapp.objdetect.Detection;
import com.bfr.usbservice.IUsbCommadRsp;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoWriter;

import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class GestureRecognition {

    public GestureRecognition(String mname, MultiDetector multiDetector, HandPoseEstimator handPoseEstimator, MotionDetector motionDetector) {
        this.name = mname;
//        this.grafcet_runnable = mysequence;

        this.multiDetector = multiDetector;
        this.handPoseEstimator = handPoseEstimator;
        this.motionDetector = motionDetector;
    }

    GestureRecogSequence gestureRecogSequence;
    // detectors
    MultiDetector multiDetector = new MultiDetector();
    ArrayList<Detection> detections = new ArrayList<Detection>();

    HandPoseEstimator handPoseEstimator = new HandPoseEstimator();
    MotionDetector motionDetector = new MotionDetector();

    // number of frames for optical flow
    final int NUMOFFRAMES= 5;

    // input frame; reminder the arguements are passed as reference in java
    Mat frame;

    String name = "";

    public boolean isStarted = false;

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = true;
    final static int INTERVAL_MIN = 350;


    private int previous_step = 0;

    // Scheduler for grafcet
    private ScheduledExecutorService myscheduler ;

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
        gestureRecogSequence = new GestureRecogSequence();
    }

    public void start()
    {
        isStarted = true;
        try{
            myscheduler = Executors.newScheduledThreadPool(1);
            // start scheduled task
            myscheduler.scheduleWithFixedDelay(gestureRecogSequence, 0, 10, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void stop()
    {
        isStarted = false;
        try{
            myscheduler.shutdown();
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

                    Log.d(name, "Hand pose Estimation");
                    left = (int)(detections.get(0).left * cols);
                    top = (int)(detections.get(0).top * rows);
                    right = (int)(detections.get(0).right * cols);
                    bottom = (int)(detections.get(0).bottom* rows);
                    Rect handROI = new Rect( left, top, (right-left), (bottom-top));
                    Mat handMat = frame.submat(handROI);
                    HandPoseEstimator.HandPose handPose =  handPoseEstimator.recognizeImage(handMat);
//                    Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
//                    new Scalar(0, 255, 0), 3);

                    if (handPose.isOpen() && handPose.isFront())
                    {
                        imNum =0;
                        step_num = 100;
                    }
                    else {
                        step_num = 5;
                        break;
                    }
                    //break;

                case 100: // Open hand start record video for optical flow
                    Log.d(name, "recording for optical flow im num:" + imNum );
                    Imgcodecs.imwrite("/sdcard/Download/" + String.format("%02d", imNum) + "_gestRecog.jpg", frame);
                    imNum +=1;

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

                    for(int i=0; i<NUMOFFRAMES;i++) {
                        Mat img = Imgcodecs.imread("/sdcard/Download/" + String.format("%02d", i)  + "_gestRecog.jpg");
                        motionDetector.detectMotion(img, false);
                        //record if motion or not at this frame
                        motion = motion || motionDetector.detectedMotion ;
                    }

                    step_num = 120;
                    break;
                case 120 : // motion result

                    if (motion)
                    {
                        Log.d(name, "COUCOU");
                        result = "COUCOU";
                    }

                    else
                    {
                        result = "STOP";
                        Log.d(name, "STOP");
                    }


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



    // Define the sequence of recognition
    public class GestureRecogSequence implements Runnable {





        GestureRecogSequence()
        {

            rows = frame.rows();
            cols = frame.cols();
        }

        @Override
        public void run() {


        } //end run

    } //end runnable class



}
