package com.bfr.opencvapp.utils;


import static org.opencv.core.CvType.CV_8UC3;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class GestureMotionDetect {

    public GestureMotionDetect(String mname, MotionDetector motionDetector) {
        this.name = mname;
        Log.d(name, "GestureMotionDetector object creation : " + name);
        this.motionDetector = motionDetector;
    }

    String name = "GestureMotionDetect";

    MotionDetector motionDetector;
    // index of img to reset
    int resetImNb = 0;

    // number of frames for optical flow
    final int NUMOFFRAMES= 10;
    // buffer to store the sequence of frame for optical flow analysis
    public static ArrayList<Mat> matArray = new ArrayList<Mat>();



    public boolean isStarted = false;

    // recognition sequence vars (steps,...)
    public int step_num =0;
    private int previous_step = 0;
    public boolean go = false;


    // coords of the detected hand bbox
    int left, right, top, bottom;
    // margin to crop the hand in pixel
    int MARGIN = 20;
    public Rect handROI;
    // black background
    Mat black;
    Mat roiInBlack; // subimage at hand roi in black image
    Mat handMat; // subimage at hand roi in original image containing the crop of the hand


    //Thres for minimum size of hand to analyse, in % of the image area
    // Suggestion 0.2 for narrow-angle camera, 0.15 for wide-angle camera
    float THRES_HAND_AREA = 0.05f ;
    // id of the first largest hand visible
    int handID=-1;

    //dims of the input image
    int rows, cols;

    // index of image to record for optical flow
    int imNum=0;

    // motion
//    boolean motion = false;
    public float optFlow = 0.0f;
    // thres for optical flow
//    float THRES_OPT_FLOW_COUCOU = 15.f;
    float THRES_OPT_FLOW_COUCOU = 30.0f;
//    float THRES_OPT_FLOW_COME_HERE = 10.0f;
    float THRES_OPT_FLOW_COME_HERE = 4.0f;
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

    public void setHandROI(Rect handROI)
    {
        this.handROI = handROI;
    }

    public void recognize(Mat input)
    {
        Mat frame = input.clone();
        rows = input.rows();
        cols = input.cols();

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
                        // reset
                        //increment index
                        imNum = 0;
                        // go to next step
                        step_num =100;
                    }
                    return;
                }


            /***/if(step_num==100) { // Open hand start record video for optical flow
                Log.d(name, "Recording for optical flow : ");

                //**** Crop hand image
                // ROI of hand
//                handROI = new Rect( left, top, (right-left), (bottom-top));
                try{
                    // black background
                    black = new Mat(rows,cols, CV_8UC3, new Scalar(0, 0, 0));
                    roiInBlack = black.submat(handROI); // subimage at hand roi in black image
                    handMat = input.submat(handROI); // subimage at hand roi in original image containing the crop of the hand
//                 copy hand crop to black background
                    handMat.copyTo(roiInBlack);
                } catch (Exception e) {
                    go = false;
                    step_num = 0;
                }

//
//
//                frame = black.clone();

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
                if (imNum >= NUMOFFRAMES) {
                    Log.d(name, "END of recording ("+imNum+") -> step 110");
                    step_num = 110;
                }
                else // need the next frame to continue recording
                {
                    return;
                }

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
                for (int i = 6; i < NUMOFFRAMES; i++) {
//                        Mat img = Imgcodecs.imread("/sdcard/Download/" + String.format("%02d", i)  + "_gestRecog.jpg");

                    // get frame from recorded buffer
                    Mat img = matArray.get(i);
                    motionDetector.detectMotion(img, false);

                    //record if motion or not at this frame
//                    motion = motion || motionDetector.detectedMotion;

//                    if(motionDetector.motionOptFlow > optFlow)
//                        optFlow = motionDetector.motionOptFlow;

                    //sum
                    optFlow += motionDetector.motionOptFlow;
                }

                optFlow = optFlow / NUMOFFRAMES;
                Log.w(name, "Optical flow =" +optFlow);

                go = false;
                step_num = 0;
                return;
            }


        } catch (Exception e) {
            Log.e(name, "ERROR :" + Log.getStackTraceString(e));
        }

    }


//    void debugRecord(String folder)
//    {
//
//        Date date = new Date();
//        SimpleDateFormat formatter = new SimpleDateFormat("yyMMddHHmmssSSS");
//        String strDate= formatter.format(date);
//        // create folder if doesn't exist
//        File saveDir = new File("", "/sdcard/Download/"+ folder + "/" + strDate);
//        if(!saveDir.exists()) {
//            // create folder
//            saveDir.mkdirs();
//        }
//
//        for (int i = 0; i<matArray.size(); i++)
//        {
////            Log.d(name, "Saving image " + i);
//            Imgcodecs.imwrite("/sdcard/Download/"+ folder + "/" + strDate+"/" + String.format("%02d", i) + "_gestRecog.jpg", matArray.get(i));
//        }
//
//    } // end record debug


//    void debugSaveImg(String folder, Mat img)
//    {
//
//        Date date = new Date();
//        SimpleDateFormat formatter = new SimpleDateFormat("yyMMddHHmmssSSS");
//        String strDate= formatter.format(date);
//        // create folder if doesn't exist
//        File saveDir = new File("", "/sdcard/Download/"+ folder + "/" + strDate);
//        if(!saveDir.exists()) {
//            // create folder
//            saveDir.mkdirs();
//        }
//            Imgcodecs.imwrite("/sdcard/Download/"+ folder + "/" + strDate+"/_gestRecog.jpg", img);
//
//
//    } // end record debug



    }
