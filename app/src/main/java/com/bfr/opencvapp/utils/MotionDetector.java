package com.bfr.opencvapp.utils;


import static com.bfr.opencvapp.utils.Utils.Color._RED;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;

public class MotionDetector {

    private final String TAG = "_VISION motionDetector";

    //current and previous frame
    Mat currFrame, prevFrame;
    private int frameHeight, frameWidth =0;
    public int frameCount =0;
    // Optical flow
    Mat flow;
    Mat magnitude, angle , magnNorm;
    // resize for better performances
    Size mSize = new Size(480, 320);

    // movement detection
    // camera is processing opticalFlow
    public float motionOptFlow = 0.0f;
    // relative position in the image, in % (with origin at the top left corner of the image)
    public float motionX, motionY;
    // Threshold for motion detection
    public float motionThres = 10.0f;
    // if optical flow > Thres
    public boolean detectedMotion = false;

    // for display
    public Mat displayMat;
    public boolean readyToDisplay = false;
    Point pt1 = new Point();
    Point pt2 = new Point();

    private boolean wDebug=false;


    public MotionDetector(){
        magnitude = new Mat();
        angle = new Mat();
        magnNorm = new Mat();
    }

   public void detectMotion(Mat frame, boolean constructVisualizationImage)
    {
        if(wDebug)
            Log.d(TAG, "Begining of detection at " + System.currentTimeMillis());

        // frame size
        frameHeight =frame.height();
        frameWidth =frame.width();

        // count frame number
        frameCount +=1;
        //reset
        if (frameCount >99999)
            frameCount =2;

        // Start after 1st frame
        if (frameCount >1)
        {
            /*** Optical flow***/

            // save previous frame
            prevFrame = currFrame;
            // cature frame from camera
            currFrame = frame;
            // convert to gray
            Imgproc.cvtColor(currFrame, currFrame, Imgproc.COLOR_BGR2GRAY);
            // resize for better performances
            Imgproc.resize(currFrame, currFrame, mSize);

            //flow
            flow = new Mat(currFrame.size(), CvType.CV_32FC2);
            //compute optical flow
            Video.calcOpticalFlowFarneback(prevFrame, currFrame, flow,
                    0.5, 3, 15, 3, 5, 1.2, 0);

            // visualization
            ArrayList<Mat> flowParts = new ArrayList<>(2);
            // resize to display
            Imgproc.resize(flow, flow, new Size(frameWidth, frameHeight));
            Core.split(flow, flowParts);

            Core.cartToPolar(flowParts.get(0), flowParts.get(1), magnitude, angle,true);
            Core.normalize(magnitude, magnNorm,0.0,1.0, Core.NORM_MINMAX);
            float factor = (float) ((1.0/360.0)*(180.0/255.0));
            Mat newAngle = new Mat();
            Core.multiply(angle, new Scalar(factor), newAngle);
            //build hsv image
            ArrayList<Mat> _hsv = new ArrayList<>() ;
            Mat hsv = new Mat(), hsv8 = new Mat(), bgr = new Mat();
            _hsv.add(newAngle);
            _hsv.add(Mat.ones(angle.size(), CvType.CV_32F));
            _hsv.add(magnNorm);
            Core.merge(_hsv, hsv);
            hsv.convertTo(hsv8, CvType.CV_8U, 255.0);
            Imgproc.cvtColor(hsv8, bgr, Imgproc.COLOR_HSV2BGR);

            Log.d(TAG, "Farneback optical flow Max = " + Core.minMaxLoc(magnitude).maxVal
                    + " at " + Core.minMaxLoc(magnitude).maxLoc );
            // assign values
            motionOptFlow = (float) Core.minMaxLoc(magnitude).maxVal;
            // relative position in the image
            motionX =((float) Core.minMaxLoc(magnitude).maxLoc.x) / frameWidth;
            motionY = ((float)Core.minMaxLoc(magnitude).maxLoc.y)/ frameHeight;
            // if > Thres
            if (motionOptFlow >= motionThres)
                detectedMotion = true;
            else
                detectedMotion = false;

            // display
            if(constructVisualizationImage) {
                if(wDebug)
                    Log.d(TAG, "Display motion "+ Core.minMaxLoc(magnitude).maxLoc.x
                    + " " + Core.minMaxLoc(magnitude).maxLoc.y
                    + " / " + frameWidth
                    + " " + frameHeight);

                displayMat = frame.clone();
                //draw circle at motion location
                if(detectedMotion){
                    pt1.x= ((float) motionX * displayMat.width());
                    pt1.y= ((float) motionY * displayMat.height());
                    Imgproc.circle(displayMat, pt1, 20, _RED
                            , 2);
                    Imgproc.putText(displayMat, "Motion", pt1,
                            2, 0.8, _RED);
                }

                readyToDisplay=true;
            }

        }
        else //1st frame
        {
            // cature frame from camera
            currFrame = frame;

            // reset
            motionOptFlow = 0.0f;
            detectedMotion = false;

            // convert to gray
            Imgproc.cvtColor(currFrame, currFrame, Imgproc.COLOR_BGR2GRAY);
            // resize for better performances
            Imgproc.resize(currFrame, currFrame, mSize);

        }
        if(wDebug)
            Log.d(TAG, "End of detection at " + System.currentTimeMillis());
    } // end mvt detection



}
