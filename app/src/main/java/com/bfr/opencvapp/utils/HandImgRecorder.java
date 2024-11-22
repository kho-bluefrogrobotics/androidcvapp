package com.bfr.opencvapp.utils;

import android.os.Handler;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoWriter;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public class HandImgRecorder {

    String TAG = "HandIMGRecorder";

    public HandImgRecorder(HandPoseEstimator handPoseEstimator){
        this.handPoseEstimator = handPoseEstimator;
        handPose = new HandPose();
    }

    HandPoseEstimator handPoseEstimator;
    HandPose handPose = null;
    List<Mat> listOfMat = new ArrayList<>();
    int imgIdx = 0;
    int NUM_OF_IMG = 15;


    /** record for debog*/

    LocalDateTime myDateObj = LocalDateTime.now();
    DateTimeFormatter myFormatObj = DateTimeFormatter.ofPattern("yyMMddHHmmss");
    VideoWriter videoWriter;
    String formattedDate = myDateObj.format(myFormatObj);
    String debugFileName = "/storage/emulated/0/Download/" + formattedDate + "_trackingDebug.avi" ;
    int fourcc =-1;
    /****/

    Handler poseEstHandler = new Handler();
    //Element to display frame from Camera
    Mat imgToAdd;
    private  Runnable poseEstimation = new Runnable() {
        @Override
        public void run() {
            try {
                synchronized (listOfMat){
                    handPose = handPoseEstimator.recognizeImage(listOfMat.get(listOfMat.size()-1));
                    Log.i(TAG, "handpose " + (listOfMat.size()-1));
                    Imgproc.circle(listOfMat.get(listOfMat.size()-1),
                            new Point(handPose.landmarks.get(12).x() * listOfMat.get(listOfMat.size()-1).cols(), handPose.landmarks.get(12).y() * listOfMat.get(listOfMat.size()-1).rows()),
                            2, new Scalar(255, 255, 0), 3);

                    videoWriter.write(listOfMat.get(listOfMat.size()-1));
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    };

    public void init(int width, int height){
        myDateObj = LocalDateTime.now();
        formattedDate = myDateObj.format(myFormatObj);
        debugFileName = "/storage/emulated/0/Download/trackingdebug/" + formattedDate + "_trackingDebug.avi" ;
        fourcc = VideoWriter.fourcc('M','J','P','G');
        Log.i(TAG, "videowriter creation " + debugFileName);
        videoWriter = new VideoWriter(debugFileName, fourcc,
                2, new Size(width, height));
        Log.i(TAG, "Ready to save video " +width+"x"+height);

        listOfMat.clear();

    }
    public void recImg(Mat img){

        synchronized (listOfMat){

            listOfMat.add(img);
            Log.i(TAG, "addedimg " + (listOfMat.size()-1));
            // queue handpose
            poseEstHandler.post(poseEstimation);
        }
        //Pose estimation

    }

    public void saveVideo(){
        try {
            videoWriter.release();
        } catch (Exception e) {
        }
    }

}
