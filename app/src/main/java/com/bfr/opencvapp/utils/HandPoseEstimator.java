package com.bfr.opencvapp.utils;


import static com.bfr.opencvapp.utils.HumanPoseLandmarks.*;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;


/** Hand Face Human object detector based on a Mobilenetv2-SSD network*/
public class HandPoseEstimator {

    private final String TAG = "HandPoseEstimator";

    //Params for TFlite interpreter
    private final boolean IS_QUANTIZED = false;
    private final Size INPUT_SIZE = new Size(224,224);

    private final int BATCH_SIZE = 1;
    private final int PIXEL_SIZE = 3;

    HandLandmarker handLandmarker;

    public enum FINGER{
        THUMB,
        INDEX,
        MIDDLE,
        RING,
        PINKIE
    }

    // confidence level of human detection for doublecheck with Movenet
    public float humanConfidence = 0.0f;

    public HandPoseEstimator(Context context){

        try{

            /** Mediapipe */

            BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder()
                    .setModelAssetPath("nnmodels/hand_landmarker.task")
                    .setDelegate(Delegate.GPU);
            BaseOptions baseOptions  = baseOptionsBuilder.build();

            HandLandmarker.HandLandmarkerOptions.Builder handOptionsBuilder = HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setNumHands(1)
                    .setMinHandDetectionConfidence(0.5f)
                    .setMinTrackingConfidence(0.01f)
                    .setMinHandPresenceConfidence(0.01f)
                    .setRunningMode(RunningMode.IMAGE);

            HandLandmarker.HandLandmarkerOptions handOptions  = handOptionsBuilder.build();

            handLandmarker = HandLandmarker.createFromOptions(context, handOptions);

        }
        catch (Exception e)
        {
            Log.e(TAG, "Error Creating the Handpose " + Log.getStackTraceString(e) );
        }

    }

    /**
     * get the detected objects in the image
     * @param frame original image in Mat format
     * @return array of detections
     */
    public HandPose recognizeImage(Mat frame)
    {
        return recognizeImage(frame, -1);
    }
    /**
     * get the detected objects in the image
     * @param frame original image in Mat format
     * @param targetHand only returns something if the detected hand corresponds to left or right target
     * @return array of detections
     */
    public HandPose recognizeImage(Mat frame, int targetHand) {

//        Log.i(TAG, "Starting Hand pose estimation" );

        HandPose handPose = new HandPose();

        //convert to bitmap
        Mat input = frame.clone();
        //TODO: check if necessary to convert color
        Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2BGR);
        Bitmap bitmapImagefull = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(input, bitmapImagefull);

//        try (FileOutputStream out = new FileOutputStream("/sdcard/Download/111.png")) {
//            bitmapImagefull.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
//            // PNG is a lossless format, the compression factor (100) is ignored
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        Imgcodecs.imwrite("/sdcard/Download/111.jpg", input);
        MPImage mpImage = new BitmapImageBuilder(bitmapImagefull).build() ;

        HandLandmarkerResult handLandmarkerResult =handLandmarker.detect(mpImage);

        if (handLandmarkerResult.landmarks().size()<=0)
        {
            Log.w(TAG, "NO HAND DETECTED");
            return null;
        }

        // if requested to check hand id
        if(targetHand>-1)
        {
            // for each hand
            for (int handId=0; handId<handLandmarkerResult.landmarks().size(); handId++)
            {
                if(targetHand==RIGHT_WRIST && handLandmarkerResult.handednesses().get(handId).get(0).categoryName().toUpperCase().contains("RIGHT") ){
                    continue;
                }
                else if(targetHand==LEFT_WRIST && handLandmarkerResult.handednesses().get(handId).get(0).categoryName().toUpperCase().contains("LEFT") ){
                    continue;
                }

                handPose.landmarks = handLandmarkerResult.landmarks().get(handId);
                handPose.handeness = handLandmarkerResult.handednesses().get(handId);

                //break at this hand
                break;
            }
//            Log.i(TAG, "Result size ="+ handLandmarkerResult.landmarks().get(0).size());


        }
        else //just take the first detected hand
        {
            handPose.landmarks = handLandmarkerResult.landmarks().get(0);
            handPose.handeness = handLandmarkerResult.handednesses().get(0);
        }

        return handPose;
    }



}
