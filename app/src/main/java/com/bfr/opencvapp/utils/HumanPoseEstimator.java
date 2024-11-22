package com.bfr.opencvapp.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/** Human pose estimator using the Blazepose Mediapipe implementation*/
public class HumanPoseEstimator {

    private final String TAG = "HumanPoseEstimation";

    PoseLandmarker poseLandmarker;

    public HumanPoseEstimator(Context context){

        try{

            /** Mediapipe model creation*/
            BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder()
                    .setModelAssetPath("nnmodels/pose_landmarker_full.task")
                    .setDelegate(Delegate.GPU);
            BaseOptions baseOptions  = baseOptionsBuilder.build();

            //empiric parameters
            PoseLandmarker.PoseLandmarkerOptions.Builder poseOptionsBuilder = PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMinPoseDetectionConfidence(0.5f)
                    .setMinTrackingConfidence(0.01f)
                    .setMinPosePresenceConfidence(0.01f)
                    .setNumPoses(1)
                    .setRunningMode(RunningMode.IMAGE);

            PoseLandmarker.PoseLandmarkerOptions poseOptions  = poseOptionsBuilder.build();

            poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions);

        }
        catch (Exception e)
        {
            Log.e(TAG, "Error Creating the Human pose estimator " + Log.getStackTraceString(e) );
        }

    }


    /**
     * get the pose in the image
     * @param frame original image in Mat format
     * @return a Pose (landmarks)
     */
    public HumanPose recognizeImage(Mat frame) {

        //convert to bitmap
        Mat input = frame.clone();
        //TODO: check if necessary to convert color
        Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2BGR);
        Bitmap bitmapImagefull = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(input, bitmapImagefull);

        // Convert to Mediapipe image
        MPImage mpImage = new BitmapImageBuilder(bitmapImagefull).build() ;

        // Pose estimation
        PoseLandmarkerResult poseLandmarkerResult = poseLandmarker.detect(mpImage);

        if (poseLandmarkerResult.landmarks().size()<=0)
        {
//            Log.w(TAG, "NO Human Pose DETECTED");
            return null;
        }

        //object to return
        HumanPose humanPose = new HumanPose();
        humanPose.landmarks = poseLandmarkerResult.landmarks().get(0);

        return humanPose;
    }

}
