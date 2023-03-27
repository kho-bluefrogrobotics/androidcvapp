package com.bfr.opencvapp.utils;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.util.List;
import java.util.concurrent.ExecutionException;

public class MLKitFaceDetector {

    private final String TAG = "MLKitFaceDetector";

    // Real-time contour detection of multiple faces
    FaceDetectorOptions options =
            new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                    .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                    .build();

    // Face detector
    private FaceDetector faceDetector = FaceDetection.getClient(options);

    // MLKit image
    InputImage inputImage;

    // Detected face
    Face resultFace = null;

    Task result;

    public Face detectSingleFaceFromBitmap(Bitmap inputBitmap)
    {
        //convert to InputImage
        inputImage = InputImage.fromBitmap(inputBitmap, 0);
        result = faceDetector.process(inputImage)
                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
//                        Log.i(TAG, String.valueOf(System.currentTimeMillis())+ " found faces : " + String.valueOf(faces.size()) );
                        // if found face
                        if (faces.size()>0) {

                            // get first (and only) face
                            resultFace = faces.get(0);

                            /* can be used to get the smiling proba, eyes open, face orientation like this:
                            faces.get(0).getSmilingProbability();
                            faces.get(0).getLeftEyeOpenProbability();
                            faces.get(0).getRightEyeOpenProbability();
                            faces.get(0).getHeadEulerAngleX();
                             */
                        }//end if found face
                    } // end onSucess
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        Log.w(TAG, "Error during detection: "  + e);
                    }
                }); // end process image

        try {
            Tasks.await(result);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return resultFace;

    } //end detectFace


    public Face detectSingleFaceFromByteArray(byte inputBuffer[], int width, int height)
    {
        //convert to InputImage
        inputImage = InputImage. fromByteArray (inputBuffer, 0, 2, width, height );
        result = faceDetector.process(inputImage)
                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
//                        Log.i(TAG, String.valueOf(System.currentTimeMillis())+ " found faces : " + String.valueOf(faces.size()) );
                        // if found face
                        if (faces.size()>0) {

                            // get first (and only) face
                            resultFace = faces.get(0);

                            /* can be used to get the smiling proba, eyes open, face orientation like this:
                            faces.get(0).getSmilingProbability();
                            faces.get(0).getLeftEyeOpenProbability();
                            faces.get(0).getRightEyeOpenProbability();
                            faces.get(0).getHeadEulerAngleX();
                             */
                        }//end if found face
                    } // end onSucess
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        Log.w(TAG, "Error during detection: "  + e);
                    }
                }); // end process image

        try {
            Tasks.await(result);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return resultFace;

    } //end detectFace

}
