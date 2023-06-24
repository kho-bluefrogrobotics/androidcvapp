package com.bfr.opencvapp.QrCodeDetector;

import android.util.Log;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

public class QrCodeDetector {
    protected final static String LOG_TAG = "QrCodeDetector";
//    protected QrCodeFinder codeFinder = null;
//    protected QrCodePoseEstimator poseEstimator = null;
    boolean useFinder = true;
    public QrCodeDetector(boolean useFinder){
        this.useFinder = useFinder;
    }

    public void setupPoseEstimator(double[][] calibrationMatrixCoeffs, double[] distortionCoeffs, double qrCodeSize){
//        if(poseEstimator == null){
//            poseEstimator = new QrCodePoseEstimator(calibrationMatrixCoeffs, distortionCoeffs);
//            poseEstimator.setupQrCodeDescription(qrCodeSize);
//        }
    }

    public void setupFinder(String wechatDetectorPrototxtPath,
                              String wechatDetectorCaffeModelPath,
                              String wechatSuperResolutionPrototxtPath,
                              String wechatSuperResolutionCaffeModelPath){
//        if(codeFinder == null){
//            codeFinder = new QrCodeFinder(wechatDetectorPrototxtPath, wechatDetectorCaffeModelPath, wechatSuperResolutionPrototxtPath, wechatSuperResolutionCaffeModelPath);
//        }
    }

    public List<QrCode> detect(Mat img, int numberOfCode) throws Exception {
        List<QrCode> detectedQrCodes = new ArrayList<QrCode>();
//        if(poseEstimator == null){
//            Log.e(LOG_TAG, "QrCode pose estimator not set");
//            throw new Exception("QrCodeDetector : QrCode pose estimator not set");
//        }

//        if(useFinder && codeFinder == null){
//            Log.e(LOG_TAG, "QrCode finder not set");
//            throw new Exception("QrCodeDetector : QrCode finder not set");
//        }

        if(!img.empty()){
            if(useFinder){
//                detectedQrCodes = codeFinder.findCodes(img);
                if(detectedQrCodes.size() > numberOfCode){
                    detectedQrCodes = detectedQrCodes.subList(0, numberOfCode); // take only the number of detected code we want
                }
            }
//            poseEstimator.estimateCodesPose(img, detectedQrCodes);
        }

        return detectedQrCodes;
    }

}
