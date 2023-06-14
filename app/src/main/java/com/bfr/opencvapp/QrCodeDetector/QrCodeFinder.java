package com.bfr.opencvapp.QrCodeDetector;

import android.util.Log;

import androidx.annotation.NonNull;


import com.bfr.opencvapp.utils.OpenCvUtils;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.wechat_qrcode.WeChatQRCode;

import java.util.ArrayList;
import java.util.List;

public class QrCodeFinder {
    final private String LOG_TAG = this.getClass().getName();
    final private int KERNEL_SIZE = 50;

    protected WeChatQRCode wechatDetector = null;


    protected Mat image;

    public QrCodeFinder(String wechatDetectorPrototxtPath,
                        String wechatDetectorCaffeModelPath,
                        String wechatSuperResolutionPrototxtPath,
                        String wechatSuperResolutionCaffeModelPath){

        try {
            wechatDetector = new WeChatQRCode(  wechatDetectorPrototxtPath,
                                                wechatDetectorCaffeModelPath,
                                                wechatSuperResolutionPrototxtPath,
                                                wechatSuperResolutionCaffeModelPath);
        } catch (Exception e) {
            Log.e(LOG_TAG, "couldn't initialize wechat detector, check that wechat model files are on the robot");
            e.printStackTrace();
            wechatDetector = null;
        }


        image = new Mat();
    }


    public List<QrCode> findCodes(@NonNull Mat img){
        List<QrCode> qrCodesFound = new ArrayList<>();
        if(img.empty()){
            Log.d(LOG_TAG, "empty image");
            return qrCodesFound;
        }
        if(wechatDetector == null){
            Log.e(LOG_TAG, "detection - wechat detector was not set");
            return qrCodesFound;
        }

        img.copyTo(this.image);
        boolean isCodeDetected = false;
        List<Mat> qrCodesCorner = new ArrayList<Mat>();
        List<String> qrCodesContent = wechatDetector.detectAndDecode(image, qrCodesCorner);
        for(int i=0; i<qrCodesCorner.size(); i++){
            QrCode newQrCode = new QrCode(OpenCvUtils.matToMatOfPoints2f(qrCodesCorner.get(i)), qrCodesContent.get(i));
            if(newQrCode.id >= 0){
                newQrCode.setMask(computeCodeMasks(qrCodesCorner.get(i)));
                qrCodesFound.add(newQrCode);
            }
        }

        return qrCodesFound;
    }

    protected Mat computeCodeMasks(Mat qrCodeCorners){
        Mat mask = Mat.zeros(image.size(), CvType.CV_8U);
        Point Corner1 = new Point(qrCodeCorners.get(0, 0)[0], qrCodeCorners.get(0, 1)[0]);
        Point Corner2 = new Point(qrCodeCorners.get(2, 0)[0], qrCodeCorners.get(2, 1)[0]);
        Mat kernel = new Mat(KERNEL_SIZE, KERNEL_SIZE, CvType.CV_8U);

        Imgproc.rectangle(mask, Corner1, Corner2, new Scalar(255, 255, 255), -1, 8, 0);
        Imgproc.dilate(mask, mask, kernel); // dilate the mask to have some margin

        return mask;
    }
}
