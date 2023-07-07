package com.bfr.opencvapp.QrCodeDetector;

import org.opencv.core.Mat;

public class QRCodePose {

    public Mat qrCodeRotation,
            qrCodeTranslation;

    public QRCodePose(){
        qrCodeRotation = new Mat();
        qrCodeTranslation = new Mat();
    }

    public QRCodePose(Mat rotationMatrix, Mat translationMatrix){
        qrCodeRotation = rotationMatrix.clone();
        qrCodeTranslation = translationMatrix.clone();
    }

}
