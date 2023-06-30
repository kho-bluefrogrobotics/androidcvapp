package com.bfr.opencvapp.QrCodeDetector;

import android.util.Log;

import com.bfr.opencvapp.utils.Utils;
import com.bfr.opencvapp.utils.Pose;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

public class QrCode {
    public Mat rotationMatrix;
    public Mat translationVector;
    public Pose pose;
    public MatOfPoint2f cornersPoints = null;
    public Mat matOfCorners = null;
    public Point center;
    public double size;
    public boolean poseKnown = false;

    public String rawContent = "";
    public int id;
    public double direction;

    protected Mat mask = null;

    // for real world corner coords
    protected QrCodeDescriptor qrCodeDescriptor = new QrCodeDescriptor();


    protected Mat cameraCalibrationMatrix;
    protected MatOfDouble cameraDistortionVector;
    // Pose
    public Mat qrCodeTranslation, qrCodeRotation;

    public QrCode(){
        rotationMatrix = new Mat();
        translationVector = new Mat();
    }

    public QrCode(String rawContent, Mat matOfCorners){
        qrCodeTranslation = new Mat();
        qrCodeRotation = new Mat();

        rotationMatrix = new Mat();
        translationVector = new Mat();

        this.rawContent = rawContent;
        if(matOfCorners.size(0) == 1) { // matrix has size 1xN with two channels (corners detected by OpenCv)
            this.matOfCorners = fromOpenCVtoWeChatCorners(matOfCorners);
        }
        else {
            this.matOfCorners = matOfCorners;
        }
    }

    public QrCode(String rawContent, Mat matOfCorners, boolean poseKnown){
        qrCodeTranslation = new Mat();
        qrCodeRotation = new Mat();

        rotationMatrix = new Mat();
        translationVector = new Mat();

        this.poseKnown = poseKnown;

        this.rawContent = rawContent;
        if(matOfCorners.size(0) == 1) { // matrix has size 1xN with two channels (corners detected by OpenCv)
            this.matOfCorners = fromOpenCVtoWeChatCorners(matOfCorners);
        }
        else {
            this.matOfCorners = matOfCorners;
        }
    }

    public QrCode(Mat matOfCorners){
        qrCodeTranslation = new Mat();
        qrCodeRotation = new Mat();

        rotationMatrix = new Mat();
        translationVector = new Mat();

        this.rawContent = "";
        if(matOfCorners.size(0) == 1) { // matrix has size 1xN with two channels (corners detected by OpenCv)
            this.matOfCorners = fromOpenCVtoWeChatCorners(matOfCorners);
        }
        else {
            this.matOfCorners = matOfCorners;
        }
    }

    public QrCode(Mat matOfCorners, boolean poseKnown){
        qrCodeTranslation = new Mat();
        qrCodeRotation = new Mat();

        rotationMatrix = new Mat();
        translationVector = new Mat();

        this.poseKnown = poseKnown;

        this.rawContent = "";
        if(matOfCorners.size(0) == 1) { // matrix has size 1xN with two channels (corners detected by OpenCv)
            this.matOfCorners = fromOpenCVtoWeChatCorners(matOfCorners);
        }
        else {
            this.matOfCorners = matOfCorners;
        }
    }

//    public QrCode(MatOfPoint2f corners, String rawContent){
//        this();
//        setCorners(corners);
//        this.rawContent = rawContent;
//        setId();
//        setDirection();
//    }

//    public QrCode(MatOfPoint2f corners, Mat rotationVector, Mat translationVector){
//        this();
//        setCorners(corners);
//        setPose(rotationVector, translationVector);
//    }

    private Mat fromOpenCVtoWeChatCorners(Mat wechatCorners)
    {
        //convert to 2xN like weChat
        Mat refactoredCorners = new Mat(4,2, CvType.CV_32FC1);
        refactoredCorners.put(0,0, wechatCorners.get(0,0)[0]);
        refactoredCorners.put(0,1, wechatCorners.get(0,0)[1]);
        refactoredCorners.put(1,0, wechatCorners.get(0,1)[0]);
        refactoredCorners.put(1,1, wechatCorners.get(0,1)[1] );
        refactoredCorners.put(2,0, wechatCorners.get(0,2)[0]);
        refactoredCorners.put(2,1, wechatCorners.get(0,2)[1] );
        refactoredCorners.put(3,0, wechatCorners.get(0,3)[0]);
        refactoredCorners.put(3,1, wechatCorners.get(0,3)[1] );

        return refactoredCorners;
    }
    public String read()
    {
        return rawContent;
    }

    public boolean getPose(double qrSize)
    {
        //set world model
        qrCodeDescriptor.setWorldModel(qrSize);

        cornersPoints = Utils.matToMatOfPoints2f(matOfCorners);
        //convert to 640x480
//        cornersPoints.put(0,0,  new double[]{(int)cornersPoints.get(0,0)[0] *640.0f/1024.0f,cornersPoints.get(0,0)[1]*640.0f/1024.0f});
//        cornersPoints.put(1,0,  new double[]{(int)cornersPoints.get(1,0)[0] *640.0f/1024.0f,cornersPoints.get(1,0)[1]*640.0f/1024.0f});
//        cornersPoints.put(2,0,  new double[]{(int)cornersPoints.get(2,0)[0] *640.0f/1024.0f,(int)cornersPoints.get(2,0)[1]*640.0f/1024.0f});
//        cornersPoints.put(3,0,  new double[]{(int)cornersPoints.get(3,0)[0] *640.0f/1024.0f,(int)cornersPoints.get(3,0)[1]*640.0f/1024.0f});

        cameraCalibrationMatrix = new Mat(3, 3, CvType.CV_64FC1);
        cameraDistortionVector = new MatOfDouble();
        // fill calibration matrix
        for(int row=0; row<3; ++row){
            for(int column=0; column<3; ++column){
                cameraCalibrationMatrix.put(row, column, Utils.cameraCalibrationMatrixCoeff[row][column]);
            }
        }
        // fill camera distortion vector
        cameraDistortionVector.fromArray(Utils.distortionCoeff);

        Calib3d.solvePnP(qrCodeDescriptor.worldModel,
                cornersPoints,
                    cameraCalibrationMatrix,
                    cameraDistortionVector,
                    qrCodeRotation,
                    qrCodeTranslation,
                    false, // check if it should stay to false
                    Calib3d.SOLVEPNP_IPPE_SQUARE);

//        Utils.logMat("matofCorners", matOfCorners);
//        Utils.logMat("corners", cornersPoints);
//        Utils.logMat("qrCodeRotation", qrCodeRotation);
//        Utils.logMat("qrCodeTranslation", qrCodeTranslation);

//        Log.w("coucouTranslate","dist: " +  qrCodeTranslation.get(2,0)[0]);

//        this.angle();

         return true;

    }


    public double getAngle(double qrSize) {

        getPose(qrSize);

        Calib3d.Rodrigues(qrCodeRotation, rotationMatrix);
//        Log.w("coucouangle", "angle =" + -180/3.14*Math.asin(rotationMatrix.get(2, 0)[0]));
//        return -Math.asin(qrCodeRotation.get(2, 0)[0]);
        //return horizontal angle in degree
        return (-180/3.14)*Math.asin(rotationMatrix.get(2, 0)[0]);
    }

    public Mat getTranslationVector() {
        return translationVector;
    }

    public void setMask(Mat mask) {
        this.mask = mask;
    }

    public Mat getMask() {
        return mask;
    }

    public boolean isPoseKnown(){
        return poseKnown;
    }
}
