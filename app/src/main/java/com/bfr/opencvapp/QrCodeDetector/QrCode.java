package com.bfr.opencvapp.QrCodeDetector;

import android.util.Log;

import com.bfr.opencvapp.utils.QRCodeUtils;
import com.bfr.opencvapp.utils.Pose;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

import java.util.List;

public class QrCode {
    public Mat rotationMatrix;
    public Mat translationVector;
    public Pose pose;
    public MatOfPoint2f corners = null;
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
    // camera calibration
    //wideangle 640x480
//    private final double[][] cameraCalibrationMatrixCoeff = {{347.1784748095083 , 0       , 326.6795720628966},
//            {0        , 345.1916479410069, 233.1696799590856},
//            {0        , 0       , 1       }};
//    private final double[] distortionCoeff = {-0.27803360529321036, 0.06339702764223658, -0.000203214885858507, -0.0014783300318126165};

//    //wide angle 1024x768
    private final double[][] cameraCalibrationMatrixCoeff = {{614.9288 , 0       , 511.257},
        {0        , 604.4629, 383.5352},
        {0        , 0       , 1       }};
    private final double[] distortionCoeff = {-0.1024,-0.3817, -0.0044, 0.0025, 0.0247};


    //zoom 640x480
//    private final double[][] cameraCalibrationMatrixCoeff = {{622.1937 , 0       , 399.7594},
//            {0        , 616.9286, 299.5756},
//            {0        , 0       , 1       }};
//    private final double[] distortionCoeff = {0.0598, -0.0309, -0.0038, -0.0033, -0.0872};

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
        this.matOfCorners = matOfCorners;
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

    public String read()
    {
        return rawContent;
    }

    public boolean getPose(double qrSize)
    {
        //set world model
        qrCodeDescriptor.setWorldModel(qrSize);


        corners = QRCodeUtils.matToMatOfPoints2f(matOfCorners);

        //convert to 640x480
////        Log.w("coucou", "before " + corners.get(0,0)[0] +","+corners.get(0,0)[1]);
//        corners.put(0,0,  new double[]{(int)corners.get(0,0)[0] *640.0f/1024.0f,corners.get(0,0)[1]*640.0f/1024.0f});
////        Log.w("coucou", "after " + corners.get(0,0)[0] +","+corners.get(0,0)[1]);
//        corners.put(1,0,  new double[]{(int)corners.get(1,0)[0] *640.0f/1024.0f,corners.get(1,0)[1]*640.0f/1024.0f});
//        corners.put(2,0,  new double[]{(int)corners.get(2,0)[0] *640.0f/1024.0f,(int)corners.get(2,0)[1]*640.0f/1024.0f});
//        corners.put(3,0,  new double[]{(int)corners.get(3,0)[0] *640.0f/1024.0f,(int)corners.get(3,0)[1]*640.0f/1024.0f});

        cameraCalibrationMatrix = new Mat(3, 3, CvType.CV_64FC1);
        cameraDistortionVector = new MatOfDouble();
        // fill calibration matrix
        for(int row=0; row<3; ++row){
            for(int column=0; column<3; ++column){
                cameraCalibrationMatrix.put(row, column, cameraCalibrationMatrixCoeff[row][column]);
            }
        }
        // fill camera distortion vector
        cameraDistortionVector.fromArray(distortionCoeff);





        Calib3d.solvePnP(qrCodeDescriptor.worldModel,
                    corners,
                    cameraCalibrationMatrix,
                    cameraDistortionVector,
                    qrCodeRotation,
                    qrCodeTranslation,
                    false, // check if it should stay to false
                    Calib3d.SOLVEPNP_IPPE_SQUARE);
//                    Calib3d.SOLVEPNP_ITERATIVE);

        QRCodeUtils.logMat("matofCorners", matOfCorners);
        QRCodeUtils.logMat("corners", corners);
        QRCodeUtils.logMat("qrCodeRotation", qrCodeRotation);
        QRCodeUtils.logMat("qrCodeTranslation", qrCodeTranslation);

        Log.w("coucouTranslate",
                "dist: " +  qrCodeTranslation.get(2,0)[0]);

        this.angle();

         return true;

    }




//    public void setPose(Mat rotationVector, Mat translationVector){
//        Calib3d.Rodrigues(rotationVector, rotationMatrix);
//        this.translationVector = translationVector;
//    }


//    public void setCorners(MatOfPoint2f corners) {
//        this.corners = corners;
//        setImageProperties();
//    }

//    protected void setImageProperties(){
//        if(corners != null){
//            center = centroid(corners);
//            List<Point> cornerList = corners.toList();
//            size = QRCodeUtils.distanceBetweenPoints(cornerList.get(0), cornerList.get(1));
//        }
//    }
//
//    protected Point centroid(MatOfPoint2f points){
//        Point centroid = new Point();
//        double xSum = 0;
//        double ySum = 0;
//        List<Point> pointList = points.toList();
//        int pointNumber = pointList.size();
//        for(Point p : pointList){
//            xSum += p.x;
//            ySum += p.y;
//        }
//
//        centroid.x = xSum / pointNumber;
//        centroid.y = ySum / pointNumber;
//
//        return centroid;
//    }
//
//    public boolean setPose(Pose robotPose){
//        if(translationVector.empty()){
//            return false;
//        }
//        double zQ = translationVector.get(2, 0)[0] / 1000;
//        double xQ = translationVector.get(0, 0)[0] / 1000;
//        pose = new Pose();
//        pose.x = robotPose.x + (zQ * Math.cos(robotPose.theta) + xQ * Math.sin(robotPose.theta));
//        pose.y = robotPose.y + (zQ * Math.sin(robotPose.theta) - xQ * Math.cos(robotPose.theta));
//        pose.theta = robotPose.theta - angle();
//        return true;
//    }
//
//    protected void setId(){
//        if(!rawContent.equals("")){
//            String[] dataArray = rawContent.split(";");
//            id = Integer.parseInt(dataArray[0]);
//        }
//        else{
//            id = -1;
//        }
//    }
//
//    protected void setDirection(){
//        if(!rawContent.equals("")){
//            String[] dataArray = rawContent.split(";");
//            switch (dataArray[1]){
//                case "LEFT":
//                    direction = (Math.PI / 2.0);
//                    break;
//
//                case "RIGHT":
//                    direction = -(Math.PI / 2.0);
//                    break;
//
//                default:
//                    direction = 0;
//                    Log.i("QR_CODE", "unknown QrCode content");
//                    break;
//            }
//        }
//    }

    public double angle() {
        Calib3d.Rodrigues(qrCodeRotation, rotationMatrix);
        Log.w("coucouangle", "angle =" + -180/3.14*Math.asin(rotationMatrix.get(2, 0)[0]));
//        return -Math.asin(qrCodeRotation.get(2, 0)[0]);
        return -180/3.14*Math.asin(rotationMatrix.get(2, 0)[0]);
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
