package com.bfr.opencvapp.QrCodeDetector;

import android.util.Log;

import com.bfr.opencvapp.utils.OpenCvUtils;
import com.bfr.opencvapp.utils.Pose;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

import java.util.List;

public class QrCode {
    public Mat rotationMatrix;
    public Mat translationVector;
    public Pose pose;
    public MatOfPoint2f corners = null;
    public Point center;
    public double size;
    public boolean poseKnown = false;

    public String rawContent = "";
    public int id;
    public double direction;

    protected Mat mask = null;

    public QrCode(){
        rotationMatrix = new Mat();
        translationVector = new Mat();
    }

    public QrCode(MatOfPoint2f corners, String rawContent){
        this();
        setCorners(corners);
        this.rawContent = rawContent;
        setId();
        setDirection();
    }

    public QrCode(MatOfPoint2f corners, Mat rotationVector, Mat translationVector){
        this();
        setCorners(corners);
        setPose(rotationVector, translationVector);
    }


    public String read()
    {
        return null;
    }


    public void setPose(Mat rotationVector, Mat translationVector){
        Calib3d.Rodrigues(rotationVector, rotationMatrix);
        this.translationVector = translationVector;
    }


    public void setCorners(MatOfPoint2f corners) {
        this.corners = corners;
        setImageProperties();
    }

    protected void setImageProperties(){
        if(corners != null){
            center = centroid(corners);
            List<Point> cornerList = corners.toList();
            size = OpenCvUtils.distanceBetweenPoints(cornerList.get(0), cornerList.get(1));
        }
    }

    protected Point centroid(MatOfPoint2f points){
        Point centroid = new Point();
        double xSum = 0;
        double ySum = 0;
        List<Point> pointList = points.toList();
        int pointNumber = pointList.size();
        for(Point p : pointList){
            xSum += p.x;
            ySum += p.y;
        }

        centroid.x = xSum / pointNumber;
        centroid.y = ySum / pointNumber;

        return centroid;
    }

    public boolean setPose(Pose robotPose){
        if(translationVector.empty()){
            return false;
        }
        double zQ = translationVector.get(2, 0)[0] / 1000;
        double xQ = translationVector.get(0, 0)[0] / 1000;
        pose = new Pose();
        pose.x = robotPose.x + (zQ * Math.cos(robotPose.theta) + xQ * Math.sin(robotPose.theta));
        pose.y = robotPose.y + (zQ * Math.sin(robotPose.theta) - xQ * Math.cos(robotPose.theta));
        pose.theta = robotPose.theta - angle();
        return true;
    }

    protected void setId(){
        if(!rawContent.equals("")){
            String[] dataArray = rawContent.split(";");
            id = Integer.parseInt(dataArray[0]);
        }
        else{
            id = -1;
        }
    }

    protected void setDirection(){
        if(!rawContent.equals("")){
            String[] dataArray = rawContent.split(";");
            switch (dataArray[1]){
                case "LEFT":
                    direction = (Math.PI / 2.0);
                    break;

                case "RIGHT":
                    direction = -(Math.PI / 2.0);
                    break;

                default:
                    direction = 0;
                    Log.i("QR_CODE", "unknown QrCode content");
                    break;
            }
        }
    }

    public double angle() {
        return -Math.asin(rotationMatrix.get(2, 0)[0]);
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
