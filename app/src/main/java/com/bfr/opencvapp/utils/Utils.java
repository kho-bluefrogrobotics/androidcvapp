package com.bfr.opencvapp.utils;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;

public class Utils {
    protected final static String LOG_TAG = "OpenCvUtils";

    /*** Path for models ***/
    public static String wechatDetectorPrototxtPath = "/sdcard/Download/detect_2021nov.prototxt";
    public static String wechatDetectorCaffeModelPath = "/sdcard/Download/detect_2021nov.caffemodel";
    public static String wechatSuperResolutionPrototxtPath = "/sdcard/Download/sr_2021nov.prototxt";
    public static String wechatSuperResolutionCaffeModelPath = "/sdcard/Download/sr_2021nov.caffemodel";

    public static String yoloQRCodeCFG = "/sdcard/Download/yolov4-tiny-custom-640.cfg";
    public static String yoloQRCodeWeights = "/sdcard/Download/yolov4-tiny-custom-640_last.weights";


    /*** Camera calibrations ***/
    //wideangle 640x480
//    public static final double[][] cameraCalibrationMatrixCoeff = {{347.1784748095083 , 0       , 326.6795720628966},
//            {0        , 345.1916479410069, 233.1696799590856},
//            {0        , 0       , 1       }};
//    public static final double[] distortionCoeff = {-0.27803360529321036, 0.06339702764223658, -0.000203214885858507, -0.0014783300318126165};

    //    //wide angle 1024x768
    public static final double[][] cameraCalibrationMatrixCoeff = {{614.9288 , 0       , 511.257},
            {0        , 604.4629, 383.5352},
            {0        , 0       , 1       }};
    public static final double[] distortionCoeff = {-0.1024,-0.3817, -0.0044, 0.0025, 0.0247};

    //zoom 640x480
//    private final double[][] cameraCalibrationMatrixCoeff = {{622.1937 , 0       , 399.7594},
//            {0        , 616.9286, 299.5756},
//            {0        , 0       , 1       }};
//    private final double[] distortionCoeff = {0.0598, -0.0309, -0.0038, -0.0033, -0.0872};


    /** transfrom matrix from type Mat to MatOfPoint2f
     *
     * @param matrix - either
     * @return - MatOfPoint2f
     */
    public static MatOfPoint2f matToMatOfPoints2f(Mat matrix){
        MatOfPoint2f resultMat = new MatOfPoint2f();
        List<Point> pointList = new ArrayList<Point>();
        // TODO : manage "all" matrix not only opencv and wechat corners
        if(matrix.size(0) == 1){ // matrix has size 1xN with two channels (corners detected by OpenCv)
            for(int i=0; i<matrix.size(1); i++){
                double[] point = matrix.get(0, i);
                if(point.length == 2){
                    pointList.add(new Point(point[0], point[1]));
                }
                else{
                    Log.e(LOG_TAG + " - matToMatOfPoint2f", "point should have a size of 2 but is " + point.length );
                }
            }
        }
        else if(matrix.size(1) == 2){ // matrix has size N*2 with one channel (corners detected by wechat)
            for(int i=0; i<matrix.size(0); i++){
                double pointU = matrix.get(i, 0)[0];
                double pointV = matrix.get(i, 1)[0];
                pointList.add(new Point(pointU, pointV));
            }
        }
        resultMat.fromList(pointList);
        return resultMat;
    }

    public static void logMat(String logTag, Mat mat){
        String result = "\n";
        for(int i=0; i<mat.size(0); i++){
            for(int j=0; j< mat.size(1); j++){
                result += ("|" + mat.get(i, j)[0]);
            }
            result += "|\n";
        }
        Log.i(logTag, "size " + mat.size() +"\n"+ result);
    }

    public static double distanceBetweenPoints(Point p1, Point p2){
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y- p2.y, 2));
    }
}
