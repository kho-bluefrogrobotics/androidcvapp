package com.bfr.opencvapp.utils;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;

public class OpenCvUtils {
    protected final static String LOG_TAG = "OpenCvUtils";

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
            for(int i=0; i<matrix.size(1); i++){
                double pointU = matrix.get(i, 0)[0];
                double pointV = matrix.get(i, 1)[0];

                pointList.add(new Point(pointU, pointV));
            }
        }
        resultMat.fromList(pointList);
        return resultMat;
    }

    public static void logMat(String logTag, Mat mat){
        String result = "\n\n";
        for(int i=0; i<mat.size(0); i++){
            for(int j=0; j< mat.size(1); j++){
                result += ("|" + mat.get(i, j)[0]);
            }
            result += "|\n";
        }
        Log.i(logTag, result);
    }

    public static double distanceBetweenPoints(Point p1, Point p2){
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y- p2.y, 2));
    }
}
