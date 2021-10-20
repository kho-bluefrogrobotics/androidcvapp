package com.bfr.opencvapp;

import android.os.Bundle;

import android.util.Log;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getName();


    private CameraBridgeViewBase mOpenCvCameraView;

    // frame captured by camera
    Mat frame;
    // List of Aruco Markers
    List<Mat> arucoCorners;
    Mat arucoIds ;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // configure camera listener
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);


    }

    // callback at Opencv connected
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully!");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }

    };

    @Override
    public void onResume() {
        super.onResume();
        // OpenCV manager initialization
        OpenCVLoader.initDebug();
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }


    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }


    public void onCameraViewStarted(int width, int height) {
    }


    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        arucoIds = new Mat();
        arucoCorners = new ArrayList<>();

        Mat frame = inputFrame.rgba();

        //convert
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        // Definition of dictionary and params
        Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_APRILTAG_36h11);
        DetectorParameters arucoParams = DetectorParameters.create();

        // Detect Marker
        Aruco.detectMarkers(frame, arucoDict, arucoCorners, arucoIds, arucoParams);

        // if marker detected
        if (arucoCorners.size()>0)
        {
            Log.i("aruco", "Number of detected Markers : "+ arucoCorners.size() ) ;

            // fora each detected marker
            for (int k=0; k<arucoCorners.size(); k++)
            {
                Log.i("aruco", "Read values in marker " + k + " : "+ arucoIds.get(k,0)[0] ) ;
                // coordinates of four corners
                int x1 = (int) arucoCorners.get(k).get(0,0)[0];
                int y1 = (int) arucoCorners.get(k).get(0,0)[1];
                int x2 = (int) arucoCorners.get(k).get(0,1)[0];
                int y2 = (int) arucoCorners.get(k).get(0,1)[1];
                int x3 = (int) arucoCorners.get(k).get(0,2)[0];
                int y3 = (int) arucoCorners.get(k).get(0,2)[1];
                int x4 = (int) arucoCorners.get(k).get(0,3)[0];
                int y4 = (int) arucoCorners.get(k).get(0,3)[1];
                // draw corners
                Imgproc.circle(frame, new Point(x1, y1),  1, new Scalar(0,255,0)  ,5);
                Imgproc.circle(frame, new Point(x2, y2),  1, new Scalar(255,0,0)  ,5);
                Imgproc.circle(frame, new Point(x3, y3),  1, new Scalar(0,0,255)  ,5);
                Imgproc.circle(frame, new Point(x4, y4),  1, new Scalar(125,0,125)  ,5);

            } // next marker

        } // end if marker detected

        return frame;
    } // end function

    public void onCameraViewStopped() {
    }

}