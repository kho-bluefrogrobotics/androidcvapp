package com.bfr.opencvapp;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.RemoteException;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.RelativeLayout;
import android.widget.Switch;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

//import com.bfr.buddysdk.BuddySDK;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;
import org.opencv.tracking.TrackerCSRT;
import org.opencv.tracking.TrackerKCF;
import org.opencv.utils.Converters;
import org.opencv.video.Tracker;

import org.opencv.video.Video;
import org.opencv.videoio.VideoWriter;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;



public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "COLORRecog";

    private CameraBridgeViewBase mOpenCvCameraView;

    // directory where the model files are saved for face detection
    private String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();



    //button to start tracking
    private Button initBtn;

    // Face UI
    private RelativeLayout BuddyFace;
    private Switch noSwitch;
    private CheckBox hideFace;

    private CheckBox trackingCheckBox;
    private CheckBox fastTrackingChckBox;
    JavaCameraView cameraView;


    //********************  image ***************************

    // tHRESHOLD OF FACE DETECTION
    final double THRESHOLD = 0.75;
    // Tracker
    Tracker mytracker;
    // frame captured by camera
    Mat frame;
    // List of detected faces
    Mat blob, detections;
    // status
    boolean istracking = false;
    int frame_count = 0;
    // Option to switch between fast but low performance Tracking, or Slower but more Robust Tracking
    boolean fastTracking = false;

    // last detection
    private int lastDetectYes, lastDetectNo;

    // Neural net for detection
    private Net net;

    // context
    Context mycontext = this;

    //Video writer
    private VideoWriter videoWriter;
    private boolean isrecording;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // run only in Landscape mode
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        setContentView(R.layout.activity_main);

        // Check permissions
        if(ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED)
        {   //Request permission
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }

        // configure camera listener
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        // link to UI
        initBtn = findViewById(R.id.initButton);
        noSwitch = findViewById(R.id.enableNoSwitch);
        hideFace = findViewById(R.id.visibleCheckBox);
        trackingCheckBox = findViewById(R.id.trackingBox);
        fastTrackingChckBox = findViewById(R.id.fastTracking);

        //start tracking button
        initBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.i("Tracking", "Tracking is starting");
            }
        });





        //**************** Callbacks for buttons

        //callback show face
        hideFace.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                //if checked
                if (hideFace.isChecked())
                {   // set tranparent
                    BuddyFace.setAlpha(0.25F);
                }
                else // unchecked
                {// set opaque
                    BuddyFace.setAlpha(1.0F);
                } // end if checked
            } // end onchange
        });// end listener

        //calbacks for Enable button
        noSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {


            } // end onChecked
        }); // end listener

        //tracking
        trackingCheckBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                //reset
                if (!trackingCheckBox.isChecked())
                {
                    // reset grafcet

                    // close record file
                    isrecording = false;
                    videoWriter.release();

                }
                else // checked
                {
                    isrecording = true;


                }

            }
        });

        //tracking
        fastTrackingChckBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                //Enable fast tracking
                if (b)
                {
                    fastTracking = true;
                }
                else // checked
                {
                    fastTracking = false;
                }
            }
        });

    } // End onCreate

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
    public void onPause() {
        super.onPause();

    }

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


        // Tracker init
        if (fastTracking)
            mytracker = TrackerKCF.create();
        else
            mytracker = TrackerCSRT.create();

        // Init write video file
        videoWriter = new VideoWriter("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
                25.0D, new Size(800, 600));
        videoWriter.open("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
                25.0D,  new Size( 800,600));

    }

    //current and previous frame
    Mat curr_frame, prev_frame;
    // Optical flow
    Mat flow;
    // resize for better performances
    Size mSize = new Size(48, 32);
    //crop for better performances from 800x600
    Rect imgROI = new Rect(new Point(250, 250), new Point(550, 380) );
    // Knn for color recognition
    private KNearest colorClassifier;
    boolean alreadyTrained = false;
    // Mat of training data for 6 colors : black, white, blue, green, red, yellow
    Mat trainData ;
    // sample to test
    Mat toTest;
    Mat data;
    //result
    Mat res;

    // Colors to recognize
    Scalar _RED = new Scalar(150,20,20);
    Scalar _BLUE = new Scalar(0,0,150);
    Scalar _GREEN = new Scalar(0,150,0);
    Scalar _YELLOW = new Scalar(150,100,0);
    Scalar _WHITE = new Scalar(170,170,170);
    Scalar _BLACK = new Scalar(0,0,0);


    // colored pixel counter
    int coloredPixels[] = {0,0, 0, 0, 0,0};

    List<Float> tmp = new ArrayList<Float>();

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        frame = inputFrame.rgba();

        // if not already trained
        if(alreadyTrained==false)
        {





            // instantiate KNN classifier
            colorClassifier = KNearest.create();
            // setup training data
            trainData = new Mat();
            List<Integer> trainLabels = new ArrayList<Integer>();

            List<String> allLines;
            // add red color
            try {
                allLines = Files.readAllLines(Paths.get("/sdcard/Download/red.txt"));
                for (int i=0; i<allLines.size(); i++) {

//                    Log.i("KNNFile", line.split(" ")[0] + line.split(" ")[1] + line.split(" ")[2]);
                    data = new Mat(1,1, CvType.CV_32FC3, new Scalar(
                            Float.valueOf(allLines.get(i).split(" ")[0]),
                            Float.valueOf(allLines.get(i).split(" ")[1]),
                            Float.valueOf(allLines.get(i).split(" ")[2])
                    ));
                    trainData.push_back(data.reshape(1,1));
                    trainLabels.add(0);

                    // limit  samples number
                    if (i>50)
                        break;

                } // next i
            } catch (IOException e) {
                e.printStackTrace();
            }



            // add green color
            try {
                allLines = Files.readAllLines(Paths.get("/sdcard/Download/green.txt"));
                for (int i=0; i<allLines.size(); i++) {

//                    Log.i("KNNFile", line.split(" ")[0] + line.split(" ")[1] + line.split(" ")[2]);
                    data = new Mat(1,1, CvType.CV_32FC3, new Scalar(
                            Float.valueOf(allLines.get(i).split(" ")[0]),
                            Float.valueOf(allLines.get(i).split(" ")[1]),
                            Float.valueOf(allLines.get(i).split(" ")[2])
                    ));
                    trainData.push_back(data.reshape(1,1));
                    trainLabels.add(1);

                    // limit  samples number
                    if (i>50)
                        break;

                } // next i
            } catch (IOException e) {
                e.printStackTrace();
            }

            // add blue color
            try {
                allLines = Files.readAllLines(Paths.get("/sdcard/Download/blue.txt"));
                for (int i=0; i<allLines.size(); i++) {

//                    Log.i("KNNFile", line.split(" ")[0] + line.split(" ")[1] + line.split(" ")[2]);
                    data = new Mat(1,1, CvType.CV_32FC3, new Scalar(
                            Float.valueOf(allLines.get(i).split(" ")[0]),
                            Float.valueOf(allLines.get(i).split(" ")[1]),
                            Float.valueOf(allLines.get(i).split(" ")[2])
                    ));
                    trainData.push_back(data.reshape(1,1));
                    trainLabels.add(2);

                    // limit  samples number
                    if (i>50)
                        break;

                } // next i
            } catch (IOException e) {
                e.printStackTrace();
            }


            // add yellow color
            try {
                allLines = Files.readAllLines(Paths.get("/sdcard/Download/yellow.txt"));
                for (int i=0; i<allLines.size(); i++) {

//                    Log.i("KNNFile", line.split(" ")[0] + line.split(" ")[1] + line.split(" ")[2]);
                    data = new Mat(1,1, CvType.CV_32FC3, new Scalar(
                            Float.valueOf(allLines.get(i).split(" ")[0]),
                            Float.valueOf(allLines.get(i).split(" ")[1]),
                            Float.valueOf(allLines.get(i).split(" ")[2])
                    ));
                    trainData.push_back(data.reshape(1,1));
                    trainLabels.add(3);

                    // limit  samples number
                    if (i>50)
                        break;

                } // next i
            } catch (IOException e) {
                e.printStackTrace();
            }

//            // add purple color
//            try {
//                allLines = Files.readAllLines(Paths.get("/sdcard/Download/purple.txt"));
//                for (int i=0; i<allLines.size(); i++) {
//
//                    data = new Mat(1,1, CvType.CV_32FC3, new Scalar(
//                            Float.valueOf(allLines.get(i).split(" ")[0]),
//                            Float.valueOf(allLines.get(i).split(" ")[1]),
//                            Float.valueOf(allLines.get(i).split(" ")[2])
//                    ));
//                    trainData.push_back(data.reshape(1,1));
//                    trainLabels.add(4);
//
//                    // limit  samples number
//                    if (i>50)
//                        break;
//
//                } // next i
//            } catch (IOException e) {
//                e.printStackTrace();
//            }

            // add White color
            try {
                allLines = Files.readAllLines(Paths.get("/sdcard/Download/purple.txt"));
                for (int i=0; i<allLines.size(); i++) {

//                    Log.i("KNNFile", line.split(" ")[0] + line.split(" ")[1] + line.split(" ")[2]);
                    data = new Mat(1,1, CvType.CV_32FC3, new Scalar(
                            255.0,
                            255.0,
                           255.0
                    ));
                    trainData.push_back(data.reshape(1,1));
                    trainLabels.add(5);

                    // limit  samples number
                    if (i>20)
                        break;

                } // next i
            } catch (IOException e) {
                e.printStackTrace();
            }

            // labels

//            trainLabels.add(5);

            // Train KNN classifier
            colorClassifier.train( trainData,  Ml.ROW_SAMPLE, Converters.vector_int_to_Mat(trainLabels));

            //init test data
            toTest = new Mat();
            tmp.add(0.0f);
            tmp.add(0.0f);
            tmp.add(0.0f);
            toTest = Converters.vector_float_to_Mat(tmp).reshape(1,1);
            Log.i("KNN", "Totest init  " + toTest.size() + " " + toTest.dump() );

            // set trained flag
            alreadyTrained = true;
        }

        // grab image
        frame = inputFrame.rgba();

        // cropped image & resize for speed & performance
        Mat croppedFace = new Mat(frame, imgROI);
        Imgproc.resize(croppedFace, frame, mSize);

        //result of KNN classification
        res = new Mat();

        // HSV frame
        Mat hsv = new Mat();
        // HLS frame
        Mat hls = new Mat();
        /// Convert to HSV
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);
        // idem HLS
        Imgproc.cvtColor(frame, hls, Imgproc.COLOR_BGR2HLS);

//        /// Using 50 bins for hue and 60 for saturation
//        int hBins = 180;
//        int sBins = 60;
////        MatOfInt histSize = new MatOfInt( hBins,  sBins);
//        MatOfInt histSize = new MatOfInt( hBins);
//        // hue varies from 0 to 179, saturation from 0 to 255
//        MatOfFloat ranges =  new MatOfFloat( 0f,180f );
//        // we compute the histogram from the 0-th and 1-st channels
//        MatOfInt channels = new MatOfInt(0, 1);
//        // resulting histogram
//        Mat hist = new Mat();
//        // List of images
//        ArrayList<Mat> histImages=new ArrayList<Mat>();
//        histImages.add(hsv);
//
//        Imgproc.calcHist(histImages,
//                channels,
//                new Mat(),
//                hist,
//                histSize,
//                ranges,
//                false);
//
//        // max in histogram
//
//        // ******** find dominant color
//        //for each color
//        double max =0;
//        int id_max=0;
//        for (int i=0; i<hist.rows(); i++)
//        {
//            // if max found
//            if (hist.get(i, 0)[0]>max)
//            {
//                // save
//                id_max = i;
//                max = hist.get(i, 0)[0];
//            }
//        } // end for each color
//
////        Log.i(TAG, "Hist " +max + " " + id_max);
//
//        if (id_max <= 30 )
//            Log.i(TAG, id_max + " "+ "BLUE");
//        else if( id_max > 30 && id_max <= 85 )
//            Log.i(TAG, id_max + " "+"GREEN");
//        else if( id_max > 85 && id_max <= 100 )
//            Log.i(TAG, id_max + " "+"YELLOW");
//        else if( id_max > 100 && id_max <= 115 )
//            Log.i(TAG, id_max + " "+"ORANGE");
//        else if( id_max > 115 && id_max <= 135 )
//            Log.i(TAG, id_max + " "+"RED");
//        else if( id_max > 135 )
//            Log.i(TAG, id_max + " "+"PURPLE");
//
//
//



        // White recog
        /// Using 50 bins for hue and 60 for saturation
        int hBins = 180;
        int sBins = 60;
//        MatOfInt histSize = new MatOfInt( hBins,  sBins);
        MatOfInt histSize = new MatOfInt( hBins);
        // hue varies from 0 to 179, saturation from 0 to 255
        MatOfFloat ranges =  new MatOfFloat( 0f,180f );
        // we compute the histogram from the 0-th and 1-st channels
        MatOfInt channels = new MatOfInt(1);
        // resulting histogram
        Mat hist = new Mat();
        // List of images
        ArrayList<Mat> histImages=new ArrayList<Mat>();
        histImages.add(hsv);

        Imgproc.calcHist(histImages,
                channels,
                new Mat(),
                hist,
                histSize,
                ranges,
                false);

        // max in histogram

        // ******** find dominant color
        //for each color
        double max =0;
        int id_max=0;
        for (int i=0; i<hist.rows(); i++)
        {
            // if max found
            if (hist.get(i, 0)[0]>max)
            {
                // save
                id_max = i;
                max = hist.get(i, 0)[0];
            }
        } // end for each color

        Log.i(TAG, "Hist " +max + " " + id_max);

        //resize back for vizualization
        Imgproc.resize(frame, frame, new Size(800, 600));

        return frame;
    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }


    public void setPixelColor(Mat frame,int c, int r,  int color_idx)
    {
        double[] data = frame.get(r, c);

//        data[0] = 255;
//                data[1] = 255;
//                data[2] = 0;
//                frame.put(r, c,data );

        switch (color_idx)
        {
            case 0:
                data[0] = 255.0;
                data[1] = 0;
                data[2] = 0;
                frame.put(r, c,data );
                break;
            case 1:
                data[0] = 0;
                data[1] = 255;
                data[2] = 0;
                frame.put(r, c,data );
                break;
            case 2:
                data[0] = 0;
                data[1] = 0;
                data[2] = 255;
                frame.put(r, c,data );
                break;
            case 3:
                data[0] = 250;
                data[1] = 250;
                data[2] = 0;
                frame.put(r, c,data );
                break;
            case 4:
                data[0] = 150;
                data[1] = 0;
                data[2] = 150;
                frame.put(r, c,data );
                break;
            case 5:
                data[0] = 255;
                data[1] = 255;
                data[2] = 255;
                frame.put(r, c,data );
                break;
        } // end switch

    } // end change color


}