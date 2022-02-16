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



import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;



public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getName();

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
    Size mSize = new Size(160, 120);
    //crop for better performances from 800x600
    Rect imgROI = new Rect(new Point(100, 100), new Point(700, 500) );
    // Knn for color recognition
    private KNearest colorClassifier;
    boolean alreadyTrained = false;
    // Mat of training data for 6 colors : black, white, blue, green, red, yellow
    Mat trainData ;
    List<Integer> trainLabels = new ArrayList<Integer>();
    Mat data;
//    Arracolors = np.array([[0, 0, 0],
//            [255, 255, 255],
//            [150, 0, 0],
//            [0, 150, 0],
//            [0, 0, 255],
//            [0, 255, 255]], dtype=np.float32)
//    classes = np.array([[0], [1], [2], [3], [4], [5]], np.float32)

    // Colors to recognize
    Scalar _RED = new Scalar(255,0,0);
    Scalar _BLUE = new Scalar(0,0,255);
    Scalar _GREEN = new Scalar(0,255,0);
    Scalar _YELLOW = new Scalar(0,255,0);
    Scalar _WHITE = new Scalar(255,255,255);
    Scalar _BLACK = new Scalar(0,0,0);



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

            data = new Mat(1,1, CvType.CV_32FC3, _GREEN);
            trainLabels.add(1);
            trainData.push_back(data.reshape(1,1));

            data = new Mat(1,1, CvType.CV_32FC3, _RED);
            trainLabels.add(2);
            trainData.push_back(data.reshape(1,1));

            Log.i("KNN", " " + trainData.size() );
            for (int u=0; u<3; u++)
                Log.i("KNN", " " + trainData.get(0, u)[0] );

            colorClassifier.train( trainData, 0, Converters.vector_int_to_Mat(trainLabels));
            alreadyTrained = true;
        }


        frame = inputFrame.rgba();

        // cropped image & resize for speed & performance
        Mat croppedFace = new Mat(frame, imgROI);
        Imgproc.resize(croppedFace, frame, mSize);

        Mat res = new Mat();
        //for each pixel
        for(int c=0; c<frame.cols(); c++)
        {
            for(int r=0; r<frame.rows(); r++)
            {
                Mat test = frame.submat(r, r, c, c).reshape(1, 1);
                Log.i("KNN", "Result  " + test.dump() );
                test = frame.submat(r, r, c, c).reshape(1, 1);
                float dist= colorClassifier.findNearest(test, 1, res);
                Log.i("KNN", "Result  " + res.dump() );
            }
        }


        //resize back for vizualization
        Imgproc.resize(frame, frame, new Size(800, 600));

        return frame;
    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }




}