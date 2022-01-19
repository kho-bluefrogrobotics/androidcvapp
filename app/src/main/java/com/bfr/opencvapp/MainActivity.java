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

import com.bfr.buddysdk.BuddySDK;

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
import org.opencv.tracking.TrackerCSRT;
import org.opencv.tracking.TrackerKCF;
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
    Size mSize = new Size(640, 480);

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {


        // COUNT FRAME
        frame_count +=1;
        //reset
        if (frame_count>100000)
            frame_count=2;

        // Start after 1st frame
        if (frame_count>1)
        {
            /*** Optical flow***/

            // save previous frame
            prev_frame = curr_frame;
            // cature frame from camera
            curr_frame = inputFrame.rgba();
            // convert to gray
            Imgproc.cvtColor(curr_frame, curr_frame, Imgproc.COLOR_BGR2GRAY);
            // resize for better performances
            Imgproc.resize(curr_frame, curr_frame, mSize);

            //flow
            flow = new Mat(curr_frame.size(), CvType.CV_32FC2);
            //compute optical flow
            Video.calcOpticalFlowFarneback(prev_frame, curr_frame, flow,
                    0.5, 3, 15, 3, 5, 1.2, 0);

            /*** Visualization***/
            // visualization
            ArrayList<Mat> flow_parts = new ArrayList<>(2);
            // resize to display
            Imgproc.resize(flow, flow, new Size(800,600));
            Core.split(flow, flow_parts);
            Mat magnitude = new Mat(), angle = new Mat(), magn_norm = new Mat();
            Core.cartToPolar(flow_parts.get(0), flow_parts.get(1), magnitude, angle,true);
            Core.normalize(magnitude, magn_norm,0.0,1.0, Core.NORM_MINMAX);
            float factor = (float) ((1.0/360.0)*(180.0/255.0));
            Mat new_angle = new Mat();
            Core.multiply(angle, new Scalar(factor), new_angle);
            //build hsv image
            ArrayList<Mat> _hsv = new ArrayList<>() ;
            Mat hsv = new Mat(), hsv8 = new Mat(), bgr = new Mat();
            _hsv.add(new_angle);
            _hsv.add(Mat.ones(angle.size(), CvType.CV_32F));
            _hsv.add(magn_norm);
            Core.merge(_hsv, hsv);
            hsv.convertTo(hsv8, CvType.CV_8U, 255.0);
            Imgproc.cvtColor(hsv8, bgr, Imgproc.COLOR_HSV2BGR);

            Log.i("Franeback", "Max = " + Core.minMaxLoc(magnitude).maxVal);

            if(BuddySDK.isInitialized)
            if(Core.minMaxLoc(magnitude).maxVal> 3.0)
                BuddySDK.Speech.startSpeaking("Je t'ai vu !");

            return bgr;


        }
        else //1st frame
        {
            // cature frame from camera
            curr_frame = inputFrame.rgba();
            // convert to gray
            Imgproc.cvtColor(curr_frame, curr_frame, Imgproc.COLOR_BGR2GRAY);
            // resize for better performances
            Imgproc.resize(curr_frame, curr_frame, mSize);

        }

        return frame;
    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }




}