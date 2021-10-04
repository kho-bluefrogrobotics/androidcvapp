package com.bfr.opencvapp;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.os.RemoteException;
import android.text.Layout;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.RelativeLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.bfr.buddysdk.sdk.Mood;
import com.bfr.buddysdk.sdk.Services;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
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
import org.opencv.tracking.legacy_TrackerMOSSE;
import org.opencv.tracking.legacy_TrackerMedianFlow;
import org.opencv.video.Tracker;
import org.opencv.video.TrackerMIL;
import org.opencv.videoio.VideoWriter;

import com.bfr.opencvapp.utils.BuddyData;
import com.bfr.usbservice.BodySensorData;
import com.bfr.usbservice.HeadSensorData;
import com.bfr.usbservice.IUsbAidlCbListner;
import com.bfr.usbservice.IUsbCommadRsp;
import com.bfr.usbservice.MotorHeadData;
import com.bfr.usbservice.MotorMotionData;


import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import com.bfr.buddysdk.sdk.BuddySDK;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getName();

    private CameraBridgeViewBase mOpenCvCameraView;

    // directory where the model files are saved for face detection
    private String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();

    // SDK
    BuddySDK mySDK = new BuddySDK();

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

    // Neural net for detection
    private Net net;


    // context
    Context mycontext = this;

    //Video writer
    private VideoWriter videoWriter;
    private boolean isrecording;


    //grafcet
    TrackingGrafcet mTrackingGrafcet = new TrackingGrafcet("VisualTracking");

    // Sensors & motor data
    BuddyData mydata = new BuddyData();

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
        BuddyFace = findViewById(R.id.visage);
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


        //  SDK
        // suscribe to callbacks
        Consumer<Services> onServiceLaunched = (Services iService) -> {
            Log.d("OpenCVAPP", "service launched ");
            try {
                if (iService == Services.SENSORSMOTORS)
//                    mySDK.getUsbInterface().registerCb(mydata);
                    mySDK.getUsbInterface().registerCb(mydata);
                //start the grafcet
                mTrackingGrafcet.start();

            } catch (RemoteException e) {
                e.printStackTrace();
            }
        };

        // init SDK
        mySDK.initSDK(this, onServiceLaunched);

        // Pass SDK and data to grafcet
        mTrackingGrafcet.init(mycontext, mySDK, mydata);



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
                // if checked
                if (noSwitch.isChecked())
                {
                    // enable No
                    try {
                        mySDK.getUsbInterface().enableNoMove(1, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String success) throws RemoteException { }
                            @Override
                            public void onFailed(String error) throws RemoteException {}
                        });
                    } //end try enable Yes
                    catch (RemoteException e)
                    {            e.printStackTrace();
                    } // end catch
                }
                else // unchecked
                {
                    //disable no
                    try {
                        mySDK.getUsbInterface().enableNoMove(0, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String success) throws RemoteException { }
                            @Override
                            public void onFailed(String error) throws RemoteException {}
                        });

                    } //end try enable Yes
                    catch (RemoteException e)
                    {
                        e.printStackTrace();
                    } // end catch
                } // end if toggle

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
                    TrackingGrafcet.step_num =0;
                    TrackingGrafcet.go = false;
                    // close record file
                    isrecording = false;
                    videoWriter.release();

                }
                else // checked
                {
                    isrecording = true;
                    // let the grafcet continue
                    TrackingGrafcet.go = true;
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
        //stop grafcet
        mTrackingGrafcet.start();
    }

    @Override
    public void onResume() {
        super.onResume();
        // OpenCV manager initialization
        OpenCVLoader.initDebug();
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        //restart grafcet
        mTrackingGrafcet.start();
    }


    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onCameraViewStarted(int width, int height) {

        // Load Face detection model
        String proto = dir + "/opencv_face_detector.pbtxt";
        String weights = dir + "/opencv_face_detector_uint8.pb";
        net = Dnn.readNetFromTensorflow(weights, proto);

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




    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // cature frame from camera
        Mat frame = inputFrame.rgba();

        // COUNT FRAME
        frame_count +=1;

        // every xxx frame
        if (frame_count%15 == 0) {

            // convert color to RGB
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

            // Blob
            blob = Dnn.blobFromImage(frame, 1.0,
                    new org.opencv.core.Size(300, 300),
                    new Scalar(104, 117, 123), /*swapRB*/true, /*crop*/false);
            net.setInput(blob);
            // Face detection
            detections = net.forward();
            int cols = frame.cols();
            int rows = frame.rows();
            detections = detections.reshape(1, (int) detections.total() / 7);

            // If found faces
            if (detections.rows() > 0) {
                // only if currently tracking something
                if (istracking) {
                    // find id of closest face
                    int id_closest=0;
                    int max_dist = 999999;
                    int dist;
                    // for each face
                    for (int i = 0; i < detections.rows(); ++i) {
                        double confidence = detections.get(i, 2)[0];
                        if (confidence > THRESHOLD) {
                            int left = (int) (detections.get(i, 3)[0] * cols);
                            int top = (int) (detections.get(i, 4)[0] * rows);
                            int right = (int) (detections.get(i, 5)[0] * cols);
                            int bottom = (int) (detections.get(i, 6)[0] * rows);

                            // if dist min
                            dist = Math.abs(left - TrackingGrafcet.tracked.x) + Math.abs(top - TrackingGrafcet.tracked.y);
                            if (dist < max_dist) {
                                // update
                                max_dist = dist;
                                id_closest = i;
                            }

                        } // end if confidence OK
                    } // next face

                    // Init tracker on closest face
                    int left = (int) (detections.get(id_closest, 3)[0] * cols);
                    int top = (int) (detections.get(id_closest, 4)[0] * rows);
                    int right = (int) (detections.get(id_closest, 5)[0] * cols);
                    int bottom = (int) (detections.get(id_closest, 6)[0] * rows);
                    Rect bbox = new Rect((int) left,
                            top,
                            right-left,
                            bottom-top
                    );
                    Log.i("Tracking", "New Init on " +  bbox.x + " " + bbox.y);

                    try {
                        if (fastTracking)
                            mytracker = TrackerKCF.create();
                        else
                            mytracker = TrackerCSRT.create();

                        mytracker.init(frame, bbox);
                    }
                    catch (Exception e)
                    {

                    }


                    //DRAW detected faces
                    for (int i = 0; i < detections.rows(); ++i) {
                        double confidence = detections.get(i, 2)[0];
                        if (confidence > THRESHOLD) {
                            int classId = (int) detections.get(i, 1)[0];
                            left = (int) (detections.get(i, 3)[0] * cols);
                            top = (int) (detections.get(i, 4)[0] * rows);
                            right = (int) (detections.get(i, 5)[0] * cols);
                            bottom = (int) (detections.get(i, 6)[0] * rows);
                            // Draw rectangle around detected object.
                            Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                                    new Scalar(0, 255, 0), 2);

                        } // end if confidence
                    } // next face
                    // end DRAW

                } // end if is tracking
                else // Not tracking yet
                {
                    // Init tracker on first face
                    int left = (int) (detections.get(0, 3)[0] * cols);
                    int top = (int) (detections.get(0, 4)[0] * rows);
                    int right = (int) (detections.get(0, 5)[0] * cols);
                    int bottom = (int) (detections.get(0, 6)[0] * rows);
                    Rect bbox = new Rect((int) left,
                            top,
                            right-left,
                            bottom-top
                    );
                    Log.i("Tracking", "First Init on " +  bbox.x + " " + bbox.y);
                    // try catch to avoid crash in case of wring detection
                    try
                    {
                        // init tracker
                        mytracker.init(frame, bbox);
                        //set status
                        istracking = true;

                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                }
            } // end if face found

        } // end if every xxx frame
        else
        {
            // if is tracking
            if (istracking)
            {
                Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
                // update the tracker
                Log.i("Tracking", "channels "+ String.valueOf(frame.channels()) );
                //Update tracker
                try { // try catch to avoid crash in case of wrong tracking
                    mytracker.update(frame, TrackingGrafcet.tracked);
                }
                catch(Exception e)
                {                }

                //Log.i("Tracking", "Tracker updated " + tracked.x + " " + tracked.y);
                // draw a rectangle
                Imgproc.rectangle(frame, new Point(TrackingGrafcet.tracked.x, TrackingGrafcet.tracked.y),
                        new Point(TrackingGrafcet.tracked.x+TrackingGrafcet.tracked.width,TrackingGrafcet.tracked.y+TrackingGrafcet.tracked.height ),
                        new Scalar(0, 0, 255), 2);
            } // end if is tracking
        } // end rest of the frames


        // record video
        if (isrecording) {
            Log.i("RecordVideo", frame.channels() + "  " + frame.cols() + "  " + frame.rows());
            videoWriter.write(frame);
        }

        return frame;
    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }



}