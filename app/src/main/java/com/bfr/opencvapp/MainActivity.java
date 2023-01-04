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
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.opencv.video.Tracker;

import org.opencv.video.TrackerNano;
import org.opencv.video.TrackerNano_Params;
import org.opencv.videoio.VideoWriter;

import com.bfr.opencvapp.utils.BuddyData;
import com.bfr.usbservice.IUsbCommadRsp;

import java.util.Collections;
import java.util.Date;
import java.util.List;

import java.util.function.Consumer;

import com.bfr.buddysdk.sdk.BuddySDK;

import com.bfr.opencvapp.grafcet.*;

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
    final double THRESHOLD = 0.9;
    // Tracker
    TrackerNano mytracker;
//    TrackerNano_Params mytrackerparams = new TrackerNano_Params();
    TrackerNano_Params mytrackerparams ;

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

    //grafcet
    TrackingGrafcet mTrackingGrafcet = new TrackingGrafcet("VisualTracking");
    TrackingYesGrafcet mTrackingYesGrafcet = new TrackingYesGrafcet("VisualTracking");

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
            Log.d("buddywelcomehost", "service launched ");
            try {
                if (iService == Services.SENSORSMOTORS)
//                    mySDK.getUsbInterface().registerCb(mydata);
                    mySDK.getUsbInterface().registerCb(mydata);
                //start the grafcet
                mTrackingGrafcet.start();
                mTrackingYesGrafcet.start();

            } catch (RemoteException e) {
                e.printStackTrace();
            }
        };

        // init SDK
        mySDK.initSDK(this, onServiceLaunched);

        // Pass SDK and data to grafcet
        mTrackingGrafcet.init(mycontext, mySDK, mydata);
        mTrackingYesGrafcet.init(mycontext, mySDK, mydata);



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

                    TrackingYesGrafcet.step_num =0;
                    TrackingYesGrafcet.go = false;
                    // close record file
                    isrecording = false;
                    videoWriter.release();

                }
                else // checked
                {
                    isrecording = true;
                    // let the grafcet continue
                    TrackingGrafcet.go = true;
                    TrackingYesGrafcet.go = true;

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
        mTrackingGrafcet.stop();
        mTrackingYesGrafcet.stop();
    }

    @Override
    public void onResume() {
        super.onResume();
        // OpenCV manager initialization
        OpenCVLoader.initDebug();
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        //restart grafcet
        mTrackingGrafcet.start();
        mTrackingYesGrafcet.start();
    }


    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onCameraViewStarted(int width, int height) {

        // Load Face detection model
//        String proto = dir + "/opencv_face_detector.pbtxt";
//        String weights = dir + "/opencv_face_detector_uint8.pb";
//        net = Dnn.readNetFromTensorflow(weights, proto);

        String model_person  = dir + "/MobileNetSSD_deploy.prototxt";
        String weights_person  = dir + "/MobileNetSSD_deploy.caffemodel";

        //net = Dnn.readNetFromCaffe(proto, weights);
        net = Dnn.readNetFromCaffe(model_person, weights_person);

        // Tracker init
//        if (fastTracking)
//            mytracker = TrackerKCF.create();
//        else
//            mytracker = TrackerCSRT.create();

        mytrackerparams = new TrackerNano_Params();
        mytrackerparams.set_backbone(dir + "/nanotrack_backbone_sim.onnx");
        mytrackerparams.set_neckhead(dir + "/nanotrack_head_sim.onnx");
        mytracker = TrackerNano.create(mytrackerparams);



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
        if (frame_count%90 == 0 || frame_count == 1) {

            // convert color to RGB
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

            // Blob
//            blob = Dnn.blobFromImage(frame, 1.0,
//                    new org.opencv.core.Size(300, 300),
//                    new Scalar(104, 117, 123), /*swapRB*/true, /*crop*/false);
            blob = Dnn.blobFromImage(frame, 0.007843,
                    new org.opencv.core.Size(300, 300),
                    new Scalar(127.5, 127.5, 127.5), /*swapRB*/true, /*crop*/false);
            net.setInput(blob);
            // Face detection
            detections = net.forward();
            int cols = frame.cols();
            int rows = frame.rows();
            detections = detections.reshape(1, (int) detections.total() / 7);

            int faceOverThresh = 0; // number of face detected with enough confidence

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

                            // increment num of faces
                            faceOverThresh +=1;

                            // if dist min
                            dist = Math.abs(left - TrackingGrafcet.tracked.x) + Math.abs(top - TrackingGrafcet.tracked.y);
                            if (dist < max_dist) {
                                // update
                                max_dist = dist;
                                id_closest = i;
                            }

                        } // end if confidence OK
                    } // next face

                    // save head position
                    mTrackingGrafcet.lastValidPos = mydata.noPos;
                    mTrackingYesGrafcet.lastValidPos = mydata.noPos;

                    // Init tracker on closest face
                    int left = (int) (detections.get(id_closest, 3)[0] * cols);
                    int top = (int) (detections.get(id_closest, 4)[0] * rows);
                    int right = (int) (detections.get(id_closest, 5)[0] * cols);
                    int bottom = (int) (detections.get(id_closest, 6)[0] * rows);
                    Rect bbox = new Rect((int) left,
                            top,
                            right-left,
                            (bottom-top)/2
                    );
                    Log.i("Tracking", "New Init on " +  bbox.x + " " + bbox.y);

                    try {
//                        if (fastTracking)
//                            mytracker = TrackerKCF.create();
//                        else
//                            mytracker = TrackerCSRT.create();

//                        mytracker = TrackerNano.create();
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
                                    new Scalar(250, 0, 0), 5);

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
                            (bottom-top)/2
                    );
                    Log.i("Tracking", "First Init on " +  bbox.x + " " + bbox.y);
                    // try catch to avoid crash in case of wring detection
                    try
                    {
//                        mytracker = TrackerNano.create();
                        // init tracker
                        mytracker.init(frame, bbox);
                        //set status
                        istracking = true;

                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                }
            } // end if face found

            if (faceOverThresh ==0) //  if No face found
            {   Log.i("current step", "NO FACE FOUND " + detections.rows());
                // go to last know position where a face was found
                mTrackingGrafcet.lostFaces = true;
                mTrackingYesGrafcet.lostFaces = true;
            }
            else
            {
//                Log.i("current step", "OK FACE FOUND " + detections.rows());
                // go to last know position where a face was found
                mTrackingGrafcet.lostFaces = false;
                mTrackingYesGrafcet.lostFaces = false;
            }


        } // end if every xxx frame
        else
        {
            // if is tracking
            if (istracking)
            {
                Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
                // update the tracker
//                Log.i("Tracking", "channels "+ String.valueOf(frame.channels()) );
                //Update tracker
                try { // try catch to avoid crash in case of wrong tracking
                    mytracker.update(frame, TrackingGrafcet.tracked);

                    Log.i(TAG, "TRACKING SCORE "+mytracker.getTrackingScore());
                }
                catch(Exception e)
                {                }

                //Log.i("Tracking", "Tracker updated " + tracked.x + " " + tracked.y);
                // draw a rectangle
                Imgproc.rectangle(frame, new Point(TrackingGrafcet.tracked.x, TrackingGrafcet.tracked.y),
                        new Point(TrackingGrafcet.tracked.x+TrackingGrafcet.tracked.width,TrackingGrafcet.tracked.y+TrackingGrafcet.tracked.height ),
                        new Scalar(0, 250, 0), 5);
            } // end if is tracking
        } // end rest of the frames

        //Rectangle sur le milieu de la frame pour vérifier que la personne est au milieu
/*        Imgproc.rectangle(frame, new Point(frame.width()/3, 0),
                new Point((frame.width()/3)*2,frame.height()),
                new Scalar(255, 0, 0), 2);

        Imgproc.rectangle(frame, new Point(0, 200),
                new Point(frame.width(),400),
                new Scalar(255, 0, 0), 2);*/



        // record video
        if (isrecording) {
            Log.i("RecordVideo", frame.channels() + "  " + frame.cols() + "  " + frame.rows());
            videoWriter.write(frame);
        }

//        Imgcodecs.imwrite("/sdcard/myimage.jpg", frame);

        return frame;
    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }



}