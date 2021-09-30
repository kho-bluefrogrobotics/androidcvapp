package com.bfr.opencvapp;

import android.content.Context;
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

import com.bfr.buddysdk.sdk.FacialEvent;
import com.bfr.buddysdk.sdk.Mood;
import com.bfr.buddysdk.sdk.Services;
import com.bfr.opencvapp.cnn.CNNExtractorService;
import com.bfr.opencvapp.cnn.impl.CNNExtractorServiceImpl;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.TrackerKCF;
import org.opencv.video.Tracker;
import org.opencv.videoio.VideoWriter;

import com.bfr.usbservice.BodySensorData;
import com.bfr.usbservice.HeadSensorData;
import com.bfr.usbservice.IUsbAidlCbListner;
import com.bfr.usbservice.IUsbCommadRsp;
import com.bfr.usbservice.MotorHeadData;
import com.bfr.usbservice.MotorMotionData;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import com.bfr.buddysdk.sdk.BuddySDK;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getName();

    private static final String IMAGENET_CLASSES = "imagenet_classes.txt";
    private static final String MODEL_FILE = "pytorch_mobilenet.onnx";

    private CameraBridgeViewBase mOpenCvCameraView;
    private Net opencvNet;

    private CNNExtractorService cnnService;

    // Neural net for detection
    private Net net;
    private boolean isnetloaded = false;

    // SDK
    BuddySDK mySDK = new BuddySDK();

    // Face detector
    private FaceDetector faceDetector;

    //button to start tracking
    private Button initBtn;

    // Face UI
    private RelativeLayout BuddyFace;
    private TextView targetTextView;
    private Switch noSwitch;
    private Button moveButton;
    private CheckBox hideFace;
    TextView noPos;
    private CheckBox trackingCheckBox;

    // Tracking & robot status
    private int noPosition;
    private int trackedFaceX, trackedFaceY;
    private int noTargetX, noTargetY;
    private String noStatus = "";
    private String noMvtStatus = "";
    private float smilingFaceProba;
    private boolean leftEyeOpen, rightEyeOpen, leftEyeOpen_previous, rightEyeOpen_previous;
    int no_speed = 10;

    // gracet
    private int previous_step ;
    private int step_num;
    private long time_in_curr_step = 0;
    private boolean bypass = false;
    // context
    Context mycontext = this;
    Consumer<Mood> onMoodSet = (Mood iMood) -> {
        Log.d("FACE MOOD", "mood set to "+iMood);
        //hasMadeCommand=true;
    };

    Consumer<FacialEvent> onFacialEvent = (FacialEvent iFacialEvent) -> {

        //hasMadeCommand=true;
    };

    //Video writer
    private VideoWriter videoWriter;
    private boolean isrecording;


    //classes
    private static final String[] classNames = {"background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};

    // runable for grafcet
    private Runnable mysequence = new Runnable()
    {
        @Override
        public void run()
        {
//            // Buddy face
//            if (smilingFaceProba > 0.7)
//            {
//                runOnUiThread(new Runnable() {
//                    @Override
//                    public void run() {
//                        mySDK.setMood(mycontext,Mood.HAPPY, onMoodSet);
//                    }
//                });
//
//            }
//            else
//            {
//                runOnUiThread(new Runnable() {
//                    @Override
//                    public void run() {
//                        mySDK.setMood(mycontext,Mood.GRUMPY, onMoodSet);
//                    }
//                });
//            }

            // if step changed
            if( !(step_num == previous_step)) {

                // display current step
                Log.i("GRAFCET", "current step: " + step_num + "  " + noPosition);
                // update
                previous_step = step_num;
                // start counting time in current step
                time_in_curr_step = System.currentTimeMillis();
                bypass = false;
            } // end if step = same
            else
            {
                // if time > 2s
                if ((System.currentTimeMillis()-time_in_curr_step > 5000) && step_num >3)
                {
                    // activate bypass
                    bypass = true;
                }
            }

            // which grafcet step?
            switch (step_num) {
                case 0: // Wait for checkbox
                    //wait until check box
                    if (trackingCheckBox.isChecked()) {
                        // go to next step
                        step_num = 1;
                    }
                    break;

                case 1: // Enable No
                    try {
                        mySDK.getUsbInterface().enableNoMove(1, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String success) throws RemoteException {
                                Log.i("enable No SUCESS:", success);
                            }
                            @Override
                            public void onFailed(String error) throws RemoteException {
                                Log.e("enable No FAILED", error);
                            }
                        });

                    } //end try enable
                    catch (RemoteException e) {
                        e.printStackTrace();
                    } // end catch

                    // go to next step
                    step_num = 2;
                    break;

                case 2: //wait for status No enable
                    // if yes not disabled
                    if (!noStatus.toUpperCase().contains("DISABLE")) {
                        // go to next step
                        step_num = 3;
                    }
                    break;

                case 3: // Move left or right or exit
                    //
                    if (!trackingCheckBox.isChecked())
                    {
                        // go to next step
                        step_num = 10;
                    }
                    // if face to track too much on the left
                    if (trackedFaceX < 350)
                    {
                        // go to next step
                        step_num = 20;
                    }
                    else if (trackedFaceX > 450)
                    {
                        // go to next step
                        step_num = 30;
                    }

//                    // Buddy face
//            if (smilingFaceProba > 0.7)
//            {
//                runOnUiThread(new Runnable() {
//                    @Override
//                    public void run() {
//                        mySDK.setMood(mycontext,Mood.HAPPY, onMoodSet);
//                    }
//                });
//
//            }
//            else
//            {
//                runOnUiThread(new Runnable() {
//                    @Override
//                    public void run() {
//                        mySDK.setMood(mycontext,Mood.GRUMPY, onMoodSet);
//                    }
//                });
//            }

                    break;

                case 10 : // Disable Motor
                    try {
                        mySDK.getUsbInterface().enableNoMove(0, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String success) throws RemoteException {
                                Log.i("enable no SUCESS:", success);
                            }
                            @Override
                            public void onFailed(String error) throws RemoteException {
                                Log.e("enable no FAILED", error);
                            }
                        });
                    } //end try enable Yes
                    catch (RemoteException e)
                    {
                        e.printStackTrace();
                    } // end catch

                    // go to next step
                    step_num = 11;

                    break;

                case 11 : // Wait for status Disable
                    if (noStatus.toUpperCase().contains("DISABLE"))
                    {   // go to next step
                        step_num = 0;
                    }
                    break;


                case 20: // Cmd to Move No to the Left
                    // reset
                    noMvtStatus = "NOT_YET";
                    try {
                        Log.i(TAG, "Sending Yes command");
                        // Move No
                        mySDK.getUsbInterface().buddySayNo(20,0,new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String success) throws RemoteException {     noMvtStatus = success;                }
                            @Override
                            public void onFailed(String error) throws RemoteException {Log.e("yesmove", error);  }
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    // go to next step
                    step_num = 22;
                    break;

                case 21 :   //waiting for acknowledge of cmd
                    if (noMvtStatus.toUpperCase().contains("OK"))
                    {
                        // go to next step
                        step_num = 22;
                    }
                    break;

                case 22 : // waiting for End of Movement
//                    if (noMvtStatus.toUpperCase().contains("NO_MOVE_FINISHED") || bypass)
//                    {
//                        // go to next step
//                        step_num = 3;
//                    }
                    // if face to track too much on the left
                    if (trackedFaceX > 350 )
                    {
                        // go to next step
                        step_num = 23;
                    }
                    break;

                case 23 : // Stop MVT
                    noMvtStatus = "NOT_YET";
                    try {
                        Log.i(TAG, "Sending No command");
                        // Move No
                        mySDK.getUsbInterface().buddySayNo(0,0,new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String success) throws RemoteException {     noMvtStatus = success;                }
                            @Override
                            public void onFailed(String error) throws RemoteException {Log.e("yesmove", error);  }
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    // go to next step
                    step_num = 3;
                    break;


                case 30: // Cmd to Move No to the Right
                    // reset
                    noMvtStatus = "NOT_YET";
                    try {
                        Log.i(TAG, "Sending Yes command");
                        // Move No
                        mySDK.getUsbInterface().buddySayNo(no_speed,noPosition+2,new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String success) throws RemoteException {     noMvtStatus = success;                }
                            @Override
                            public void onFailed(String error) throws RemoteException {Log.e("yesmove", error);  }
                        });


                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    // go to next step
                    step_num = 32;
                    break;


                case 31 :   //waiting for acknowledge of cmd
                    if (noMvtStatus.toUpperCase().contains("OK"))
                    {
                        // go to next step
                        step_num = 32;
                    }
                    break;

                case 32 : // waiting for End of Movement

                    if (noMvtStatus.toUpperCase().contains("NO_MOVE_FINISHED") || bypass)
                    {
                        // go to next step
                        step_num = 3;
                    }
                    break;

                default :
                    // go to next step
                    step_num = 0;
                    break;
            } //End switch

        } // end run
    }; // end new runnable

    //grafcet
    bfr_Grafcet myGrafcet = new bfr_Grafcet(mysequence, "myGrafcet");


    // for Buddy calbacks
    public class BuddyData extends IUsbAidlCbListner.Stub
    {
        @Override
        // called when the datas from the motor are received
        public void ReceiveMotorMotionData(MotorMotionData msg) throws RemoteException {
        }

        @Override
        // called when the datas from the head motor are received
        public void ReceiveMotorHeadData(MotorHeadData msg) throws RemoteException {
//            Log.i("Move_NO", "No position : " + String.valueOf(msg.noPosition));
            noPosition = msg.noPosition;
            noStatus = msg.noMode;
        } // end receiveMotorHead Data

        @Override
        // called when the datas from the head sensor motor are received
        public void ReceiveHeadSensorData(HeadSensorData msg) throws RemoteException {
        }

        @Override
        // called when the datas from the body sensor motor are received
        public void ReceiveBodySensorData(BodySensorData data) throws RemoteException {
        }
    }

    // Instantiate class
    private final BuddyData mydata = new BuddyData();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // initialize implementation of CNNExtractorService
        this.cnnService = new CNNExtractorServiceImpl();
        // configure camera listener
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        // link to UI
        initBtn = findViewById(R.id.initButton);
        BuddyFace = findViewById(R.id.visage);
        targetTextView = findViewById(R.id.noTargetTxtView);
        noSwitch = findViewById(R.id.enableNoSwitch);
        moveButton = findViewById(R.id.MoveNoButton);
        hideFace = findViewById(R.id.visibleCheckBox);
        noPos = findViewById(R.id.noPosTxtView);
        trackingCheckBox = findViewById(R.id.trackingBox);

        // Load model
        // directory where the files are saved
        String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();

        String proto = dir + "/MobileNetSSD_deploy.prototxt";
        String weights = dir + "/MobileNetSSD_deploy.caffemodel";
        Toast.makeText(this, dir , Toast.LENGTH_SHORT).show();
//        net = Dnn.readNetFromCaffe(proto, weights);
//        net = cnnService.getConvertedNet("", TAG);
        Log.i(TAG, "Network loaded successfully");

        //start tracking button
        initBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.i("Tracking", "Tracking is starting");
            }
        });

        // Real-time contour detection of multiple faces
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                        .enableTracking()
                        .build();
        FaceDetector detector = FaceDetection.getClient(options);
        faceDetector = detector;

        //  SDK
        // suscribe to callbacks
        Consumer<Services> onServiceLaunched = (Services iService) -> {
            Log.d("OpenCVAPP", "service launched ");
            try {
                if (iService == Services.SENSORSMOTORS)
//                    mySDK.getUsbInterface().registerCb(mydata);
                    mySDK.getUsbInterface().registerCb(mydata);
                //start the grafcet
                myGrafcet.start();

            } catch (RemoteException e) {
                e.printStackTrace();
            }
        };
        mySDK.initSDK(this, onServiceLaunched);

        // start grafcet
        myGrafcet.start();

        //callback show face
        hideFace.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                //if checked
                if (hideFace.isChecked())
                {   // set tranparent
                    BuddyFace.setAlpha(0.4F);
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
                } // end if togle

            } // end onChecked
        }); // end listener

        //calbacks for Move button
        moveButton.setOnClickListener(v -> {
            try {
                int no_speed = 50;
                Log.i("Move_NO", "Moving No to " + targetTextView.getText() + " at " + String.valueOf(no_speed) + "deg/s");
                // Make No move
                mySDK.getUsbInterface().buddySayNo(no_speed, Integer.parseInt(String.valueOf(targetTextView.getText())),new IUsbCommadRsp.Stub() {
                    @Override
                    public void onSuccess(String success) throws RemoteException {      Log.i("Move_NO: ", success);              }
                    @Override
                    public void onFailed(String error) throws RemoteException {Log.e("Move_NO: ", error);  }
                });
            } catch (RemoteException e)
            {                e.printStackTrace();
            } // end try catch
        }); // end listener

        //tracking
        trackingCheckBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                //reset
                //reset
                if (!trackingCheckBox.isChecked())
                {
                    // reset grafcet
                    step_num =0;
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


    }

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


    // frame captured by camera
    Mat frame;
    // conversion to MLKit
    InputImage inputImage;
    Bitmap bitmapImage = null ;
    // Face detector
    Task result;
    // List of detected faces
    Vector<DetectedFace> foundFaces = new Vector<DetectedFace>();

    // List of Aruco Markers
    List<Mat> arucoCorners;
    Mat arucoIds ;


    public void onCameraViewStarted(int width, int height) {
        // obtaining converted network
        String onnxModelPath = getPath(MODEL_FILE, this);
        if (onnxModelPath.trim().isEmpty()) {
            Log.i(TAG, "Failed to get model file");
            return;
        }
//        opencvNet = cnnService.getConvertedNet(onnxModelPath, TAG);

        // Tracker init
//        mytracker = TrackerKCF.create();

        // write video file
        videoWriter = new VideoWriter("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
                25.0D, new Size(800, 600));
        videoWriter.open("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
                25.0D,  new Size( 800,600));

        arucoIds = new Mat();
        arucoCorners = new ArrayList<>();

    }




    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {


        Mat frame = inputFrame.rgba();

        //convert
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        // Definition of dictionary and params
        Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_APRILTAG_36h11);
        DetectorParameters arucoParams = DetectorParameters.create();

        // Detect Marker

        Aruco.detectMarkers(frame, arucoDict, arucoCorners, arucoIds, arucoParams);


        if (arucoCorners.size()>0)
        {
            Aruco.drawDetectedMarkers(frame, arucoCorners);
            // print list of codes
            Log.i("aruco", String.valueOf(arucoIds.get(0,0)[0]));
        }

        // draw a rectangle
        Imgproc.rectangle(frame, new Point(10, 10), new Point(60,60), new Scalar(255, 10, 10));


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

    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        try {
            // read the defined data from assets
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    } // end getpath




}