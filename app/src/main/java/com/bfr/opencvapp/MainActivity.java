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
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.TrackerKCF;
import org.opencv.video.Tracker;

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
                if (!trackingCheckBox.isChecked())
                    step_num =0;
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

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
//                        Log.i("MoodFace", smilingFaceProba + " " + ((smilingFaceProba - 0.5F)/0.5F) + " " + (1.0F-smilingFaceProba+0.5F));
                mySDK.setFacePositivity(0.3F);
                mySDK.setFaceEnergy(1.0F-smilingFaceProba);
                // if change in eyes open
                if (leftEyeOpen!=leftEyeOpen_previous)
                {
                    // if close eye
                    if (!leftEyeOpen)
                    {
                        mySDK.playEvent(mycontext, FacialEvent.CLOSE_RIGHT_EYE, onFacialEvent) ;
                    }
                    else
                    {
                        mySDK.playEvent(mycontext, FacialEvent.OPEN_RIGHT_EYE, onFacialEvent) ;
                    }// end if cloed eye
                    //update status
                    leftEyeOpen_previous = leftEyeOpen;
                }//end if change in eyes open

                // if change in eyes open
                if (rightEyeOpen!=rightEyeOpen_previous)
                {
                    // if close eye
                    if (!rightEyeOpen)
                    {
                        mySDK.playEvent(mycontext, FacialEvent.CLOSE_LEFT_EYE, onFacialEvent) ;
                    }
                    else
                    {
                        mySDK.playEvent(mycontext, FacialEvent.OPEN_LEFT_EYE, onFacialEvent) ;
                    }// end if cloed eye
                    //update status
                    rightEyeOpen_previous = rightEyeOpen;
                }//end if change in eyes open
            }
        });

        // if not loaded
        if (!isnetloaded)
        {
            // Load model
            // directory where the files are saved
            String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();

            String proto = dir + "/MobileNetSSD_deploy.prototxt";
            String weights = dir + "/MobileNetSSD_deploy.caffemodel";
//            runOnUiThread(new Runnable() {
//                @Override
//                public void run() {
//                    Toast.makeText(this, "coucou" , Toast.LENGTH_SHORT).show();
//                }
//            });

//            net = Dnn.readNetFromCaffe(proto, weights);
            //net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
//            net.setPreferableTarget(Dnn.DNN_TARGET_OPENCL_FP16);

            isnetloaded = true;
        }


        // Caapture Frame from camera
        frame = inputFrame.rgba();
        Log.i("MLKit", String.valueOf(System.currentTimeMillis()) + " Taking picture: ");
        //Detecting Face

        //convert to bitmap
        bitmapImage = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame, bitmapImage);
        //convert to InputImage
        inputImage = InputImage.fromBitmap(bitmapImage, 0);
        result = faceDetector.process(inputImage)
                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
//                        Log.i("MLKit", String.valueOf(System.currentTimeMillis())+ " found faces : " + String.valueOf(faces.size()) +" while " +  String.valueOf(foundFaces.size()));
                        // adjusting size
                        if (foundFaces.size()<faces.size())
                        {   // add elements
                            do {
                                foundFaces.add(new DetectedFace());
                            } while (foundFaces.size()<faces.size());
                        } // end if foundface size < i+1
                        else if (foundFaces.size()>faces.size())
                        {   // remove elements
                            do {
                                foundFaces.remove(0);
                            } while (foundFaces.size()>faces.size());
                        } // end if foundface size < i+1

//                        Log.i("MLKit", String.valueOf(System.currentTimeMillis())+  " NOW faces : " + String.valueOf(faces.size()) +" while " +  String.valueOf(foundFaces.size()));

                        // for each detected face
                        for (int i=0; i < faces.size(); i++) {
                            // assiging values
                            foundFaces.get(i).x1 = faces.get(i).getBoundingBox().left;
                            foundFaces.get(i).y1 = faces.get(i).getBoundingBox().top;
                            foundFaces.get(i).x2 = faces.get(i).getBoundingBox().right;
                            foundFaces.get(i).y2 = faces.get(i).getBoundingBox().bottom;
                            // tracking id
                            foundFaces.get(i).trackingId = faces.get(i).getTrackingId();
                            //classifications
                            foundFaces.get(i).smilingProbability = faces.get(i).getSmilingProbability();
                            foundFaces.get(i).leftEyeOpenProbability = faces.get(i).getLeftEyeOpenProbability();
                            foundFaces.get(i).rightEyeOpenProbability = faces.get(i).getRightEyeOpenProbability();
//                        Log.i("MLKit", String.valueOf(System.currentTimeMillis()) + " Face detected : "
//                                + String.valueOf(foundFaces.get(i).trackingId) + "  "
//                                + String.valueOf(foundFaces.get(i).x1) +
//                                    " " + String.valueOf(foundFaces.get(i).y1)
//                                + " " + String.valueOf(foundFaces.get(i).x2)
//                                + " " + String.valueOf(foundFaces.get(i).y2) );

                        } // next face
//                        Log.i("MLKit", String.valueOf(System.currentTimeMillis())+  " Finally faces : " + String.valueOf(faces.size()) +" while " +  String.valueOf(foundFaces.size()));

                    } // end onSucess
                }); // end process image

        try {
            Tasks.await(result);
        } catch (ExecutionException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // draw a rectangle around face
//        Log.i("MLKit", String.valueOf(System.currentTimeMillis())+ " Found faces : " + String.valueOf(foundFaces.size()) );
        //
        try
        {   // for each found face
            for (int i=0; i<foundFaces.size(); i++)
            {
//                Log.i("MLKit", String.valueOf(System.currentTimeMillis())+ " Attemtpting Drawing index=" + String.valueOf(i));
//                Log.i("MLKit", String.valueOf(System.currentTimeMillis()) + "Drawing face : "
//                        + String.valueOf(foundFaces.get(i).x1) +
//                        " " + String.valueOf(foundFaces.get(i).y1)
//                        + " " + String.valueOf(foundFaces.get(i).x2)
//                        + " " + String.valueOf(foundFaces.get(i).y2) );

            } // next face

            // assign value to track : first face to track
            if(foundFaces.size()>0)
            {
                Imgproc.rectangle(frame, new Point(foundFaces.get(0).x1, foundFaces.get(0).y1),
                        new Point(foundFaces.get(0).x2,foundFaces.get(0).y2),
                        new Scalar(0, 0, 255), 2);
                Imgproc.putText(frame,   "Target",
                        new Point(foundFaces.get(0).x1, foundFaces.get(0).y1-5),2, 0.8, new Scalar(255,0 , 0));

                trackedFaceX = (int) (foundFaces.get(0).x1 +  Math.ceil((foundFaces.get(0).x2 - foundFaces.get(0).x1) /2 ));
                trackedFaceY = (int) (foundFaces.get(0).y1 +  Math.ceil((foundFaces.get(0).y2 - foundFaces.get(0).y1) /2 ));
                smilingFaceProba = foundFaces.get(0).smilingProbability;
                Log.i("Eyeproba", foundFaces.get(0).leftEyeOpenProbability + "  " + foundFaces.get(0).rightEyeOpenProbability);
                if(foundFaces.get(0).leftEyeOpenProbability>0.5)
                {
                    leftEyeOpen=true;
                }
                else
                {
                    leftEyeOpen=false;
                }
                if(foundFaces.get(0).rightEyeOpenProbability>0.5)
                {
                    rightEyeOpen=true;
                }
                else
                {
                    rightEyeOpen=false;
                }

            }
        } // end try
        catch (Exception e)
        {            e.printStackTrace();
        }



//        Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2,y2), new Scalar(255, 10, 10));
//        Imgproc.putText(frame, String.valueOf(trackId) + "  " + String.valueOf(smilingProba), new Point(x1, y1),2, 0.8, new Scalar(255,0 , 0));


//        if (false)
//        {
//            // draw a rectangle
//            Imgproc.rectangle(frame, new Point(600, 100), new Point(700,200), new Scalar(255, 10, 10));
//
//            final int IN_WIDTH = 300;
//            final int IN_HEIGHT = 300;
//            final float WH_RATIO = (float)IN_WIDTH / IN_HEIGHT;
//            final double IN_SCALE_FACTOR = 0.007843;
//            final double MEAN_VAL = 127.5;
//            final double THRESHOLD = 0.75;
//
//
//            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
//
//            // atempt to find a person
//            // Forward image through network.
//            Mat blob = Dnn.blobFromImage(frame, IN_SCALE_FACTOR,
//                    new org.opencv.core.Size(IN_WIDTH, IN_HEIGHT),
//                    new Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), /*swapRB*/false, /*crop*/false);
//            net.setInput(blob);
//            Mat detections = net.forward();
//            int cols = frame.cols();
//            int rows = frame.rows();
//            detections = detections.reshape(1, (int)detections.total() / 7);
//            for (int i = 0; i < detections.rows(); ++i) {
//                double confidence = detections.get(i, 2)[0];
//                if (confidence > THRESHOLD) {
//                    int classId = (int)detections.get(i, 1)[0];
//                    int left   = (int)(detections.get(i, 3)[0] * cols);
//                    int top    = (int)(detections.get(i, 4)[0] * rows);
//                    int right  = (int)(detections.get(i, 5)[0] * cols);
//                    int bottom = (int)(detections.get(i, 6)[0] * rows);
//                    // Draw rectangle around detected object.
//                    Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
//                            new Scalar(0, 255, 0));
//                    String label = classNames[classId] + ": " + confidence;
//                    int[] baseLine = new int[1];
//                    org.opencv.core.Size labelSize = Imgproc.getTextSize(label, 2, 0.5, 1, baseLine);
//                    // Draw background for label.
//                    //Imgproc.rectangle(frame, new Point(left, top - labelSize.getHeight()),
//                    //        new Point(left + labelSize.getWidth(), top + baseLine[0]),
//                    //        new Scalar(255, 255, 255), Imgproc.FILLED);
//                    Imgproc.rectangle(frame, new Point(left, top - labelSize.height),
//                            new Point(left + labelSize.width, top + baseLine[0]),
//                            new Scalar(255, 255, 255));
//                    // Write class name and confidence.
////                Imgproc.putText(frame, label, new Point(left, top),
//                    Imgproc.putText(frame, String.valueOf(classId), new Point(left, top),
//                            2, 0.8, new Scalar(255,0 , 0));
//                }   // end if confidence OK
//            } // next detection
//
//
//            // if found person
//            if (foundperson)
//            {
//                // if not tracking yet
//                if (!istracking)
//                {
//                    // init
//                    // init tracker on drawn rect
//                    Rect initBB = new Rect(600, 100, 100, 100);
//                    mytracker.init(frame, initBB);
//
//                    foundperson = false;
//                    istracking = true;
//                }
//                else // is tracking
//                {
//
//                }
//
//            }
//
//        } // end if modulo 10
//        else // rest of the frames
//        {
//            // if is tracking
//            if (istracking)
//            {
//                //Update tracker
//                mytracker.update(frame, tracked);
//
//                // draw a rectangle
//                Imgproc.rectangle(frame, new Point(tracked.x, tracked.y), new Point(tracked.x+tracked.width,tracked.y+tracked.height), new Scalar(0, 0, 255));
//            }
//        }


        return frame;
    } // end function

    public void onCameraViewStopped() {
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