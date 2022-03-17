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


    // Neural net for detection
    private Net net;
    private boolean isnetloaded = false;

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


    //grafcet


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

        String proto = dir + "/opencv_face_detector.pbtxt";
        String weights = dir + "/opencv_face_detector_uint8.pb";
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


        // start grafcet

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

                }
                else // unchecked
                {

                } // end if togle

            } // end onChecked
        }); // end listener

        //calbacks for Move button
        moveButton.setOnClickListener(v -> {

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


        Mat frame = inputFrame.rgba();


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