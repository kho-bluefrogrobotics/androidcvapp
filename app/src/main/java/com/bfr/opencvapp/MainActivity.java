package com.bfr.opencvapp;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Environment;
import android.os.RemoteException;
import android.os.Trace;
import android.text.Layout;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.RelativeLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.TrackerCSRT;
import org.opencv.tracking.TrackerKCF;
import org.opencv.tracking.legacy_TrackerMOSSE;
import org.opencv.tracking.legacy_TrackerMedianFlow;
import org.opencv.video.Tracker;
import org.opencv.video.TrackerMIL;
import org.opencv.videoio.VideoWriter;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;

import com.bfr.opencvapp.utils.BuddyData;


import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.CameraInput;
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView;
import com.google.mediapipe.solutioncore.VideoInput;
import com.google.mediapipe.solutions.facemesh.FaceMesh;
import com.google.mediapipe.solutions.facemesh.FaceMeshOptions;
import com.google.mediapipe.solutions.facemesh.FaceMeshResult;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getName();

    private CameraBridgeViewBase mOpenCvCameraView;

    //Mediapipe
    private FaceMesh facemesh;
    // Video demo UI and video loader components.
    private VideoInput videoInput;
    // Live camera demo UI and camera components.
    private CameraInput cameraInput;

    //button to start tracking
    private Button initBtn;

    // Face UI
    private RelativeLayout BuddyFace;
    private Switch noSwitch;
    private CheckBox hideFace;

    private CheckBox trackingCheckBox;
    private CheckBox recordingChckBox;

    EditText faceNameTxt;


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
    boolean oncreated = false;

    // Neural net for detection
    private Net net;


    // context
    Context mycontext = this;

    //Video writer
    private VideoWriter videoWriter;
    private boolean isrecording;

    SimpleDateFormat sdf = new SimpleDateFormat("yyMMddHHmmss");
    String currentDateandTime = "";


    // List of known faces
    private List<trainFace> knownFaces = new ArrayList<>();;
    // directory where the comportemental are saved
    String dir ;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Copy the assets in the Android/data folder
        copyAssets();

        oncreated = true;

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
        recordingChckBox = findViewById(R.id.Recording);
        faceNameTxt = findViewById(R.id.editPersonName);

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



        //tracking
        recordingChckBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                //Enable fast tracking
                if (b)
                {
                    isrecording = true;
                }
                else // checked
                {
                    isrecording = false;
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

        while(!oncreated)
        {
            //wait
        }

        // Load model
        // directory where the models are
        dir = getExternalFilesDir(null).toString()+"/nn_models/";

        // Load Face detection model
        String proto = dir + "/opencv_face_detector.pbtxt";
        String weights = dir + "/opencv_face_detector_uint8.pb";
        net = Dnn.readNetFromTensorflow(weights, proto);

        // Load Facenet model for face recognition
        File tfliteModel = new File(dir+"/mobile_face_net.tflite");
        if(tfliteModel.exists()){
            Log.i("coucou", "FOUND TFLITE MODEL");
        }

        tfLite = new Interpreter(tfliteModel );

        Log.i("coucou", "face recog model loaded");

        // Init write video file
        videoWriter = new VideoWriter("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
                25.0D, new Size(800, 600));
        videoWriter.open("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
                25.0D,  new Size( 800,600));

        // process all known faces
        computeKnownFaces(knownFaces);

    }



    //
    Mat frameToSave ;
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // cature frame from camera
        Mat frame = inputFrame.rgba();

        // COUNT FRAME
        frame_count +=1;

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

        // save copy to save
        frameToSave = frame.clone();

        // If found faces
        if (detections.rows() > 0) {
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

            /*** Eyes detection */
//            // Init
//            CascadeClassifier eyes_cascade =new CascadeClassifier();
//            eyes_cascade.load( getExternalFilesDir(null).toString()+"/nn_models/"+"haarcascade_eye_tree_eyeglasses.xml");
//            MatOfRect eyes = new MatOfRect();
//            // Eye detection
//            eyes_cascade.detectMultiScale(frame, eyes );
//            if (!eyes.empty())
//            {
//                Log.i("Eyes",eyes.rows() + " "+eyes.get(0, 0)[0]
//                        + " " + eyes.get(0, 0)[1]
//                        + " " + eyes.get(0, 0)[2]
//                        + " " + eyes.get(0, 0)[3]
//                );
//            }
//
//            List<Rect> listOfEyes = eyes.toList();
//            for (Rect eye : listOfEyes) {
//                Point eyeCenter = new Point(0 + eye.x + eye.width / 2, 0 + eye.y + eye.height / 2);
//                int radius = (int) Math.round((eye.width + eye.height) * 0.25);
//                Imgproc.circle(frame, eyeCenter, radius, new Scalar(255, 0, 0), 4);
//            }

            int candidate = 0;
            String recog = "";
            try {
                // Image of only face
                Rect faceROI = new Rect(new Point(left, top), new Point(right, bottom));
                Mat croppedFace = new Mat(frameToSave, faceROI);
                Mat croppedFaceResize = new Mat();
                Imgproc.resize(croppedFace, croppedFaceResize, new Size(INPUTSIZE, INPUTSIZE));

                //if recording
                if (isrecording) {
                    //filename
                    currentDateandTime = sdf.format(new Date());
                    //record
                    Imgcodecs.imwrite( getExternalFilesDir(null).toString()+"/faces/"+ faceNameTxt.getText().toString() +"_" + currentDateandTime + ".jpg",
                            croppedFaceResize);
                    isrecording = false;
                    //Confirm
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(getApplicationContext(), "*** SAVED Face: " + faceNameTxt.getText().toString() ,
                                    Toast.LENGTH_LONG).show();
                        }
                    });
                    Thread.sleep(2000);
                    // update known faces
                    // process all known faces
                    computeKnownFaces(knownFaces);

                    return frame;
                }

                //convert to bitmap
                Bitmap faceBitmap = Bitmap.createBitmap(croppedFaceResize.cols(), croppedFaceResize.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(croppedFaceResize, faceBitmap);

                // compute embeddings
                float[][] faceEmbeddings = computeEmbeddings(faceBitmap);

                // todebug
                Log.i("coucou", "Embeedings Before " + String.valueOf(faceEmbeddings[0][0]) + "  "
                        + String.valueOf(faceEmbeddings[0][1]) + "  "
                        + String.valueOf(faceEmbeddings[0][2]) + "  "
                        + String.valueOf(faceEmbeddings[0][3]) + "  ");

                //*******************************************************************************
                //********************************  RECOGNITION *********************************

//        final RectF boundingBox = new RectF(face.getBoundingBox());
//            final RectF boundingBox = new RectF();

                // Recognize face
                candidate = findNearest(faceEmbeddings[0], knownFaces);
                recog = knownFaces.get(candidate).name
                .replace("0", "")
                        .replace("1", "")
                        .replace("2", "")
                        .replace("3", "")
                        .replace("4", "")
                        .replace("5", "")
                        .replace("6", "");

//                Log.i("coucou", "Recognized face: " + knownFaces.get(candidate).name.toUpperCase());

            } catch (Exception e) {
                e.printStackTrace();
            }

            String finalRecog = recog;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(getApplicationContext(), "Recognized face: " + finalRecog.replace(".jpg", "").split("_")[0].toUpperCase(), Toast.LENGTH_SHORT).show();
                }
            });

        } // end if face found





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


    // preallocated buffer for face image
    private int[] intValues;
    private ByteBuffer imgData;

    // Facenet model
    private Interpreter tfLite;
    // params for Facenet Model
    private int INPUTSIZE = 112;
    private boolean isModelQuantized = true;
    private float IMAGE_MEAN = 127F;
    private float IMAGE_STD = 127F;
    private static final int OUTPUT_SIZE = 192;
    // Resulting face embeedings (signature) from Facenet
    private float[][] embeedings;
    // Known (previously saved)  faces
    private HashMap<String, SimilarityClassifier.Recognition> registered = new HashMap<>();


    public float[][] computeEmbeddings(Bitmap faceBitmap)
    {
        // init
        intValues = new int[INPUTSIZE * INPUTSIZE];

        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        int facewidth = faceBitmap.getWidth();
        int faceHeight = faceBitmap.getHeight();
        faceBitmap.getPixels(intValues, 0, facewidth, 0, 0, facewidth,faceHeight );
        // fill in the data vales for the FaceNet model
        int numBytesPerChannel;
        isModelQuantized = false;
        if (isModelQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        imgData = ByteBuffer.allocateDirect(1 * INPUTSIZE * INPUTSIZE * 3 * numBytesPerChannel);
        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();
        //for each row
        for (int i = 0; i < INPUTSIZE; ++i) {
            // for each col
            for (int j = 0; j < INPUTSIZE; ++j) {
                int pixelValue = intValues[i * INPUTSIZE + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                } // end if quantized model
            } // next col
        } // next row

        // Here outputMap is changed to fit the Face Mask detector
        Map<Integer, Object> outputMap = new HashMap<>();

        // Init Face embeedings (signature)
        embeedings = new float[1][OUTPUT_SIZE];
        // Assign to Facenet output
        outputMap.put(0, embeedings);

        // input of Facenet model
        Object[] inputArray = {imgData};

        // todebug
//        Log.i("coucou", "Embeedings Before " + String.valueOf(embeedings[0][0]) + "  "
//                + String.valueOf(embeedings[0][1]) + "  "
//                + String.valueOf(embeedings[0][2]) + "  "
//                +String.valueOf(embeedings[0][3]) + "  ");

        // Run the inference in FaceNet Model
        Trace.beginSection("run");
        //tfLite.runForMultipleInputsOutputs(inputArray, outputMapBack);
        // Face net model


        if(tfLite ==null)
            Log.i("coucou", "tflite null");
        if(inputArray ==null)
            Log.i("coucou", "input null");
        if(outputMap ==null)
            Log.i("coucou", "output null");
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        Trace.endSection();

        // todebug
//        Log.i("coucou", "Embeedings After " + String.valueOf(embeedings[0][0]) + "  "
//                + String.valueOf(embeedings[0][1]) + "  "
//                + String.valueOf(embeedings[0][2]) + "  "
//                +String.valueOf(embeedings[0][3]) + "  ");

        return embeedings;

    } //end compute Embeddings












    // recognize Face
    public List<SimilarityClassifier.Recognition> recognizeImage(final Bitmap bitmap, boolean storeExtra) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // fill in the data vales for the FaceNet model
        imgData.rewind();
        for (int i = 0; i < INPUTSIZE; ++i) {
            for (int j = 0; j < INPUTSIZE; ++j) {
                int pixelValue = intValues[i * INPUTSIZE + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }

        // for Trace
        Trace.endSection(); // preprocessBitmap
        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");

        // input of Facenet model
        Object[] inputArray = {imgData};

        // for Trace
        Trace.endSection();

        // Here outputMap is changed to fit the Face Mask detector
        Map<Integer, Object> outputMap = new HashMap<>();

        // Init Face embeedings (signature)
        embeedings = new float[1][OUTPUT_SIZE];
        // Assign to Facenet output
        outputMap.put(0, embeedings);

        // todebug
//        Log.i("coucou", "Embeedings Before " + String.valueOf(embeedings[0][0]) + "  "
//                + String.valueOf(embeedings[0][1]) + "  "
//                + String.valueOf(embeedings[0][2]) + "  "
//                +String.valueOf(embeedings[0][3]) + "  ");

        // Run the inference in FaceNet Model
        Trace.beginSection("run");
        //tfLite.runForMultipleInputsOutputs(inputArray, outputMapBack);
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        Trace.endSection();

        // todebug
//        Log.i("coucou", "Embeedings After " + String.valueOf(embeedings[0][0]) + "  "
//                + String.valueOf(embeedings[0][1]) + "  "
//                + String.valueOf(embeedings[0][2]) + "  "
//                +String.valueOf(embeedings[0][3]) + "  ");


        // init for distance computation
        float distance = Float.MAX_VALUE;
        String id = "0";
        String label = "?";

//        // if already have recorded faces
//        if (registered.size() > 0) {
//            // find closest face
//            final Pair<String, Float> nearest = findNearest(embeedings[0]);
//            if (nearest != null) {
//                // save closest
//                final String name = nearest.first;
//                label = name;
//                distance = nearest.second;
//                Log.i("NEAREST","nearest: " + name + " - distance: " + distance);
//            }
//        }

        // Recognized face
        final int numDetectionsOutput = 1;
        final ArrayList<SimilarityClassifier.Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
        SimilarityClassifier.Recognition rec = new SimilarityClassifier.Recognition(
                id,
                label,
                distance,
                new RectF());
        // add recognized face to array
        recognitions.add( rec );

        // Save embeedings (signature)
        if (storeExtra) {
            rec.setExtra(embeedings);
        }

        Trace.endSection();
        // return array of recognitions
        return recognitions;
    }




    // looks for the nearest embeeding in the dataset (using L2 norm)
    private int findNearest(float[] emb, List<trainFace> Faces) {

        //init
        int foundFaceId = 0;
        float dist = 9999999F;
        // for each known face
        for (int face_id=0; face_id<Faces.size()-1; face_id++)
        {
            //init
            float currentDist = 0;
            int sizemin = Math.min(emb.length, Faces.get(face_id).embeedings.length);
            // compute L2 distance
            for (int k = 0; k < sizemin; k++) {
                float diff = emb[k] - Faces.get(face_id).embeedings[k];
                currentDist += diff*diff;
            }
            // sqrt
            currentDist = (float) Math.sqrt(currentDist);

            // which face is the nearest?
            if (currentDist < dist)
            {
                //save index
                foundFaceId = face_id;
                //update
                dist = currentDist;
            }

        } // next face

        return foundFaceId;

    }


    /** Memory-map the model file in Assets. */
    private  MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        //TODO: function load java object on file

    }



    private void computeKnownFaces(List<trainFace> trainFaces)
    {
        // image to open
        Mat img;

        //init
        knownFaces.clear();

        File directory = new File(getExternalFilesDir(null).toString()+"/faces/");
        if (directory.exists()) {
            File[] files = directory.listFiles();
            if (files != null) {
                // for each file
                for (int i = 0; i < files.length; ++i) {
                    File file = files[i];
                    if (file.isDirectory()) {
                        // do nothing
                    } else {
                        // todebug
                        Log.i("coucou", "Computing FILE " +directory+ "/" + file.getName() ) ;

                        // add a known face
                        knownFaces.add(new trainFace());
                        // open image
                        img = Imgcodecs.imread(directory + "/"+file.getName());
                        Bitmap faceBitmap = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(img, faceBitmap);
                        //
                        float[][] faceSignature = computeEmbeddings(faceBitmap);
                        // assign embedding to last face
                        knownFaces.get(knownFaces.size()-1).embeedings = faceSignature[0];
                        //assign name
                        knownFaces.get(knownFaces.size()-1).name = file.getName();

                    } // end if fil is a directory
                } // next file
            } //end if file null
        } // end if directory exists


        // for each saved file
        for (int i = 0; i<trainFaces.size(); i++)
        {

        }
    }

    private class trainFace
    {
        // name of the person
        String name = "";
        // Resulting face embeedings (signature) from Facenet
        private float[] embeedings;

    }


    /*** Copy assets
     *
     */

    private void copyAssets() {

/*** copy a file */
// get assets
        AssetManager assetManager = getAssets();

        // list of folders
        String[] folders = null;
        try {
            folders = assetManager.list("");
        } catch (IOException e) {
            Log.e("Assets", "Failed to get asset file list.", e);
        }

        // list of comportemental in folder
        String[] files = null;
        // for each folder
        if (folders != null) for (String foldername : folders) {
            Log.i("Assets", "Found folder: " + foldername  );
            // list of comportemental
            try {
                files = assetManager.list(foldername);
            } catch (IOException e) {
                e.printStackTrace();
            }
            // for each file
            if (files != null) for (String filename : files) {
                Log.i("Assets", "Found comportemental" + foldername + "/" +filename );
                // Files
                InputStream in = null;
                OutputStream out = null;
                //copy file
                try {
                    // open right asset
                    in = assetManager.open(foldername+"/"+filename);
                    // create folder if doesn't exist
                    File folder = new File(getExternalFilesDir(null), foldername);
                    if(!folder.exists())
                        folder.mkdirs();

                    // path in Android/data/<package>/comportemental
                    File outFile = new File(getExternalFilesDir(null), foldername+"/"+filename);
                    // destination file
                    out = new FileOutputStream(outFile);
                    // copy file
                    copyFile(in, out);
                    Log.i("Assets", "Copied " + foldername + "/" +filename );
                } catch(IOException e) {
                    Log.e("tag", "Failed to copy asset file: " + filename, e);
                }
                finally {
                    if (in != null) {
                        try {
                            in.close();
                        } catch (IOException e) {
                            // NOOP
                        }
                    }
                    if (out != null) {
                        try {
                            out.close();
                        } catch (IOException e) {
                            // NOOP
                        }
                    }
                }
            }

        } // next folder

    }// end copyAssets


    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
            out.write(buffer, 0, read);
        }
    }


}