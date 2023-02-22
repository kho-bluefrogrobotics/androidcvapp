package com.bfr.opencvapp;

import static org.opencv.core.CvType.*;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.RelativeLayout;
import android.widget.Switch;
import android.widget.Toast;

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
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import org.opencv.objdetect.FaceRecognizerSF;

import org.opencv.videoio.VideoWriter;

import com.bfr.opencvapp.utils.BuddyData;
import com.bfr.opencvapp.utils.FacialIdentity;
import com.bfr.opencvapp.utils.IdentitiesDatabase;
import com.bfr.opencvapp.utils.MLKitFaceDetector;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import com.bfr.buddysdk.sdk.BuddySDK;

import com.bfr.opencvapp.grafcet.*;
import com.google.mlkit.vision.face.Face;


import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;



public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "FaceRecognizerSface";

    private CameraBridgeViewBase mOpenCvCameraView;

    // directory where the model files are saved for face detection
//    private String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();
    private String dir = "/sdcard/Android/data/com.bfr.opencvapp/files/";

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

    // Parameters for Base facial detection
    final int IN_WIDTH = 300;
    final int IN_HEIGHT = 300;
    final float WH_RATIO = (float)IN_WIDTH / IN_HEIGHT;
    final double IN_SCALE_FACTOR = 0.007843;
    final double MEAN_VAL = 127.5;
    final double THRESHOLD = 0.8;
    // List of detected faces
    Mat blob, detections;
    // Neural net for detection
    private Net net;
    //
    float MARGIN_FACTOR = 0.1f;

    //Parameters for Facial recognition
    Size inputFaceSize = new Size(112,112);
    // List of detected faces
    Mat faceBlob;
    // Neural net for detection
    private Net sfaceNet;
    private FaceRecognizerSF faceRecognizer;
    Mat faceEmbedding;
    Mat faceMat;
    Rect faceROI;

    // MLKit face detector
    MLKitFaceDetector myMLKitFaceDetector = new MLKitFaceDetector();

    // image rotation
    Point center;
    double angle = 90;
    double scale = 1.0;
    Mat mapMatrix;

    // for saving face
    boolean isSavingFace = false;
    boolean started = false;
    // List of known faces
    private IdentitiesDatabase idDatabase;
//    public ArrayList<FacialIdentity> identities = new ArrayList<>();

    SimpleDateFormat sdf = new SimpleDateFormat("yyMMddHHmmss");
    String currentDateandTime = "";

    // context
    Context mycontext = this;
    CheckBox saveCheckbox;
    EditText personNameExitText;

    //Video writer
    private VideoWriter videoWriter;

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


        // Pass SDK and data to grafcet
        mTrackingGrafcet.init(mycontext, mySDK, mydata);
        mTrackingYesGrafcet.init(mycontext, mySDK, mydata);



        //**************** Callbacks for buttons

        saveCheckbox = findViewById(R.id.saveNameCkbox);
        personNameExitText = findViewById(R.id.personNameEditTxt);

//        //callback show face
//        hideFace.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
//            @Override
//            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
//                //if checked
//                if (hideFace.isChecked())
//                {   // set tranparent
//                    BuddyFace.setAlpha(0.25F);
//                }
//                else // unchecked
//                {// set opaque
//                    BuddyFace.setAlpha(1.0F);
//                } // end if checked
//            } // end onchange
//        });// end listener
//

        saveCheckbox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {

                if (isChecked)
                    isSavingFace = isChecked;
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

        try {
            copyAssets();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        // init
        faceEmbedding=new Mat();

        idDatabase= new IdentitiesDatabase();
        try {
            FileInputStream fileIn = new FileInputStream("/sdcard/identities.ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            idDatabase.identities = ( ArrayList<FacialIdentity>) in.readObject();
            in.close();
            fileIn.close();

            Log.w(TAG, "coucou loading identities file : " + idDatabase.identities.get(idDatabase.identities.size()-1).name);
        } catch (Exception e) {
           Log.e(TAG, "Error loading identities file : " + e);

        }

        idDatabase = new IdentitiesDatabase();
        idDatabase.loadFromStorage();

//        identities.add(new FacialIdentity("UNKNOWN", new Mat(1, 128 , CV_32F)));
//        identities.add(new FacialIdentity("UNKNOWN", Mat.zeros(1,128, CV_32F)));

        // Load Face detection model
        String proto = dir + "/nnmodels/opencv_face_detector.pbtxt";
        String weights = dir + "/nnmodels/opencv_face_detector_uint8.pb";
        net = Dnn.readNetFromTensorflow(weights, proto);

        // Load face recog model
        faceRecognizer = FaceRecognizerSF.create(dir + "/nnmodels/face_recognition_sface_2021dec.onnx",
//        faceRecognizer = FaceRecognizerSF.create(dir + "/nnmodels/face_recognition_sface_2021dec-act_int8-wt_int8-quantized.onnx",
                "");
        sfaceNet = Dnn.readNetFromONNX(dir + "/nnmodels/face_recognition_sface_2021dec.onnx");

//        // Init write video file
//        videoWriter = new VideoWriter("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
//                25.0D, new Size(800, 600));
//        videoWriter.open("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
//                25.0D,  new Size( 800,600));

        started = true;
    }

    double elapsedTime=0.0;
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // cature frame from camera
        Mat frame = inputFrame.rgba();

        if (!started)
            return frame;

        try{


        // color conversion
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        elapsedTime = System.currentTimeMillis();

        // Forward image through network.
        blob =  Dnn.blobFromImage(frame, 1.0,
                new org.opencv.core.Size(300, 300),
                new Scalar(104, 117, 123), /*swapRB*/true, /*crop*/false);
        net.setInput(blob);
        Mat detections = net.forward();

        Log.i(TAG, "elapsed time face detect : " + (System.currentTimeMillis()-elapsedTime)  );
        elapsedTime = System.currentTimeMillis();

        int cols = frame.cols();
        int rows = frame.rows();
        detections = detections.reshape(1, (int)detections.total() / 7);

        //for each detected face
        for (int i = 0; i < detections.rows(); ++i) {
            double confidence = detections.get(i, 2)[0];
            if (confidence > THRESHOLD) {
                int left   = (int)(detections.get(i, 3)[0] * cols);
                int top    = (int)(detections.get(i, 4)[0] * rows);
                int right  = (int)(detections.get(i, 5)[0] * cols);
                int bottom = (int)(detections.get(i, 6)[0] * rows);

                //If face touches the margin, skip -> we need a fully visible face for recognition
                if(left<10 || top <10 || right> frame.cols()-10 || bottom > frame.rows()-10)
                    // take next detected face
                    continue;
                /*** Recognition ***/


                    //crop image around face
                    faceROI= new Rect( Math.max(left, (int)(left- MARGIN_FACTOR *(right-left)) ),
                            Math.max(top, (int)(top- MARGIN_FACTOR *(bottom-top)) ),
                            (int)(right-left)+(int)(2* MARGIN_FACTOR *(right-left)),
                            (int)(bottom-top) + +(int)(MARGIN_FACTOR *(bottom-top)));
                    faceMat = frame.submat(faceROI);



                ////////////////////////////// face orientation
                //convert to bitmap
                Bitmap bitmapImage = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(frame, bitmapImage);
                // detect-classify face
                Face detectedFace = myMLKitFaceDetector.detectSingleFaceFromBitmap(bitmapImage);

                Log.i(TAG, "elapsed time face MLKit : " + (System.currentTimeMillis()-elapsedTime)  );
                elapsedTime = System.currentTimeMillis();

                if(detectedFace==null){
                    Imgproc.putText(frame, "MLKIT Failure", new Point(100, 100),1, 2,
                            new Scalar(0, 255, 0), 2);
                    return frame;
                }
                // image rotation
                Point center = new Point((int)faceMat.cols()/2,(int) faceMat.rows()/2);
                double angle = detectedFace.getHeadEulerAngleZ();
                //Imgproc.putText(frame, " "+angle, new Point(100, 100),1, 2,
                //        new Scalar(0, 255, 0), 2);
                mapMatrix = Imgproc.getRotationMatrix2D(center, -angle, 1.0);
                // rotate
                Imgproc.warpAffine(faceMat, faceMat, mapMatrix, new Size(faceMat.cols(), faceMat.rows()));

                Log.i(TAG, "elapsed time rotation : " + (System.currentTimeMillis()-elapsedTime)  );
                elapsedTime = System.currentTimeMillis();

                // Compute embeddings
//                faceBlob = Dnn.blobFromImage(faceMat, 1, inputFaceSize, new Scalar(0, 0, 0), true, false);
//                sfaceNet.setInput(faceBlob);
//                faceEmbedding = sfaceNet.forward();

                faceRecognizer.feature(faceMat, faceEmbedding);

                Log.i(TAG, "elapsed time calc embedding : " + (System.currentTimeMillis()-elapsedTime)
                +"\n " + faceEmbedding.size() +" type " + faceEmbedding.depth());
                elapsedTime = System.currentTimeMillis();

                if(isSavingFace)
                {
                    ////////////////////////////// Saving new face

                    currentDateandTime = sdf.format(new Date());
//                    Mat newEmbeddings = faceEmbedding.clone();
                    idDatabase.identities.add(new FacialIdentity(personNameExitText.getText()+"_"+currentDateandTime, faceEmbedding.clone()));

                    //saving file
                    Thread savingThread = new Thread(new Runnable() {
                        @Override
                        public void run() {

                            idDatabase.saveToStorage();
                            // reset
                            isSavingFace = false;
                        }
                    });
                    savingThread.start();


                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(getApplicationContext(), "*** SAVED Face: " + personNameExitText.getText()+"_"+currentDateandTime ,
                                    Toast.LENGTH_LONG).show();
                            saveCheckbox.setChecked(false);
                        }
                    });

                }
                else
                {
                    ////////////////////////////// Matching

                    // Look for closest
                    double cosineScore, maxScore = 0.0;
                    int identifiedIdx=0;

                    Log.i(TAG, "Saved face database: " + idDatabase.identities.size());

                    // for each known face
                    for (int faceIdx=1; faceIdx< idDatabase.identities.size(); faceIdx++)
                    {
                        Log.i(TAG, "Checking face database: " + faceIdx + " " + idDatabase.identities.get(faceIdx).name);
                        Log.i(TAG, "FaceEmbed : " + faceEmbedding.size() );
                        Log.i(TAG, "Reference : " + idDatabase.identities.get(faceIdx).embedding.size() );

                        //compute similarity;
                        cosineScore = faceRecognizer.match(faceEmbedding, idDatabase.identities.get(faceIdx).embedding,
                                FaceRecognizerSF.FR_COSINE);
                        Log.i(TAG, "TODEBUG FaceCOmputing : " + idDatabase.identities.get(faceIdx).name + " - " + cosineScore
                         + "   " + idDatabase.identities.get(faceIdx).embedding.get(0,0)[0]
                         + " " + idDatabase.identities.get(faceIdx).embedding.get(0,1)[0]
                         + " " + idDatabase.identities.get(faceIdx).embedding.get(0,2)[0]
                         + " " + idDatabase.identities.get(faceIdx).embedding.get(0,3)[0]
                         + " " + idDatabase.identities.get(faceIdx).embedding.get(0,4)[0]
                        );
                        // if better score
                        if(cosineScore>maxScore) {
                            maxScore = cosineScore;
                            identifiedIdx = faceIdx;
                        }
                    }

                    //
                    //Imgproc.putText(frame, identities.get(identifiedIdx).name, new Point(100, 100),1, 2,
                    Imgproc.putText(frame, idDatabase.identities.get(identifiedIdx).name.split("_")[0].toUpperCase(), new Point(left, top-10),1, 3,
                                    new Scalar(0, 0, 250), 2);
                    Imgproc.putText(frame, idDatabase.identities.get(identifiedIdx).name.split("_")[0].toUpperCase(), new Point(left-2, top-12),1, 3,
                                    new Scalar(0, 255, 0), 2);
//                    Log.i(TAG, "Found face : " + identities.get(identifiedIdx).name );

                    // Draw rectangle around detected face.
                    Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                        new Scalar(0, 255, 0), 2);

                } //end if isSavingFace

                //Stop for-loop (only one face)
                break;

            }   // end if confidence OK

        } // next detection

        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
        finally {
            return frame;
        }
    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }




    private void copyAssets() throws IOException {
        Log.i(TAG, "Copying assets"  );
/*** copy a file */
// get assets
        AssetManager assetManager = getAssets();

        // list of folders
        String[] folders = null;

        folders = assetManager.list("");

        // list of comportemental in folder
        String[] files = null;
        // for each folder
        if (folders != null) for (String foldername : folders) {
            Log.i(TAG, "Found folder: " + foldername  );
            // list of comportemental
            try {
                files = assetManager.list(foldername);
            } catch (IOException e) {
                e.printStackTrace();
            }
            // for each file
            if (files != null) for (String filename : files) {
//                Log.i("Assets", "Found comportemental" + foldername + "/" +filename );
                // Files
                InputStream in = null;
                OutputStream out = null;
                //copy file
                try {
                    // open right asset
                    in = assetManager.open(foldername+"/"+filename);

                    {
                        // create folder if doesn't exist
                        File folder = new File(getExternalFilesDir(null), foldername);
                        if(!folder.exists()) {
                            // create folder
                            folder.mkdirs();
                        }
                        // if file to keep (vocal, commercial, ...)
                        File file = new File ( foldername+"/"+filename );

                        if ( !file.exists() )
                        {
                            Log.d(TAG, "Asset not found on device " + file.getAbsolutePath() );
//                        if (!fileExist("/storage/emulated/0/Android/data/com.bfr.buddywelcomehost/files/"+foldername+"/"+filename)) {
                            // path in Android/data/<package>/comportemental
                            File outFile = new File(getExternalFilesDir(null), foldername + "/" + filename);
                            // destination file
                            out = new FileOutputStream(outFile);
                            // copy file
                            copyFile(in, out);
                            Log.i(TAG, "Assets Copied " + foldername + "/" + filename);
                        } else {
                            Log.i(TAG, "Assets File already Found " +  file.getAbsolutePath());
                        } //end if file exists
                    }
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


    // copy files (assets)
    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
            out.write(buffer, 0, read);
        }
    }
    // check file existence in storage
    public boolean fileExist(String fname){
        File file = getBaseContext().getFileStreamPath(fname);
        return file.exists();
    }

}