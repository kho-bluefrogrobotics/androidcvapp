package com.bfr.opencvapp;

import static org.opencv.core.CvType.*;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
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
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.opencv.objdetect.FaceRecognizerSF;

import org.opencv.videoio.VideoWriter;

import com.bfr.opencvapp.utils.BuddyData;
import com.bfr.opencvapp.utils.FaceRecognizer;
import com.bfr.opencvapp.utils.FacialIdentity;
import com.bfr.opencvapp.utils.IdentitiesDatabase;
import com.bfr.opencvapp.utils.MLKitFaceDetector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import com.bfr.buddysdk.sdk.BuddySDK;

import com.bfr.opencvapp.grafcet.*;
import com.bfr.opencvapp.utils.MultiDetector;
import com.bfr.opencvapp.utils.TfLiteFaceRecognizer;
import com.google.mlkit.vision.face.Face;


import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;



public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "FaceRecognizerSface_app";

    private CameraBridgeViewBase mOpenCvCameraView;

    // directory where the model files are saved for face detection
//    private String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();
    private String dir = "/sdcard/Android/data/com.bfr.opencvapp/files/";

    // SDK
    BuddySDK mySDK = new BuddySDK();


    //********************  image ***************************

    //Video capture
    Mat frame_orig, frame;
    // Parameters for Base facial detection
    final double THRESHOLD = 0.6;

    //Tflite Multidetector
    MultiDetector multiDetector;
    ArrayList<MultiDetector.Recognition> tfliteDetections = new ArrayList<MultiDetector.Recognition>();
    int left, right, top, bottom;
    // Neural net for detection
    private FaceRecognizer faceRecognizerObj;

    // for saving face
    boolean isSavingFace = false;
    boolean errorSavingFace = false;
    boolean started = false;

    // context
    Context context = this;
    CheckBox saveCheckbox, preprocessCheckbox;
    Button showAll, removeIdx;
    EditText personNameExitText, idxToRemove;

    //Video writer
    private VideoWriter videoWriter;

    //grafcet
    TrackingGrafcet mTrackingGrafcet = new TrackingGrafcet("VisualTracking");
    TrackingYesGrafcet mTrackingYesGrafcet = new TrackingYesGrafcet("VisualTracking");

    // Sensors & motor data
    BuddyData mydata = new BuddyData();

    //to debug
    double elapsedTime=0.0;
    String toDisplay="";

    class Countdown {
        public int time=3;
        public double elapsedTime = 0.0;
        public boolean start = false;
    }
    Countdown countToPicture = new Countdown();

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
        mTrackingGrafcet.init(context, mySDK, mydata);
        mTrackingYesGrafcet.init(context, mySDK, mydata);



        //**************** Callbacks for buttons

        saveCheckbox = findViewById(R.id.saveNameCkbox);
        personNameExitText = findViewById(R.id.personNameEditTxt);
        preprocessCheckbox = findViewById(R.id.PreprocessCheckBox);
        showAll = findViewById(R.id.displayAllBtn);
        removeIdx = findViewById(R.id.removeBtn);
        idxToRemove = findViewById(R.id.idxRemove);

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
                {
                    isSavingFace = isChecked;
                    countToPicture.start = isChecked;
                    countToPicture.time = 3;
                    countToPicture.elapsedTime= System.currentTimeMillis();
                }

            }
        });

        findViewById(R.id.candidatesBtn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                // display all candidates
                int k=5;
                ArrayList<FacialIdentity> candidates = faceRecognizerObj.getTopKResults(k);
                toDisplay="";
                for (int c=0; c<k; c++)
                {
                    toDisplay = toDisplay + "\n" + candidates.get(c).name.split("_")[0] + " "+ String.format(java.util.Locale.US,"%.4f", candidates.get(c).recogScore);
                }
                //display UI
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(getApplicationContext(), toDisplay , Toast.LENGTH_LONG).show();
                    }
                }); // end UI
            }
        });

        preprocessCheckbox.setChecked(true);
        preprocessCheckbox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                faceRecognizerObj.withPreprocess = isChecked;
            }
        });

        showAll.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                // display all stored identities
                toDisplay="";
                for (int c=0; c<faceRecognizerObj.getSavedIdentities().size(); c+=2)
                {
                    try {
                        toDisplay = toDisplay +
                                c + " - " +faceRecognizerObj.getSavedIdentities().get(c).name.split("_")[0]
                                + "    " + (c+1) + " - " +faceRecognizerObj.getSavedIdentities().get(c+1).name.split("_")[0] + "\n";
                    }
                    catch (Exception e)
                    {
                        Log.e(TAG, e.toString());
                    }

                }
                // adjusting to odd number (add last one)
                if (faceRecognizerObj.getSavedIdentities().size()%2!=0)
                    toDisplay = toDisplay  +
                            (faceRecognizerObj.getSavedIdentities().size()-1) + " - " +faceRecognizerObj.getSavedIdentities().get(faceRecognizerObj.getSavedIdentities().size()-1)
                            .name.split("_")[0]+"\n";

                //display UI
                Log.i(TAG, toDisplay);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(getApplicationContext(), toDisplay , Toast.LENGTH_LONG).show();
                    }
                }); // end UI

            } // end onClick
        });

        removeIdx.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try{
                    faceRecognizerObj.removeSavedIdentity(Integer.parseInt(personNameExitText.getText().toString()),
                            new FaceRecognizer.IFaceRecogRsp() {
                                @Override
                                public void onSuccess(String success) {

                                }

                                @Override
                                public void onFailed(String error) {

                                }
                            });
                }
                catch(Exception e)
                {
                    Toast.makeText(getApplicationContext(), "ERROR: " + e.toString() , Toast.LENGTH_LONG).show();
                }


            } // end onClick
        });

        findViewById(R.id.loadBtn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                faceRecognizerObj.loadFaces(new FaceRecognizer.IFaceRecogRsp() {
                    @Override
                    public void onSuccess(String success) {

                    }

                    @Override
                    public void onFailed(String error) {

                    }
                });
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

        frame_orig= new Mat();
        frame = new Mat();
        // init face detector
        multiDetector = new MultiDetector(context);
        //init face recognizer
        faceRecognizerObj = new FaceRecognizer();

//        // Init write video file
//        videoWriter = new VideoWriter("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
//                25.0D, new Size(800, 600));
//        videoWriter.open("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
//                25.0D,  new Size( 800,600));

        started = true;
    }

    @SuppressLint("SuspiciousIndentation")
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // cature frame from camera
        frame_orig = inputFrame.rgba();

        int cols = frame_orig.cols();
        int rows = frame_orig.rows();

        // resize
//        Imgproc.resize(frame_orig, frame, new Size(1024,768));

        frame = frame_orig;


        if (!started)
            return frame;

        try{

        // color conversion
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        elapsedTime = System.currentTimeMillis();


        //convert to bitmap
        Mat resizedFrame = new Mat();
        Imgproc.resize(frame, resizedFrame, new Size(320,320));
        Bitmap bitmapImagefull = Bitmap.createBitmap(resizedFrame.cols(), resizedFrame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(resizedFrame, bitmapImagefull);

        //Face detection
        tfliteDetections = multiDetector.recognizeImage(bitmapImagefull);
        Log.i(TAG, "elapsed time face detect with tflite : " + (System.currentTimeMillis()-elapsedTime)  );
            elapsedTime = System.currentTimeMillis();

        for (int i = 0; i < tfliteDetections.size(); ++i) {
        double confidence = tfliteDetections.get(i).confidence;
        double detectedClass = tfliteDetections.get(i).getDetectedClass();
        if (confidence > THRESHOLD && detectedClass == 1 ) {

                if(isSavingFace)
                {
                    if(countToPicture.start)
                    {
                        //if  face large enough
                        float wThres= 150f/1024f;
                        if(tfliteDetections.get(i).right-tfliteDetections.get(i).left>wThres)
                        {
                            Imgproc.putText(frame_orig, "Placez-vous en face ",
                                    new Point(150, 200),1, 3,
                                    new Scalar(0, 0, 0), 10);
                            Imgproc.putText(frame_orig, "Placez-vous en face",
                                    new Point(150, 200),1, 3,
                                    new Scalar(0, 250, 0), 4);
                            Imgproc.putText(frame_orig, "de la camera" ,
                                    new Point(150, 250),1, 3,
                                    new Scalar(0, 0, 0), 10);
                            Imgproc.putText(frame_orig, "de la camera" + " ",
                                    new Point(150, 250),1, 3,
                                    new Scalar(0, 250, 0), 4);

                            Imgproc.putText(frame_orig, ""+countToPicture.time,
                                    new Point(300, 400),1, 10,
                                    new Scalar(0, 0, 0), 30);
                            Imgproc.putText(frame_orig, ""+countToPicture.time,
                                    new Point(300, 400),1, 10,
                                    new Scalar(255, 0, 0), 15);
                            if (System.currentTimeMillis()-countToPicture.elapsedTime>1000 && countToPicture.time>0) {
                                countToPicture.time-=1;
                                countToPicture.elapsedTime= System.currentTimeMillis();
                            }
                            if (countToPicture.time<=0)
                                countToPicture.start=false;
                            break;
                        }
                        else // face not big enough
                        {
                            Imgproc.putText(frame_orig, "Approchez vous",
                                    new Point(150, 200),1, 3,
                                    new Scalar(0, 0, 0), 10);
                            Imgproc.putText(frame_orig, "Approchez vous",
                                    new Point(150, 200),1, 3,
                                    new Scalar(0, 250, 0), 4);
                            Imgproc.putText(frame_orig, "du robot",
                                    new Point(150, 250),1, 3,
                                    new Scalar(0, 0, 0), 10);
                            Imgproc.putText(frame_orig, "du robot",
                                    new Point(150, 250),1, 3,
                                    new Scalar(0, 250, 0), 4);
                            //reset
                            countToPicture.time = 5;
                            break;
                        }


                    }

                    faceRecognizerObj.saveFace(frame,
                            tfliteDetections.get(i).left,
                            tfliteDetections.get(i).right,
                            tfliteDetections.get(i).top,
                            tfliteDetections.get(i).bottom,
                            personNameExitText.getText().toString(),
                            new FaceRecognizer.IFaceRecogRsp() {
                                @Override
                                public void onSuccess(String success) {
                                    //reset
                                    isSavingFace=false;

                                    //display UI
                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            Toast.makeText(getApplicationContext(), "*** SAVED Face: " + personNameExitText.getText().toString().toUpperCase() ,
                                                    Toast.LENGTH_LONG).show();
                                            saveCheckbox.setChecked(false);
                                        }
                                    }); // end UI
                                }

                                @Override
                                public void onFailed(String error) {
                                    Log.e(TAG, "coucou ERROR : " + error);

                                    errorSavingFace = true;
                                    countToPicture.time = 7;
                                    countToPicture.elapsedTime= System.currentTimeMillis();

                                    //display UI
                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            Toast.makeText(getApplicationContext(), "ERROR !!! Please try again " ,
                                                    Toast.LENGTH_LONG).show();
                                            saveCheckbox.setChecked(false);
                                            isSavingFace=false;
                                        }
                                    }); // end UI
                                }
                            }
                    );


                }
                else if (errorSavingFace)
                {
                        Imgproc.putText(frame_orig, "Erreur merci",
                                new Point(150, 200),1, 3,
                                new Scalar(0, 0, 0), 10);
                        Imgproc.putText(frame_orig, "Erreur merci",
                                new Point(150, 200),1, 3,
                                new Scalar(0, 250, 0), 4);
                        Imgproc.putText(frame_orig, "de ressayer",
                                new Point(150, 250),1, 3,
                                new Scalar(0, 0, 0), 10);
                        Imgproc.putText(frame_orig, "de ressayer",
                                new Point(150, 250),1, 3,
                                new Scalar(0, 250, 0), 4);

                        if (System.currentTimeMillis()-countToPicture.elapsedTime>500 && countToPicture.time>0) {
                            countToPicture.time-=1;
                            countToPicture.elapsedTime= System.currentTimeMillis();
                        }
                        if (countToPicture.time<=0)
                            errorSavingFace=false;
                        break;

                }
                else   // Facial recognition
                {

                   FacialIdentity identified =  faceRecognizerObj.RecognizeFace(frame,
                            tfliteDetections.get(i).left,
                            tfliteDetections.get(i).right,
                            tfliteDetections.get(i).top,
                           tfliteDetections.get(i).bottom);

                    // for display only
                    left = (int)(tfliteDetections.get(i).left * cols);
                    top = (int)(tfliteDetections.get(i).top * rows);
                    right = (int)(tfliteDetections.get(i).right * cols);
                    bottom = (int)(tfliteDetections.get(i).bottom* rows);

                    // Display name
                    if(identified!=null){
                        Imgproc.putText(frame_orig, identified.name.toUpperCase() + " "+String.format(java.util.Locale.US,"%.4f", identified.recogScore),
                                new Point(left-2, top-12),1, 3,
                                new Scalar(0, 0, 0), 5);
                        Imgproc.putText(frame_orig, identified.name.toUpperCase() + " "+String.format(java.util.Locale.US,"%.4f", identified.recogScore),
                                new Point(left-2, top-12),1, 3,
                                new Scalar(0, 255, 0), 2);
                    }
                    // Draw rectangle around detected face.
                    Imgproc.rectangle(frame_orig, new Point(left, top), new Point(right, bottom),
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
            // resize
//            Imgproc.resize(frame_orig, frame, new Size(800,600));
            return frame_orig;
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