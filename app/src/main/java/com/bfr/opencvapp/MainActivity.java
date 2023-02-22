package com.bfr.opencvapp;

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
import android.widget.RelativeLayout;
import android.widget.Switch;

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
import com.bfr.opencvapp.utils.MLKitFaceDetector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collections;
import java.util.List;

import com.bfr.buddysdk.sdk.BuddySDK;

import com.bfr.opencvapp.grafcet.*;
import com.google.mlkit.vision.face.Face;

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

    // MLKit face detector
    MLKitFaceDetector myMLKitFaceDetector = new MLKitFaceDetector();

    // image rotation
    Point center;
    double angle = 90;
    double scale = 1.0;
    Mat mapMatrix;

    // context
    Context mycontext = this;

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
//        //calbacks for Enable button
//        noSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
//            @Override
//            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
//                // if checked
//                if (noSwitch.isChecked())
//                {
//                    // enable No
//                    try {
//                        mySDK.getUsbInterface().enableNoMove(1, new IUsbCommadRsp.Stub() {
//                            @Override
//                            public void onSuccess(String success) throws RemoteException { }
//                            @Override
//                            public void onFailed(String error) throws RemoteException {}
//                        });
//                    } //end try enable Yes
//                    catch (RemoteException e)
//                    {            e.printStackTrace();
//                    } // end catch
//                }
//                else // unchecked
//                {
//                    //disable no
//                    try {
//                        mySDK.getUsbInterface().enableNoMove(0, new IUsbCommadRsp.Stub() {
//                            @Override
//                            public void onSuccess(String success) throws RemoteException { }
//                            @Override
//                            public void onFailed(String error) throws RemoteException {}
//                        });
//
//                    } //end try enable Yes
//                    catch (RemoteException e)
//                    {
//                        e.printStackTrace();
//                    } // end catch
//                } // end if toggle
//
//            } // end onChecked
//        }); // end listener

        //tracking
//        trackingCheckBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
//            @Override
//            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
//                //reset
//                if (!trackingCheckBox.isChecked())
//                {
//                    // reset grafcet
//                    TrackingGrafcet.step_num =0;
//                    TrackingGrafcet.go = false;
//
//                    TrackingYesGrafcet.step_num =0;
//                    TrackingYesGrafcet.go = false;
//                    // close record file
//
//                    videoWriter.release();
//
//                }
//                else // checked
//                {
//                    // let the grafcet continue
//                    TrackingGrafcet.go = true;
//                    TrackingYesGrafcet.go = true;
//
//                }
//
//            }
//        });


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

    boolean started = false;
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
        // Load Face detection model
        String proto = dir + "/nnmodels/opencv_face_detector.pbtxt";
        String weights = dir + "/nnmodels/opencv_face_detector_uint8.pb";
        net = Dnn.readNetFromTensorflow(weights, proto);

        // Load face recog model
        faceRecognizer = FaceRecognizerSF.create(dir + "/nnmodels/face_recognition_sface_2021dec.onnx",
                "");


        // Init write video file
        videoWriter = new VideoWriter("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
                25.0D, new Size(800, 600));
        videoWriter.open("/storage/emulated/0/saved_video.avi", VideoWriter.fourcc('M','J','P','G'),
                25.0D,  new Size( 800,600));

        started = true;
    }


    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // cature frame from camera
        Mat frame = inputFrame.rgba();

        if (!started)
            return frame;

        // color conversion
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        // Forward image through network.
        blob =  Dnn.blobFromImage(frame, 1.0,
                new org.opencv.core.Size(300, 300),
                new Scalar(104, 117, 123), /*swapRB*/true, /*crop*/false);
        net.setInput(blob);
        Mat detections = net.forward();

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
                // Draw rectangle around detected object.
//                Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
//                        new Scalar(0, 255, 0), 2);

                /*** Recognition ***/

                //crop image around face
                Rect faceROI= new Rect((int)(left- MARGIN_FACTOR *(right-left)),
                        (int)(top- MARGIN_FACTOR *(bottom-top)),
                        (int)(right-left)+(int)(2* MARGIN_FACTOR *(right-left)),
                        (int)(bottom-top) + +(int)(MARGIN_FACTOR *(bottom-top)));
                Mat faceMat = frame.submat(faceROI);

                ////////////////////////////// face orientation
                //convert to bitmap
                Bitmap bitmapImage = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(frame, bitmapImage);
                // detect-classify face
                Face detectedFace = myMLKitFaceDetector.detectSingleFaceFromBitmap(bitmapImage);

                if(detectedFace==null){
                    Imgproc.putText(frame, "MLKIT Failure", new Point(100, 100),1, 2,
                            new Scalar(0, 255, 0), 2);
                    return frame;
                }
                // image rotation
                Point center = new Point((int)faceMat.cols()/2,(int) faceMat.rows()/2);
                double angle = detectedFace.getHeadEulerAngleZ();
                Imgproc.putText(frame, " "+angle, new Point(100, 100),1, 2,
                        new Scalar(0, 255, 0), 2);
                Mat mapMatrix = Imgproc.getRotationMatrix2D(center, -angle, 1.0);
                // rotate
                Imgproc.warpAffine(faceMat, faceMat, mapMatrix, new Size(faceMat.cols(), faceMat.rows()));

                ////////////////////////////// Matching
                // Compute embeddings
//                faceBlob = Dnn.blobFromImage(faceMat, 1, inputFaceSize, new Scalar(0, 0, 0), true, false);
//                sfaceNet.setInput(faceBlob);
//                faceEmbedding = sfaceNet.forward();

                faceRecognizer.feature(faceMat, faceEmbedding);

                // Look for closest
                // for each known face
                faceRecognizer.match(faceEmbedding, faceEmbedding, FaceRecognizerSF.FR_COSINE);


            }   // end if confidence OK
        } // next detection

        return frame;
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