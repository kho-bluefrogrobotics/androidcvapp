package com.bfr.opencvapp;


import static org.opencv.core.CvType.*;
import static org.opencv.videoio.Videoio.CAP_PROP_POS_FRAMES;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Environment;
import android.os.RemoteException;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
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
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.opencv.objdetect.FaceRecognizerSF;

import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;



import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddyActivity;
import com.bfr.buddysdk.BuddySDK;

import com.bfr.opencvapp.grafcet.*;
import com.bfr.opencvapp.MultiDetector;
import com.bfr.opencvapp.utils.TfLiteMidas;
import com.bfr.opencvapp.utils.TfLiteYoloXHumanHeadHands;


public class MainActivity extends BuddyActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "opencvApp";

    private CameraBridgeViewBase mOpenCvCameraView;

    // directory where the model files are saved for face detection
//    private String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();
    private String dir = "/sdcard/Android/data/com.bfr.opencvapp/files/";


TrackingNoGrafcet trackingNoGrafcet = new TrackingNoGrafcet("TrackingNo");
TrackingYesGrafcet trackingYesGrafcet = new TrackingYesGrafcet("TrackingYes");

    //********************  image ***************************

    //Video capture
    Mat frame_orig, frame;
    VideoCapture videoCapture;
    // Parameters for Base facial detection
    final double THRESHOLD = 0.6;

    double elpasedtime = 0.0;

    //Tflite Multidetector
    MultiDetector detector;
    ArrayList<MultiDetector.Recognition> tfliteDetections = new ArrayList<MultiDetector.Recognition>();
    int left, right, top, bottom;

    // Pose estimator
    TfLiteBlazePose blazePose;

    public static PersonTracker personTracker;
    Rect tracked;

    TfLiteYoloXHumanHeadHands humanHeadHandsDetector;

    // context
    Context context = this;
    public CheckBox alignCheckbox;
    public CheckBox recordCkbx;
    Button initButton;

    private ImageView cameraImageView;
    private CameraBridgeViewBase cameraBridgeViewBase;
    private CameraBridgeViewBase.CvCameraViewListener2 cameraListener;


    public AlignGrafcet alignGrafcet = new AlignGrafcet("BodyAlign") ;

    //todebug
    boolean recording=false;
    VideoWriter videoWriter;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // run only in Landscape mode
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        setContentView(R.layout.activity_main);

        // link with UI
        alignCheckbox = findViewById(R.id.alignBox);
        initButton= findViewById(R.id.initButton);
        recordCkbx = findViewById(R.id.recordCkbx);

        // Check permissions
        if(ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED)
        {   //Request permission
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }

        if(ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED)
        {   //Request permission
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        // configure camera listener
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCameraPermissionGranted();
        mOpenCvCameraView.setCvCameraViewListener(this);


        /*** Listeners*/

        alignCheckbox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                trackingNoGrafcet.go=b;
                trackingYesGrafcet.go=b;
                alignGrafcet.go=b;

                if(!b)
                {
                    BuddySDK.USB.enableWheels(0, 0, new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {

                        }

                        @Override
                        public void onFailed(String s) throws RemoteException {

                        }
                    });
                    BuddySDK.USB.enableYesMove(0, new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {

                        }

                        @Override
                        public void onFailed(String s) throws RemoteException {

                        }
                    });
                    BuddySDK.USB.enableNoMove(0, new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {

                        }

                        @Override
                        public void onFailed(String s) throws RemoteException {

                        }
                    });
                }

            }
        });

        initButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                trackingNoGrafcet.go = false;
                trackingNoGrafcet.step_num = 0;

                trackingYesGrafcet.go = false;
                trackingYesGrafcet.step_num = 0;
                alignGrafcet.go = false;
                alignGrafcet.step_num = 0;

//                int targetX = (int) (personTracker.tracked.box.x + personTracker.tracked.box.width/2);
//
//                float noAngle = (targetX-(1024/2))*0.09375f;
//                Log.i("trackingNo", "Rotating to " + noAngle);
//                BuddySDK.USB.buddySayNo(45.0f, noAngle, new IUsbCommadRsp.Stub() {
//                    @Override
//                    public void onSuccess(String s) throws RemoteException {
//
//                    }
//
//                    @Override
//                    public void onFailed(String s) throws RemoteException {
//
//                    }
//                });//end moveNo


            }
        });


        recordCkbx.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {

                if (b)
                {
                    int fourcc = VideoWriter.fourcc('M','J','P','G');

                    LocalDateTime myDateObj = LocalDateTime.now();
                    DateTimeFormatter myFormatObj = DateTimeFormatter.ofPattern("yyMMddHHmmss");

                    String formattedDate = myDateObj.format(myFormatObj);
                    String debugFileName = "/storage/emulated/0/Download/" + formattedDate + "_trackingDebug.avi" ;
                    videoWriter = new VideoWriter(debugFileName, fourcc,
                            13, new Size(1024, 768));
                    videoWriter.open(debugFileName, fourcc,
                            13, new Size(1024, 768));
                    recording = true;
                }
                else {
                    videoWriter.release();
                    recording = false;
                }
            }
        });

        // create folder if doesn't exist
        File folder = new File("/sdcard/Download/trackingdebug/", "");
        if(!folder.exists()) {
            // create folder
            folder.mkdirs();
        }

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
        try{
            alignGrafcet.stop();
        } catch (Exception e) {
            e.printStackTrace();
        }


    }

    @Override
    public void onResume() {
        super.onResume();
        // OpenCV manager initialization
        OpenCVLoader.initDebug();
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        Log.w("coucou", "coucou onResume");


    }


    @Override
    public void onDestroy() {
        super.onDestroy();

    }


        TfLiteMidas mytfliterecog;
    public void onCameraViewStarted(int width, int height) {

        try {
            copyAssets();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        Log.w("coucou", "coucou started");

        detector = new MultiDetector(this);
        blazePose = new TfLiteBlazePose(context);
        humanHeadHandsDetector = new TfLiteYoloXHumanHeadHands(context);

        personTracker = new PersonTracker(detector, blazePose, humanHeadHandsDetector);

        tracked = new Rect();

        videoCapture = new VideoCapture("/sdcard/Download/240314105955_trackingDebug.avi");
        videoCapture.set(CAP_PROP_POS_FRAMES, 10);
        frame = new Mat();



    }

    @SuppressLint("SuspiciousIndentation")
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // cature frame from camera
        frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2BGR);


//        videoCapture.read(frame);

        if (recording)
            videoWriter.write(frame);

        try
        {
            personTracker.visualTracking(frame, false, true);
            personTracker.readyToDisplay =false;

            Imgproc.cvtColor( personTracker.displayMat,  personTracker.displayMat, Imgproc.COLOR_RGB2BGR);
            return personTracker.displayMat;
        }
        catch (Exception e)
        {
            return new Mat(768,1024, CV_8UC3, new Scalar(0, 0, 0));

        }


    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }



    private Bitmap arrayToBitmap(float[] img_array, int imageSizeX, int imageSizeY) {
        float maxval = Float.NEGATIVE_INFINITY;
        float minval = Float.POSITIVE_INFINITY;
        for (float cur : img_array) {
            maxval = Math.max(maxval, cur);
            minval = Math.min(minval, cur);
        }
        float multiplier = 0;
        if ((maxval - minval) > 0) multiplier = 255 / (maxval - minval);

        int[] img_normalized = new int[img_array.length];
        for (int i = 0; i < img_array.length; ++i) {
            float val = (float) (multiplier * (img_array[i] - minval));
            img_normalized[i] = (int) val;
        }

        int width = imageSizeX;
        int height = imageSizeY;
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);

        for (int ii = 0; ii < width; ii++) //pass the screen pixels in 2 directions
        {
            for (int jj = 0; jj < height; jj++) {
                //int val = img_normalized[ii + jj * width];
                int index = (width - ii - 1) + (height - jj - 1) * width;
                if(index < img_array.length) {
                    int val = img_normalized[index];
                    bitmap.setPixel(ii, jj, Color.rgb(val, val, val));
                }
            }
        }

        return bitmap;
    }

    @Override
    public void onSDKReady() {

        Log.w("coucou","coucou onSDKReady");

        BuddySDK.USB.enableNoMove(1, new IUsbCommadRsp.Stub() {
            @Override
            public void onSuccess(String s) throws RemoteException {

            }

            @Override
            public void onFailed(String s) throws RemoteException {

            }
        });
        BuddySDK.USB.enableYesMove(1, new IUsbCommadRsp.Stub() {
            @Override
            public void onSuccess(String s) throws RemoteException {

            }

            @Override
            public void onFailed(String s) throws RemoteException {

            }
        });

        trackingNoGrafcet.start(20);
        trackingYesGrafcet.start(20);
        alignGrafcet.start(10);

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

            if (foldername.contains("images") || foldername.contains("imgs") ||
                    foldername.contains("mlkit_") || foldername.contains("mobile_") ||
                    foldername.contains("_bundled") || foldername.contains("shaders")||
                    foldername.contains("webkit")
            )
                continue;

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