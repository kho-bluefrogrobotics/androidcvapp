package com.bfr.opencvapp;

import static org.opencv.core.CvType.*;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;
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

import org.opencv.videoio.VideoWriter;



import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

import com.bfr.buddysdk.sdk.BuddySDK;

import com.bfr.opencvapp.grafcet.*;
import com.bfr.opencvapp.utils.MultiDetector;
import com.bfr.opencvapp.utils.TfLiteFaceRecognizer;



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

    // for saving face
    boolean isSavingFace = false;
    boolean errorSavingFace = false;
    boolean started = false;

    // context
    Context context = this;
    CheckBox saveCheckbox, preprocessCheckbox;
    Button showAll, removeIdx;
    EditText personNameExitText, idxToRemove;



    //to debug
    double elapsedTime=0.0;
    String toDisplay="";


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

        if(ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED)
        {   //Request permission
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        // configure camera listener
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);



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

    TfLiteFaceRecognizer mytfliterecog;
    public void onCameraViewStarted(int width, int height) {

        try {
            copyAssets();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        mytfliterecog = new TfLiteFaceRecognizer(context);

    }


    @SuppressLint("SuspiciousIndentation")
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // cature frame from camera
        frame = inputFrame.rgba();

        //convert to bitmap
        Mat resizedFaceFrame = new Mat();
        Imgproc.resize(frame, resizedFaceFrame, new Size(255,255));
        Bitmap bitmapImage = Bitmap.createBitmap(resizedFaceFrame.cols(), resizedFaceFrame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(resizedFaceFrame, bitmapImage);

        float[][][][] result=mytfliterecog.recognizeImage(bitmapImage);


        float[] concat = new float[result[0].length*result[0][0].length];

        int dwith = result[0][0].length;
        int dheight = result[0].length;
        Log.w("coucou", " length "+dwith + " " + dheight);

        for (int i=0; i<dheight; i++)
        {
            for (int j=0; j<dwith; j++)
            {
                concat[i*j+j] = result[0][i][j][0]; //Stores element in an array
                //Log.w("coucou", ""+ concat[i*j+j]);
            }
        }

        Bitmap displayBitmap = arrayToBitmap(concat, 320, 256);
        Mat gray = new Mat();
        Utils.bitmapToMat(displayBitmap, gray);
//        Mat gray = new Mat(result[0].length, result[0][0].length, CV_8UC3, new Scalar(0,0,0));
//        Mat gray = new Mat();
//        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGB2GRAY);


//        for (int i=0; i<gray.rows(); i++)
//        {
//            for (int j=0; j<gray.cols(); j++)
//            {
//                double[] newValue = gray.get(i, j); //Stores element in an array
////                newValue[0]=155;
//                newValue[0]=(int)(result[0][i][j][0]*255.0);
////                Log.w("coucou", ""+ result[0][i][j][0] + " " +newValue[0]);
//                gray.put(i, j, newValue); //Puts element back into matrix
//            }
//        }

        Imgproc.resize(gray, gray, new Size(frame.cols(), frame.rows()) );

        return gray;

    } // end function

    public void onCameraViewStopped() {

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