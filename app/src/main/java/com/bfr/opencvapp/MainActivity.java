package com.bfr.opencvapp;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.RemoteException;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddyActivity;
import com.bfr.buddysdk.BuddySDK;

import com.bfr.opencvapp.grafcet.*;
//import com.bfr.opencvapp.utils.TfLiteMidas;
import com.bfr.opencvapp.utils.TfLiteYoloX;

public class MainActivity extends BuddyActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "FaceRecognizerSface_app";

    private CameraBridgeViewBase mOpenCvCameraView;

    // directory where the model files are saved for face detection
//    private String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();
    private String dir = "/sdcard/Android/data/com.bfr.opencvapp/files/";



    //********************  image ***************************

    //Video capture
    Mat frame_orig, frame;

    // context
    Context context = this;
    public static CheckBox alignCheckbox;
    Button initButton;

    private ImageView cameraImageView;
    private CameraBridgeViewBase cameraBridgeViewBase;
    private CameraBridgeViewBase.CvCameraViewListener2 cameraListener;


    public AlignGrafcet alignGrafcet ;


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
                Log.i("AlignGrafcet", "init all");
                alignGrafcet.go = false;
                alignGrafcet.step_num = 0;

                takephoto = true;
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

    boolean takephoto = false;
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


    TfLiteYoloX yoloX;

    public void onCameraViewStarted(int width, int height) {

        try {
            copyAssets();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        yoloX = new TfLiteYoloX(context);

    }



    @SuppressLint("SuspiciousIndentation")
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // cature frame from camera
        frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

//        Imgproc.resize(frame, frame, new Size(800,600));

        // segment floor
        ArrayList<TfLiteYoloX.Recognition> listOfDetections = yoloX.runInference(frame);

        for (int i = 0; i< listOfDetections.size(); i++)
        {
            float score= listOfDetections.get(i).confidence;
            int x1 = (int) (listOfDetections.get(i).left * frame.cols());
            int y1 = (int) (listOfDetections.get(i).top * frame.rows());
            int x2 = (int) (listOfDetections.get(i).right * frame.cols());
            int y2 = (int) (listOfDetections.get(i).bottom * frame.rows());
            int classId = listOfDetections.get(i).getDetectedClass();

//            Log.w(TAG, i + " " + classId + " " + score + " coords=" + x1 + "," + y1 + "," + x2 + "," + y2);

            Scalar color = null;
            switch (classId){
                case 0:
                    color = new Scalar(255,0,0);
                    break;
                case 1:
                    color = new Scalar(0,255,0);
                    break;

                case 2:
                    color = new Scalar(0,0,255);
                    break;
            }
            Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2), color, 4);

        }



        return frame;


    } // end function


    private int classOfPixel(Mat result, int pixIdx)
    {
        // crop just one col
        //crop image
        Rect colROI= new Rect(
                // alternative to crop more: Math.max(left, (int)(left- MARGIN_FACTOR *(right-left)) ),
                0,
                pixIdx,
                //alternative to crop more: (int)(right-left)+(int)(MARGIN_FACTOR *(right-left)),
                1,
                64 );

        Mat colInResult = result.submat(colROI);

        return (int) Core.minMaxLoc(colInResult).maxLoc.y;
    }

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

    @Override
    public void onSDKReady() {

        Log.w("coucou","coucou onSDKReady");

        alignGrafcet = new AlignGrafcet("AlignGrafcet");
        alignGrafcet.start();
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