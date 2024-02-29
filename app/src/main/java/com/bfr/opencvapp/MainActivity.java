package com.bfr.opencvapp;

import static com.bfr.opencvapp.utils.Utils.ANDROID_GREEN;

import static org.opencv.android.Utils.bitmapToMat;

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

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddyActivity;
import com.bfr.buddysdk.BuddySDK;

import com.bfr.opencvapp.grafcet.*;
//import com.bfr.opencvapp.utils.TfLiteMidas;
import com.bfr.opencvapp.utils.TfLiteClassifiier;
import com.bfr.opencvapp.utils.TfLiteYoloX;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.mediapipe.framework.image.BitmapExtractor;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferExtractor;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.segmentation.Segmentation;
import com.google.mlkit.vision.segmentation.SegmentationMask;
import com.google.mlkit.vision.segmentation.Segmenter;
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;


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

    TfLiteClassifiier classifiier;


    ImageSegmenter imagesegmenter;
    MPImage mpImage;


    public void onCameraViewStarted(int width, int height) {

        try {
            copyAssets();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        yoloX = new TfLiteYoloX(context);
//        classifiier = new TfLiteClassifiier(context);

        ImageSegmenter.ImageSegmenterOptions optionsmp =
                ImageSegmenter.ImageSegmenterOptions.builder()
                        .setBaseOptions(
                                BaseOptions.builder().setModelAssetPath("selfie_segmenter_landscape.tflite").build())
                        .setRunningMode(RunningMode.IMAGE)
                        .setOutputCategoryMask(true)
                        .setOutputConfidenceMasks(false)
                        .build();
        imagesegmenter = ImageSegmenter.createFromOptions(context, optionsmp);



    }

    SegmentationMask segmentationMask;

    @SuppressLint("SuspiciousIndentation")
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // cature frame from camera
        frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        int x1=0, y1=0, x2=0, y2=0, classId =0;

//        Imgproc.resize(frame, frame, new Size(800,600));

        // segment floor
//        ArrayList<TfLiteYoloX.Recognition> listOfDetections = yoloX.runInference(frame);
//
//        for (int i = 0; i< listOfDetections.size(); i++)
//        {
//            float score= listOfDetections.get(i).confidence;
//            x1 = (int) (listOfDetections.get(i).left * frame.cols());
//            y1 = (int) (listOfDetections.get(i).top * frame.rows());
//            x2 = (int) (listOfDetections.get(i).right * frame.cols());
//            y2 = (int) (listOfDetections.get(i).bottom * frame.rows());
//            classId = listOfDetections.get(i).getDetectedClass();
//
////            Log.w(TAG, i + " " + classId + " " + score + " coords=" + x1 + "," + y1 + "," + x2 + "," + y2);
//
//            Scalar color = null;
//            switch (classId){
//                case 0:
//                    color = new Scalar(255,0,0);
//                    break;
//                case 1:
//                    color = new Scalar(0,255,0);
//                    break;
//
//                case 2:
//                    color = new Scalar(0,0,255);
//                    break;
//            }
//            Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2), color, 4);
////            Imgproc.putText(frame, String.valueOf(score), new Point(x1, y1-10), 1, 2, new Scalar(0,0,0), 5 );
////            Imgproc.putText(frame, String.valueOf(score), new Point(x1, y1-10), 1, 2, color, 2 );
//
//            Rect toCrop = new Rect(
//                    x1, //limit to ext bound : avoid negative values
//                    y1, //limit to ext bound : avoid negative values
//                    x2-x1, //limit to ext bound : avoid out of the image
//                    y2-y1 //limit to ext bound : avoid out of the image
//            );
//            try{
//                //convert to bitmap
//                Mat croppedTargetMat = frame.submat(toCrop);
//                float[] recog = classifiier.runInference(croppedTargetMat);
//
//                Imgproc.putText(frame, String.valueOf(recog[425]), new Point(x1, y1-10), 1, 2, new Scalar(0,0,0), 5 );
//                Imgproc.putText(frame, String.valueOf(recog[425]), new Point(x1, y1-10), 1, 2, new Scalar(120,255,200), 2 );
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//
//
//        }





        //convert to bitmap
        Mat croppedTargetMat = frame.clone();
//                            Imgproc.resize(croppedTargetMat, croppedTargetMat, new Size(256,256));
        Bitmap bitmapImage = Bitmap.createBitmap(croppedTargetMat.cols(), croppedTargetMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(croppedTargetMat, bitmapImage);

        InputImage inputImage = InputImage.fromBitmap(bitmapImage, 0);

        

        
        
//// MLKit
//        SelfieSegmenterOptions options =
//                new SelfieSegmenterOptions.Builder()
//                        .setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
//                        .enableRawSizeMask()
//                        .build();
//
//        Segmenter segmenter = Segmentation.getClient(options);
//
//
//        Task<SegmentationMask> result =
//                segmenter.process(inputImage)
//                        .addOnSuccessListener(
//                                new OnSuccessListener<SegmentationMask>() {
//                                    @Override
//                                    public void onSuccess(SegmentationMask mask) {
//                                        // Task completed successfully
//                                        // ...
//                                        Log.i(TAG, "Task Complete!");
//                                        segmentationMask = mask;
//
//
//                                        ByteBuffer maskbuff = segmentationMask.getBuffer();
//                                        int maskWidth = segmentationMask.getWidth();
//                                        int maskHeight = segmentationMask.getHeight();
//                                        Bitmap maskBitmap = Bitmap.createBitmap(maskWidth,maskHeight, Bitmap.Config.RGB_565);
//
//                                        for (int y = 0; y < maskHeight; y++) {
//                                            for (int x = 0; x < maskWidth; x++) {
//                                                // Gets the confidence of the (x,y) pixel in the mask being in the foreground.
//                                                float foregroundConfidence = maskbuff.getFloat();
//                                                if (foregroundConfidence>0.8)
//                                                {
////                                                    Log.i(TAG, "proba " + x + "," + y + " = " + foregroundConfidence);
//                                                    maskBitmap.setPixel(x, y, ANDROID_GREEN);
//                                                }
//                                                Utils.bitmapToMat(maskBitmap, frame);
//
//                                                Imgproc.resize(frame, frame, new Size(1024,768));
//                                            }
//                                        }
//
//
//                                    } //end onsuccess
//                                })
//                        .addOnFailureListener(
//                                new OnFailureListener() {
//                                    @Override
//                                    public void onFailure(@NonNull Exception e) {
//                                        // Task failed with an exception
//                                        // ...
//                                    }
//                                });
////
////                try {
////                        Tasks.await(result);
////                    } catch (Exception e) {
////                        e.printStackTrace();
////                    }



        // Mediapipe


        // Convert an Android’s Bitmap object to a MediaPipe’s Image object.
        mpImage = new BitmapImageBuilder(bitmapImage).build();
        ImageSegmenterResult segmenterResult = imagesegmenter.segment(mpImage);


//       Bitmap resultingBitmap = BitmapExtractor.extract( segmenterResult.categoryMask().get());

        try{
//            ByteBuffer myByteBuffer =  ByteBufferExtractor.extract(segmenterResult.categoryMask().get());
//            Bitmap resultingBitmap = BitmapExtractor.extract( segmenterResult.categoryMask().get());

            MPImage categoryMask =  segmenterResult.categoryMask().get();
            ByteBuffer myByteBuffer =  ByteBufferExtractor.extract(categoryMask);


            int[] pixels = new int[myByteBuffer.capacity()];
            int[] originalPixels  = new int[bitmapImage.getWidth()*bitmapImage.getHeight()];

            bitmapImage.getPixels(originalPixels, 0, bitmapImage.getWidth(),
                    0, 0, bitmapImage.getWidth(), bitmapImage.getHeight());

            for (int ii=0; ii<pixels.length; ii++)
            {
                if(myByteBuffer.get(ii)>=0) // if something else recognized than background
                    pixels[ii] = originalPixels[ii]; //get(crop) pixel value from the captured image
            }

            Bitmap resultingbmp = Bitmap.createBitmap(
                    pixels,
                    categoryMask.getWidth(),
                    categoryMask.getHeight(),
                    Bitmap.Config.ARGB_8888
            );
            Mat todisplay = new Mat();

            bitmapToMat(resultingbmp, todisplay);
            return todisplay;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return frame;


    } // end function



    public void onCameraViewStopped() {

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