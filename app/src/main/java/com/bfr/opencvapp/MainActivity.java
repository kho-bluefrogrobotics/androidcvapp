package com.bfr.opencvapp;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.icu.text.AlphabeticIndex;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;

import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import org.opencv.core.Mat;

import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.QRCodeDetector;
import org.opencv.videoio.VideoWriter;
import org.opencv.wechat_qrcode.WeChatQRCode;
import org.opencv.wechat_qrcode.Wechat_qrcode;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;



public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "QRCode detector";

    private CameraBridgeViewBase mOpenCvCameraView;

    // directory where the model files are saved for face detection
    private String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();


    //button to start tracking
    private Button initBtn;

    // Face UI
    private CheckBox hideFace;



    //********************  image ***************************

    // frame captured by camera
    Mat frame;

    // context
    Context mycontext = this;

    //Video writer
    private VideoWriter videoWriter;



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

        // link to UI
        initBtn = findViewById(R.id.initButton);


        //start tracking button
        initBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.i("Tracking", "Tracking is starting");
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

    // WeChat QRCode detector
    protected WeChatQRCode wechatDetector = null;
    String wechatDetectorPrototxtPath = "/sdcard/Download/detect_2021nov.prototxt";
    String wechatDetectorCaffeModelPath = "/sdcard/Download/detect_2021nov.caffemodel";
    String wechatSuperResolutionPrototxtPath = "/sdcard/Download/sr_2021nov.prototxt";
    String wechatSuperResolutionCaffeModelPath = "/sdcard/Download/sr_2021nov.caffemodel";

    private Net superResNet;
    public void onCameraViewStarted(int width, int height) {

        //init WeChat QRCode detector
        try {
            wechatDetector = new WeChatQRCode(  wechatDetectorPrototxtPath,
                    wechatDetectorCaffeModelPath,
                    wechatSuperResolutionPrototxtPath,
                    wechatSuperResolutionCaffeModelPath);
        } catch (Exception e) {
            Log.e(TAG, "couldn't initialize wechat detector, check that wechat model files are on the robot");
            e.printStackTrace();
            wechatDetector = null;
        }

        //
        superResNet = Dnn.readNetFromCaffe(wechatSuperResolutionPrototxtPath, wechatSuperResolutionCaffeModelPath);
    }


    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        frame = inputFrame.rgba();

        int x = 0;
        int y = 0;

        List<Mat> qrCodesCorner = new ArrayList<Mat>();
        wechatDetector.setScaleFactor(1.0f);
        List<String> qrCodesContent = wechatDetector.detectAndDecode(frame, qrCodesCorner);

        // Super resolution
        Mat frame_resize = new Mat();
        Imgproc.resize(frame, frame_resize, new Size(224, 224));
        Imgproc.cvtColor(frame_resize, frame_resize, Imgproc.COLOR_RGB2GRAY);
        Mat blob = Dnn.blobFromImage(frame_resize, 1/255.0,
                new org.opencv.core.Size(224, 224),
                new Scalar(0.0, 0.0, 0.0), /*swapRB*/false, /*crop*/false);
        superResNet.setInput(blob);
        Mat superResMat = superResNet.forward();

        Log.w(TAG, "mysize:  " + superResMat.size());


        /*** Traditional QRCode detection
         *
         */
        QRCodeDetector decoder = new QRCodeDetector();
        Mat points = new Mat();
        String data = decoder.detectAndDecode(frame, points);




        /***************Display*******************/
        if (qrCodesContent.size()>0)
        {
            Log.w(TAG, "QRCode detected :" + qrCodesContent.get(0));
            Log.w(TAG, "QRCode position :" + qrCodesCorner.get(0).get(0,0)[0]);

            x = (int) (qrCodesCorner.get(0).get(0,0)[0]);
            y = (int) (qrCodesCorner.get(0).get(0,1)[0]);
            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(255, 0, 0), 10);

            x = (int) (qrCodesCorner.get(0).get(1,0)[0]);
            y = (int) (qrCodesCorner.get(0).get(1,1)[0]);
            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(0, 255, 0), 10);

            x = (int) (qrCodesCorner.get(0).get(2,0)[0]);
            y = (int) (qrCodesCorner.get(0).get(2,1)[0]);
            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(0, 0, 255), 10);

            x = (int) (qrCodesCorner.get(0).get(3,0)[0]);
            y = (int) (qrCodesCorner.get(0).get(3,1)[0]);
            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(150, 0, 150), 10);

        }

        if (!points.empty()) {

            for (int i = 0; i < points.cols(); i++) {
                Point pt1 = new Point(points.get(0, i));
                Point pt2 = new Point(points.get(0, (i + 1) % 4));
                Imgproc.line(frame, pt1, pt2, new Scalar(150, 250, 0), 3);
            }

        }

        /***
         *  YOLO custom detector
         */
        return frame;
    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }


}