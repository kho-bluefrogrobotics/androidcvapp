package com.bfr.opencvapp;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


import com.bfr.opencvapp.QrCodeDetector.QRCodeReader;
import com.bfr.opencvapp.QrCodeDetector.QrCode;
import com.bfr.opencvapp.utils.Utils;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;

import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.dnn_superres.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.QRCodeDetector;
import org.opencv.videoio.VideoWriter;
import org.opencv.wechat_qrcode.WeChatQRCode;


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

    private final int IMG_WIDTH = 800;
    private final int IMG_HEIGHT = 600;


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

    String yoloCFG = "/sdcard/Download/yolov4-tiny-custom-640.cfg";
    String yoloWeights = "/sdcard/Download/yolov4-tiny-custom-640_last.weights";

    private Net superResNet;

    // Yolo
    private Net yoloQrDetector;

    //Super resolution
    DnnSuperResImpl mSupRes = null;

    QRCodeReader mQRCodeReader;

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
        //
        yoloQrDetector = Dnn.readNetFromDarknet(yoloCFG, yoloWeights);

        //super resolution
        mSupRes = DnnSuperResImpl.create();
//        mSupRes.readModel( "/sdcard/Download/EDSR_x4.pb");
        mSupRes.readModel( "/sdcard/Download/ESPCN_x4.pb");
//        mSupRes.setModel("edsr", 4);
        mSupRes.setModel("espcn", 4);

        mQRCodeReader = new QRCodeReader();

    }

    /**
     Returns index of maximum element in the list
     */
    private  int argmax(List<Float> array)
    {
    float max = array.get(0);
    int re = 0;
        for (int i = 1; i < array.size(); i++) {
        if (array.get(i) > max) {
            max = array.get(i);
            re = i;
        }
    }
        return re;
}
    List<String> qrCodesContent;
    List<Mat> qrCodesCorner ;//= new ArrayList<Mat>();
    Mat points;// = new Mat();
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        frame = inputFrame.rgba();

        int x = 0;
        int y = 0;

        //crop image around face
//        Rect resizeROI= new Rect(frame.cols()/4, frame.rows()/4,3*frame.cols()/4, 3*frame.rows()/4 );
        Rect resizeROI= new Rect(frame.cols()/4, frame.rows()/4,frame.cols()/4, frame.rows()/4 );
        Mat resized = frame.submat(resizeROI);

        // Super resolution upsample
        Mat supResMat = new Mat();
        Mat inSuperReso = new Mat();
//        Imgproc.cvtColor(frame, inSuperReso, Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(resized, inSuperReso, Imgproc.COLOR_RGB2YCrCb);
        mSupRes.upsample(inSuperReso, supResMat);
        Log.w("Upsampling", "Resulting mat: " + supResMat.size());
//        Imgproc.resize(resized, resized , new Size(1024,768));
        Imgproc.cvtColor(supResMat, frame, Imgproc.COLOR_YCrCb2RGB);
//        frame = supResMat;

        List<QrCode> listQr =  mQRCodeReader.Detect(frame, QRCodeReader.DetectionMethod.HIGH_PRECISION);

        try {
            if (listQr.size() > 0) {
                // for each QRCode
                for (int i = 0; i < listQr.size(); i++) {
                    Log.w("sizescoucou", i + " " + listQr.get(i).rawContent + " " + listQr.get(i).matOfCorners.size());
                    x = (int) (listQr.get(i).matOfCorners.get(0, 0)[0]);
                    y = (int) (listQr.get(i).matOfCorners.get(0, 1)[0]);
                    Imgproc.circle(frame, new Point(x, y), 2, new Scalar(255, 0, 0), 10);

                    x = (int) (listQr.get(i).matOfCorners.get(1, 0)[0]);
                    y = (int) (listQr.get(i).matOfCorners.get(1, 1)[0]);
                    Imgproc.circle(frame, new Point(x, y), 2, new Scalar(0, 255, 0), 10);

                    x = (int) (listQr.get(i).matOfCorners.get(2, 0)[0]);
                    y = (int) (listQr.get(i).matOfCorners.get(2, 1)[0]);
                    Imgproc.circle(frame, new Point(x, y), 2, new Scalar(0, 0, 255), 10);

                    x = (int) (listQr.get(i).matOfCorners.get(3, 0)[0]);
                    y = (int) (listQr.get(i).matOfCorners.get(3, 1)[0]);
                    Imgproc.circle(frame, new Point(x, y), 2, new Scalar(150, 0, 150), 10);


                    if (listQr.get(i).poseKnown) {
                        String angle = "" + listQr.get(i).getAngle(12.5);
//                Log.w(TAG, "QRCOde trouve: " + angle + " " + listQr.get(0).getTranslationVector().get(1,0));
                        Imgproc.putText(frame, angle, new Point(x, y+20), 1, 2, new Scalar(0, 0, 255), 6);
                        Imgproc.putText(frame, angle, new Point(x, y+20), 1, 2, new Scalar(255, 255, 255), 2);

                        Imgproc.putText(frame, "" + listQr.get(i).qrCodeTranslation.get(2, 0)[0], new Point(x, y+60), 1, 2, new Scalar(0, 0, 255), 6);
                        Imgproc.putText(frame, "" + listQr.get(i).qrCodeTranslation.get(2, 0)[0], new Point(x, y+60), 1, 2, new Scalar(255, 255, 255), 2);
                    } else {
                        Imgproc.putText(frame, "POSE UNKNONW", new Point((int) IMG_WIDTH / 4, (int) IMG_HEIGHT / 4), 1, 2, new Scalar(0, 0, 255), 6);
                        Imgproc.putText(frame, "POSE UNKNONW", new Point((int) IMG_WIDTH / 4, (int) IMG_HEIGHT / 4), 1, 2, new Scalar(255, 255, 255), 2);
                    } //end if pose known
                } //next qrcode
            } //end if list of QRCode empty

        } catch (Exception e) {
            e.printStackTrace();
        }


        return frame ;


//        return frame;

    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }


}