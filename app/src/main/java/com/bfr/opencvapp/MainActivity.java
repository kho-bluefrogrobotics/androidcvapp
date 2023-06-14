package com.bfr.opencvapp;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC3;
import static org.opencv.core.CvType.CV_8U;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
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

import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;

import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.dnn_superres.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.QRCodeDetector;
import org.opencv.videoio.VideoWriter;
import org.opencv.wechat_qrcode.WeChatQRCode;
import org.opencv.wechat_qrcode.Wechat_qrcode;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import boofcv.abst.fiducial.QrCodeDetector;
import boofcv.alg.fiducial.qrcode.QrCode;
import boofcv.android.ConvertBitmap;
import boofcv.factory.fiducial.ConfigQrCode;
import boofcv.factory.fiducial.FactoryFiducial;
import boofcv.struct.image.GrayU8;


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

    String yoloCFG = "/sdcard/Download/yolov4-tiny-custom-640.cfg";
    String yoloWeights = "/sdcard/Download/yolov4-tiny-custom-640_last.weights";

    private Net superResNet;

    // Yolo
    private Net yoloQrDetector;

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
        //DnnSuperResImpl mSupRes = DnnSuperResImpl.create();
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


        Thread wechat = new Thread(new Runnable() {
            @Override
            public void run() {
                wechatDetector.setScaleFactor(1.0f);
                qrCodesCorner = new ArrayList<Mat>();
                qrCodesContent = wechatDetector.detectAndDecode(frame, qrCodesCorner);
            }
        });

        wechat.start();


        // Super resolution
//        Mat frame_resize = new Mat();
//        Imgproc.resize(frame, frame_resize, new Size(224, 224));
//        Mat frame_resize = frame.clone();
//        Imgproc.cvtColor(frame_resize, frame_resize, Imgproc.COLOR_RGB2GRAY);
//        Mat blob = Dnn.blobFromImage(frame_resize, 1/255.0,
//                new org.opencv.core.Size(224, 224),
//                new Scalar(0.0, 0.0, 0.0), /*swapRB*/false, /*crop*/false);
//        superResNet.setInput(blob);
//        Mat superResMat = superResNet.forward();
//        Log.w(TAG, "mysize:  " + superResMat.size());


        /*** Traditional QRCode detection
         *
         */
        Thread tradi = new Thread(new Runnable() {
            @Override
            public void run() {
                QRCodeDetector decoder = new QRCodeDetector();
                points = new Mat();
                String data = decoder.detectAndDecode(frame, points);
            }
        });

        tradi.start();




        /***
         *  YOLO custom detector
         */
        Thread yolo = new Thread(new Runnable() {
            @Override
            public void run() {
                yoloQrDetector.getLayerNames();
                Mat frame_yolo = new Mat();
                Imgproc.cvtColor(frame, frame_yolo, Imgproc.COLOR_RGBA2RGB);
                Mat blob = Dnn.blobFromImage(frame_yolo, 1/255.0,
                        new org.opencv.core.Size(640, 640),
                        new Scalar(new double[]{0.0, 0.0, 0.0}), /*swapRB*/true, /*crop*/false, CV_32F);
                yoloQrDetector.setInput(blob);

                //  -- determine  the output layer names that we need from YOLO
                // The forward() function in OpenCV’s Net class needs the ending layer till which it should run in the network.

                List<String> layerNames = yoloQrDetector.getLayerNames();
                List<String> outputLayers = new ArrayList<String>();
                for (Integer i : yoloQrDetector.getUnconnectedOutLayers().toList()) {
                    outputLayers.add(layerNames.get(i - 1));
                }

                List<Mat> outputs = new ArrayList<Mat>();
                yoloQrDetector.forward(outputs, outputLayers);

                for(Mat output : outputs) {
                    //  loop over each of the detections. Each row is a candidate detection,
                    System.out.println("Output.rows(): " + output.rows() + ", Output.cols(): " + output.cols());
                    for (int i = 0; i < output.rows(); i++) {
                        Mat row = output.row(i);
                        List<Float> detect = new MatOfFloat(row).toList();
                        List<Float> score = detect.subList(5, output.cols());
                        int class_id = argmax(score); // index maximalnog elementa liste
                        float conf = score.get(class_id);
                        if (conf >= 0.5) {
                            int center_x = (int) (detect.get(0) * frame_yolo.cols());
                            int center_y = (int) (detect.get(1) * frame_yolo.rows());
                            int width = (int) (detect.get(2) * frame_yolo.cols());
                            int height = (int) (detect.get(3) * frame_yolo.rows());

                            Imgproc.circle(frame, new Point(center_x,center_y), width, new Scalar(255, 0, 0), 5);
                            Imgproc.putText(frame, String.valueOf(conf), new Point(center_x,center_y), 2, 1, new Scalar(200, 255, 10));
//                    int x = (center_x - width / 2);
//                    int y = (center_y - height / 2);
//                    Rect2d box = new Rect2d(x, y, width, height);
//                    result.get("boxes").add(box);
//                    result.get("confidences").add(conf);
//                    result.get("class_ids").add(class_id);
                        }
                    }
                }
            }
        });

        yolo.start();

//        /***
//         * Boofcv
//         */
//
//        // convert mat to boof cv
//        Bitmap bmp = null;
//        Mat rgb = new Mat();
//        Imgproc.cvtColor(frame, rgb, Imgproc.COLOR_BGR2RGB);
//        try {
//            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
//            Utils.matToBitmap(rgb, bmp);
//        }
//        catch (CvException e){
//            Log.d("Exception",e.getMessage());
//        }
//
//        // Easiest way to convert a Bitmap into a BoofCV type
//        GrayU8 image = ConvertBitmap.bitmapToGray(bmp, (GrayU8)null, null);
//
//        ConfigQrCode config = new ConfigQrCode();
////		config.considerTransposed = false; // by default, it will consider incorrectly encoded markers. Faster if false
//        QrCodeDetector<GrayU8> detector = FactoryFiducial.qrcode(config, GrayU8.class);
//
//        detector.process(image);
//
//        // Gets a list of all the qr codes it could successfully detect and decode
//        List<QrCode> detections = detector.getDetections();
//
//        if(detections.size()>0)
//        {
//            Log.w(TAG, "Found QRCode with boofCV");
//        }

        try {
            wechat.join();
            yolo.join();
            tradi.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
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


        return frame;
    } // end function

    public void onCameraViewStopped() {
        videoWriter.release();
    }


}