package com.bfr.opencvapp.QrCodeDetector;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.wechat_qrcode.WeChatQRCode;

import java.util.ArrayList;
import java.util.List;

public class QRCodeReader {

    String TAG = "QRCode reader";
    // WeChat QRCode detector
    private WeChatQRCode wechatDetector = null;
    private Net superResoNet;
    String wechatDetectorPrototxtPath = "/sdcard/Download/detect_2021nov.prototxt";
    String wechatDetectorCaffeModelPath = "/sdcard/Download/detect_2021nov.caffemodel";
    String wechatSuperResolutionPrototxtPath = "/sdcard/Download/sr_2021nov.prototxt";
    String wechatSuperResolutionCaffeModelPath = "/sdcard/Download/sr_2021nov.caffemodel";

    // YOLOv4 based custom QR detector
    private Net yoloQrDetector;
    String yoloCFG = "/sdcard/Download/yolov4-tiny-custom-640.cfg";
    String yoloWeights = "/sdcard/Download/yolov4-tiny-custom-640_last.weights";


    //results from Wechat
    List<String> qrCodesContent;
    List<Mat> qrCodesCorner ;//= new ArrayList<Mat>();
    Mat points;// = new Mat();

    //corner positions


    public QRCodeReader()
    {
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
        // super resolution
        superResoNet = Dnn.readNetFromCaffe(wechatSuperResolutionPrototxtPath, wechatSuperResolutionCaffeModelPath);
        wechatDetector.setScaleFactor(1.0f);
        // init result
        qrCodesContent = new ArrayList<String>();
        qrCodesCorner = new ArrayList<Mat>();

        //init YOLO detector
        yoloQrDetector = Dnn.readNetFromDarknet(yoloCFG, yoloWeights);
    }

    public List<QrCode> QRCodeDetectAndDecode(Mat frame)
    {

        List<QrCode> foundQrCodes = new ArrayList<>();

        //reset
        qrCodesContent.clear();
        qrCodesCorner.clear();

        qrCodesContent = wechatDetector.detectAndDecode(frame, qrCodesCorner);

        int x, y;
        // compile results
        for (int i=0; i<qrCodesContent.size(); i++)
        {
//            Log.w(TAG, "QRCOde trouve: " + qrCodesContent.get(i));

            x = (int) (qrCodesCorner.get(i).get(0,0)[0]);
            y = (int) (qrCodesCorner.get(i).get(0,1)[0]);
            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(255, 0, 0), 10);

            x = (int) (qrCodesCorner.get(i).get(1,0)[0]);
            y = (int) (qrCodesCorner.get(i).get(1,1)[0]);
            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(0, 255, 0), 10);

            x = (int) (qrCodesCorner.get(i).get(2,0)[0]);
            y = (int) (qrCodesCorner.get(i).get(2,1)[0]);
            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(0, 0, 255), 10);

            x = (int) (qrCodesCorner.get(i).get(3,0)[0]);
            y = (int) (qrCodesCorner.get(i).get(3,1)[0]);
            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(150, 0, 150), 10);



            foundQrCodes.add(new QrCode(qrCodesContent.get(i), qrCodesCorner.get(i)));
        }

        return foundQrCodes;
    }

}
