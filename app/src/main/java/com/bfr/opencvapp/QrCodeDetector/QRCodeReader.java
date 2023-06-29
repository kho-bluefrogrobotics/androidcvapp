package com.bfr.opencvapp.QrCodeDetector;

import android.util.Log;

import com.bfr.opencvapp.utils.Utils;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.QRCodeDetector;
import org.opencv.wechat_qrcode.WeChatQRCode;

import java.util.ArrayList;
import java.util.List;

public class QRCodeReader {

    String TAG = "QRCode reader";
    // WeChat QRCode detector
    private WeChatQRCode wechatDetector = null;

    // YOLOv4 based custom QR detector
    private Net yoloQrDetector = null;

    //results from Wechat
    List<String> qrCodesContent;
    List<Mat> qrCodesCorner ;//= new ArrayList<Mat>();
    Mat points;// = new Mat();


    public enum DetectionMethod {
        FAST,
        NORMAL,
        HIGH_PRECISION
    }

    public QRCodeReader()
    {
        //init WeChat QRCode detector
        try {
            wechatDetector = new WeChatQRCode(  Utils.wechatDetectorPrototxtPath,
                    Utils.wechatDetectorCaffeModelPath,
                    Utils.wechatSuperResolutionPrototxtPath,
                    Utils.wechatSuperResolutionCaffeModelPath);
        } catch (Exception e) {
            Log.e(TAG, "couldn't initialize wechat detector, check that wechat model files are on the robot");
            e.printStackTrace();
            wechatDetector = null;
        }
        // super resolution set
        wechatDetector.setScaleFactor(1.0f);
        // init result
        qrCodesContent = new ArrayList<String>();
        qrCodesCorner = new ArrayList<Mat>();

        //init YOLO detector
        yoloQrDetector = Dnn.readNetFromDarknet(Utils.yoloQRCodeCFG, Utils.yoloQRCodeWeights);

    }

    public List<QrCode> DetectAndDecode(Mat frame)
    {
        return DetectAndDecode(frame, DetectionMethod.NORMAL);
    }

    private boolean isOverlapping(Mat openCVQRCorners, Mat weChatCorners, double thres)
    {
        try
        {
            //horizontal overlap
            //calc sum of horizontal distances
            double hDiff = Math.abs(weChatCorners.get(0,0)[0]-openCVQRCorners.get(0,0)[0])
                    + Math.abs(weChatCorners.get(1,0)[0]-openCVQRCorners.get(0,1)[0]);

            //vertical overlap
            //calc sum of vertical distances
            double vDiff = Math.abs(weChatCorners.get(0,1)[0]-openCVQRCorners.get(0,0)[1])
                    + Math.abs(weChatCorners.get(3,1)[0]-openCVQRCorners.get(0,3)[1]);

            Log.w("coucou", "diff " + (hDiff+vDiff) );
            if(hDiff + vDiff <=thres)
                return true;
            else
                return false;
        } catch (Exception e) {
            Log.e(TAG, Log.getStackTraceString(e));
            return false;
        }

    }
    public List<QrCode> DetectAndDecode(Mat frame, DetectionMethod method)
    {
        List<QrCode> foundQrCodes = new ArrayList<>();

        // traditional opencv method for QRCode detection
        QRCodeDetector decoder = new QRCodeDetector();
        points = new Mat();
        String data = decoder.detectAndDecode(frame, points);
        if (!points.empty())
            foundQrCodes.add(new QrCode(data, points));

        // Wechat detector
        //reset
        qrCodesContent.clear();
        qrCodesCorner.clear();
        // detect
        qrCodesContent = wechatDetector.detectAndDecode(frame, qrCodesCorner);
        // compile results
        //
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
            if(!isOverlapping( points, qrCodesCorner.get(i),10.0))
                foundQrCodes.add(new QrCode(qrCodesContent.get(i), qrCodesCorner.get(i)));
        }



        if (!points.empty()) {



            Imgproc.circle(frame, new Point(points.get(0,0)[0], points.get(0,0)[1]), 10, new Scalar(255, 0, 0), 2);
            Imgproc.circle(frame, new Point(points.get(0,3)[0], points.get(0,3)[1]), 10, new Scalar(0, 255, 0), 2);




//            for (int i = 0; i < points.cols(); i++) {
//                Point pt1 = new Point(points.get(0, i));
//                Point pt2 = new Point(points.get(0, (i + 1) % 4));
//                Imgproc.line(frame, pt1, pt2, new Scalar(150, 250, 0), 3);
//            }
//            foundQrCodes.add(new QrCode(data, points));

        }



        return foundQrCodes;
    }

}
