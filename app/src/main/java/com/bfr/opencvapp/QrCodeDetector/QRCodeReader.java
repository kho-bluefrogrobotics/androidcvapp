package com.bfr.opencvapp.QrCodeDetector;

import static org.opencv.core.CvType.CV_32F;

import android.util.Log;

import com.bfr.opencvapp.utils.Utils;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
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
    /**
     Checks if 2 QRCodes are overlapping,
     typically to filter out a weChat detected QRCode if already detected by the traditional way
     */
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

            //Log.w("coucou", "diff " + (hDiff+vDiff) );
            if(hDiff + vDiff <=thres)
                return true;
            else
                return false;
        } catch (Exception e) {
            Log.e(TAG, Log.getStackTraceString(e));
            return false;
        }

    }

    public List<QrCode> Detect(Mat frame)
    {
        return Detect(frame, DetectionMethod.NORMAL);
    }
    public List<QrCode> Detect(Mat frame, DetectionMethod method)
    {

        List<QrCode> foundQrCodes = new ArrayList<>();

        // traditional opencv method for QRCode detection
        QRCodeDetector decoder = new QRCodeDetector();
        points = new Mat();
        String data = decoder.detectAndDecode(frame, points);
//        if (!points.empty())
//            foundQrCodes.add(new QrCode(data, points));

        // Wechat detector
        //reset
        qrCodesContent.clear();
        qrCodesCorner.clear();
        // detect
        qrCodesContent = wechatDetector.detectAndDecode(frame, qrCodesCorner);
        int x=0, y=0;
        for (int i=0; i<qrCodesContent.size(); i++)
        {

//            x = (int) (qrCodesCorner.get(i).get(0,0)[0]);
//            y = (int) (qrCodesCorner.get(i).get(0,1)[0]);
//            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(255, 0, 0), 10);
//
//            x = (int) (qrCodesCorner.get(i).get(1,0)[0]);
//            y = (int) (qrCodesCorner.get(i).get(1,1)[0]);
//            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(0, 255, 0), 10);
//
//            x = (int) (qrCodesCorner.get(i).get(2,0)[0]);
//            y = (int) (qrCodesCorner.get(i).get(2,1)[0]);
//            Imgproc.circle(frame, new Point(x, y), 2, new Scalar(0, 0, 255), 10);
//
//            x = (int) (qrCodesCorner.get(i).get(3,0)[0]);
//            y = (int) (qrCodesCorner.get(i).get(3,1)[0]);
//            Log.w(TAG, "QRCOde trouve: " + qrCodesContent.get(i));

//            foundQrCodes.add(new QrCode("wechat"+i, qrCodesCorner.get(i)));

        }


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
                if (conf >= 0.6) {
                    int center_x = (int) (detect.get(0) * frame_yolo.cols());
                    int center_y = (int) (detect.get(1) * frame_yolo.rows());
                    int width = (int) (detect.get(2) * frame_yolo.cols());
                    int height = (int) (detect.get(3) * frame_yolo.rows());

                    //Imgproc.circle(frame, new Point(center_x,center_y), width, new Scalar(255, 0, 0), 5);
                    //Imgproc.putText(frame, String.valueOf(conf), new Point(center_x,center_y), 2, 1, new Scalar(200, 255, 10));

                    //register corner
                    int topleftX = (center_x - width / 2);
                    int topleftY = (center_y - height / 2);
                    int toprightX = topleftX+width;
                    int toprightY = topleftY;
                    int bottomleftX = topleftX;
                    int bottomleftY = topleftY+height;
                    int bottomrightX = toprightX;
                    int bottomrightY = bottomleftY;
                    Mat corners = new Mat(4,2, CvType.CV_32FC1);
                    corners.put(0,0, new double[]{topleftX });
                    corners.put(0,1, new double[]{topleftY });
                    corners.put(1,0, new double[]{toprightX });
                    corners.put(1,1, new double[]{toprightY });
                    corners.put(2,0, new double[]{bottomrightX });
                    corners.put(2,1, new double[]{bottomrightY });
                    corners.put(3,0, new double[]{bottomleftX });
                    corners.put(3,1, new double[]{bottomleftY });
                    foundQrCodes.add(new QrCode("yolo", corners));
                            //                    int x = (center_x - width / 2);
//                    int y = (center_y - height / 2);
//                    Rect2d box = new Rect2d(x, y, width, height);
//                    result.get("boxes").add(box);
//                    result.get("confidences").add(conf);
//                    result.get("class_ids").add(class_id);
                        }
                    }
                }


        return foundQrCodes;
    }

    public List<QrCode> DetectAndDecode(Mat frame)
    {
        return DetectAndDecode(frame, DetectionMethod.NORMAL);
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
