package com.bfr.opencvapp.QrCodeDetector;

import static com.bfr.opencvapp.utils.Utils.matToMatOfPoints2f;
import static org.opencv.core.CvType.CV_32F;

import android.util.Log;

import com.bfr.opencvapp.utils.Utils;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
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
        wechatDetector.setScaleFactor(2.0f);
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
     Get the centroid of the qrcode
     */
    protected Point computeCentroid(MatOfPoint2f points){
        Point centroid = new Point();
        double xSum = 0;
        double ySum = 0;
        List<Point> pointList = points.toList();
        int pointNumber = pointList.size();
        for(Point p : pointList){
            xSum += p.x;
            ySum += p.y;
        }

        centroid.x = xSum / pointNumber;
        centroid.y = ySum / pointNumber;

        return centroid;
    }
    /**
     Checks if 2 QRCodes are overlapping,
     typically to filter out a weChat detected QRCode if already detected by the traditional way
     */
    private boolean isOverlapping(Mat firstCorners, Mat secondCorners)
    {
        boolean horizOverlap = false;
        boolean vertOverlap = false;

        try
        {
            Point firstCenter= computeCentroid(matToMatOfPoints2f(firstCorners));
            //horizontal limits
            double maxX =0;
            double minX =99999999;
            for (int i=0; i<secondCorners.rows(); i++)
            {
                if (maxX<=secondCorners.get(i,0)[0])
                {   //asign max value
                    maxX=secondCorners.get(i,0)[0];
                }
                if (minX>=secondCorners.get(i,0)[0])
                {   //asign min value
                    minX=secondCorners.get(i,0)[0];
                }
            }

            //if center of first QRCode between the horizontal limits
            horizOverlap = (firstCenter.x>=minX) && (firstCenter.x<=maxX);



            //vertical limits
            double maxY =0;
            double minY =99999999;
            for (int i=0; i<secondCorners.rows(); i++)
            {
                if (maxY<=secondCorners.get(i,1)[0])
                {   //asign max value
                    maxY=secondCorners.get(i,1)[0];
                }
                if (minY>=secondCorners.get(i,1)[0])
                {   //asign min value
                    minY=secondCorners.get(i,1)[0];
                }
            }

            //if center of first QRCode between the horizontal limits
            vertOverlap = (firstCenter.y>=minY) && (firstCenter.y<=maxY);

           return (horizOverlap && vertOverlap);

        } catch (Exception e) {
            Log.e(TAG, Log.getStackTraceString(e));
            return false;
        }

    }

    public List<QrCode> DetectAndDecode(Mat frame)
    {
        return DetectAndDecode(frame, DetectionMethod.NORMAL);
    }

    public List<QrCode> DetectAndDecode(Mat frame, DetectionMethod method)
    {

        List<QrCode> foundQrCodes = new ArrayList<>();
        List<QrCode> foundByOpenCV = new ArrayList<>();
        List<QrCode> foundByWeChat = new ArrayList<>();
        List<QrCode> foundByYolo = new ArrayList<>();

        /*** Opencv detector ***/
        Thread openCVThread = new Thread(new Runnable() {
            @Override
            public void run() {
                // traditional opencv method for QRCode detection
                QRCodeDetector decoder = new QRCodeDetector();
                points = new Mat();
                String data = decoder.detectAndDecode(frame, points);
                if (!points.empty())
                    // add detected QRCode to list
                    foundByOpenCV.add(new QrCode(data, points, true));
            }
        });
        openCVThread.start();

        /*** Wechat detector ***/
        Thread wechatThread = new Thread(new Runnable() {
            @Override
            public void run() {

                //reset
                qrCodesContent.clear();
                qrCodesCorner.clear();
                // detect
                qrCodesContent = wechatDetector.detectAndDecode(frame, qrCodesCorner);
                int x=0, y=0;
                for (int i=0; i<qrCodesContent.size(); i++)
                {
                    // add detected QRCode to list
                    foundByWeChat.add(new QrCode(qrCodesContent.get(i), qrCodesCorner.get(i), true));
                }
            }
        });
        if(method==DetectionMethod.NORMAL || method==DetectionMethod.HIGH_PRECISION)
            wechatThread.start();

        /*** YOLO detector ***/
        Thread yoloThread = new Thread(new Runnable() {
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
                        int class_id = argmax(score);
                        float conf = score.get(class_id);
                        if (conf >= 0.5) {
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
                            // add detected QRCode to list
                            foundByYolo.add(new QrCode("", corners, false));
                        } // end if conf
                    } // next yolo detection
                }//next output
            }
        });
        if(method==DetectionMethod.HIGH_PRECISION)
            yoloThread.start();


        try {
            openCVThread.join();
            //for each found QRCode
            for(int i=0; i<foundByOpenCV.size(); i++)
            {
                Log.w("coucou", "trouve par opencv " + i + " "+ foundByOpenCV.get(i).rawContent);
                //add QRCode to the final list
                foundQrCodes.add(foundByOpenCV.get(i));
            }
            if(method==DetectionMethod.NORMAL || method==DetectionMethod.HIGH_PRECISION)
            {
                wechatThread.join();
                //for each found QRCode
                for(int i=0; i<foundByWeChat.size(); i++)
                {
                    Log.w("coucou", "trouve par Wechat " + i + " "+ foundByWeChat.get(i).rawContent);
                    boolean alreadyDetected=false;
                    // for each already registered QRCodes
                    for (int j=0;j<foundQrCodes.size(); j++)
                        //if overlapping with a QRCode
                        if(isOverlapping(foundByWeChat.get(i).matOfCorners, foundQrCodes.get(j).matOfCorners))
                            alreadyDetected = true;

                    // if not overlapping with any of the registered QRCodes
                    if (!alreadyDetected)
                        //add QRCode to the final list
                        foundQrCodes.add(foundByWeChat.get(i));
                }
            }

            if(method==DetectionMethod.HIGH_PRECISION)
            {
                yoloThread.join();
                //for each found QRCode
                for(int i=0; i<foundByYolo.size(); i++)
                {
                    Log.w("coucou", "trouve par Yolo " + i + " "+ foundByYolo.get(i).rawContent);
                    boolean alreadyDetected=false;
                    // for each already registered QRCodes
                    for (int j=0;j<foundQrCodes.size(); j++)
                        //if overlapping with a QRCode
                        if(isOverlapping(foundByYolo.get(i).matOfCorners, foundQrCodes.get(j).matOfCorners))
                            alreadyDetected = true;

                    // if not overlapping with any of the registered QRCodes
                    if (!alreadyDetected)
                        //add QRCode to the final list
                        foundQrCodes.add(foundByYolo.get(i));
                } //next yolo qrcode
            } //

        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return foundQrCodes;
    }



}
