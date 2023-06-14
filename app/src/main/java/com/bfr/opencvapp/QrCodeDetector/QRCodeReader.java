package com.bfr.opencvapp.QrCodeDetector;

import android.util.Log;

import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.wechat_qrcode.WeChatQRCode;

public class QRCodeReader {

    // WeChat QRCode detector
    protected WeChatQRCode wechatDetector = null;
    String wechatDetectorPrototxtPath = "/sdcard/Download/detect_2021nov.prototxt";
    String wechatDetectorCaffeModelPath = "/sdcard/Download/detect_2021nov.caffemodel";
    String wechatSuperResolutionPrototxtPath = "/sdcard/Download/sr_2021nov.prototxt";
    String wechatSuperResolutionCaffeModelPath = "/sdcard/Download/sr_2021nov.caffemodel";

    // YOLOv4 based custom QR detector
    private Net yoloQrDetector;
    String yoloCFG = "/sdcard/Download/yolov4-tiny-custom-640.cfg";
    String yoloWeights = "/sdcard/Download/yolov4-tiny-custom-640_last.weights";

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
        //
        superResNet = Dnn.readNetFromCaffe(wechatSuperResolutionPrototxtPath, wechatSuperResolutionCaffeModelPath);

        //init YOLO detector
        yoloQrDetector = Dnn.readNetFromDarknet(yoloCFG, yoloWeights);
    }
}
