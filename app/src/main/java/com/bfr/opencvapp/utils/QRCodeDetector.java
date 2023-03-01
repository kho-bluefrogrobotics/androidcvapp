package com.bfr.opencvapp.utils;

import org.opencv.wechat_qrcode.WeChatQRCode;
import org.opencv.wechat_qrcode.Wechat_qrcode;

public class QRCodeDetector {

    //Wechat detector
    Wechat_qrcode wechatQRCodeDetector;
    // Models
    private final String DRI = "/sdcard/";
    private final String DETECTOR_NAME = "/sdcard/";
    private final String DETECTOR_WEIGHTS = "/sdcard/";
    private final String DECODER_NAME = "/sdcard/";
    private final String DECODER_WEIGHTS = "/sdcard/";

    public QRCodeDetector()
    {

        //Create model
        wechatQRCodeDetector = new WeChatQRCode(  wechatDetectorPrototxtPath,
                wechatDetectorCaffeModelPath,
                wechatSuperResolutionPrototxtPath,
                wechatSuperResolutionCaffeModelPath);
    }
}
