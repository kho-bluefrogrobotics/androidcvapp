package com.bfr.opencvapp.utils;

import static org.opencv.core.CvType.CV_8UC3;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.media.Image;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class Utils {

    private static final String TAG = "SERVICE_VISION_utils";

    public enum CvAlgoType {
        FACE_DETECTION,
        PERSON_DETECTION,
        COLOR_RECOGNITION,
        APRILTAG_DETECTION,
        MOTION_DETECTION,
        VISUAL_TRACKING,
        FACE_RECOGNITION,
        QRCODE_DETECTION,
        POSE_ESTIMATION;
    }

    public static class Color{

        public static final Scalar
                _RED =  new Scalar(255,0,0),
                _BLUE = new Scalar(0,0,255),
                _GREEN = new Scalar(0,255,0),
                _YELLOW = new Scalar(150,150,0),
                _PURPLE = new Scalar(155,0,155),
                _WHITE = new Scalar(255,255,255),
                _BLACK = new Scalar(0,0,0);
    }

    //RGB color for bitmaps
    public static int ANDROID_BLACK = android.graphics.Color.rgb(0, 0, 0);
    public static int ANDROID_WHITE = android.graphics.Color.rgb(255, 255, 255);
    public static int ANDROID_RED = android.graphics.Color.rgb(255, 0, 0);
    public static int ANDROID_BLUE = android.graphics.Color.rgb(0, 0, 255);
    public static int ANDROID_GREEN = android.graphics.Color.rgb(0, 255, 0);


    /**
    /*** Path for models (should be automatically copied at startup from the assets)***/
    //where to find the models
    private final String DIR = "/sdcard/Android/data/com.bfr.opencvapp/files/nnmodels/";
    public static String modelsDir = "/sdcard/Android/data/com.bfr.opencvapp/files/nnmodels/";
    // QRCode
    public static String wechatDetectorPrototxtPath = modelsDir + "detect_2021nov.prototxt";
    public static String wechatDetectorCaffeModelPath = modelsDir + "detect_2021nov.caffemodel";
    public static String wechatSuperResolutionPrototxtPath = modelsDir + "sr_2021nov.prototxt";
    public static String wechatSuperResolutionCaffeModelPath = modelsDir + "sr_2021nov.caffemodel";

    public static String yoloQRCodeCFG = modelsDir + "yolov4-tiny-custom-640.cfg";
    public static String yoloQRCodeWeights = modelsDir + "yolov4-tiny-custom-640_last.weights";

    // Human Tracking
    public static String vitTrackerModel = modelsDir + "object_tracking_vittrack_2023sep.onnx";

    /*** Camera calibrations ***/
    //wideangle 640x480
//    public static final double[][] cameraCalibrationMatrixCoeff = {{347.1784748095083 , 0       , 326.6795720628966},
//            {0        , 345.1916479410069, 233.1696799590856},
//            {0        , 0       , 1       }};
//    public static final double[] distortionCoeff = {-0.27803360529321036, 0.06339702764223658, -0.000203214885858507, -0.0014783300318126165};

    //    //wide angle 1024x768
    public static final double[][] cameraCalibrationMatrixCoeff = {{614.9288 , 0       , 511.257},
            {0        , 604.4629, 383.5352},
            {0        , 0       , 1       }};
    public static final double[] distortionCoeff = {-0.1024,-0.3817, -0.0044, 0.0025, 0.0247};

    //zoom 640x480
//    private final double[][] cameraCalibrationMatrixCoeff = {{622.1937 , 0       , 399.7594},
//            {0        , 616.9286, 299.5756},
//            {0        , 0       , 1       }};
//    private final double[] distortionCoeff = {0.0598, -0.0309, -0.0038, -0.0033, -0.0872};

    /** transfrom matrix from type Mat to MatOfPoint2f
     *
     * @param matrix - either
     * @return - MatOfPoint2f
     */
    public static MatOfPoint2f matToMatOfPoints2f(Mat matrix){
        MatOfPoint2f resultMat = new MatOfPoint2f();
        List<Point> pointList = new ArrayList<Point>();
        // TODO : manage "all" matrix not only opencv and wechat corners
        if(matrix.size(0) == 1){ // matrix has size 1xN with two channels (corners detected by OpenCv)
            for(int i=0; i<matrix.size(1); i++){
                double[] point = matrix.get(0, i);
                if(point.length == 2){
                    pointList.add(new Point(point[0], point[1]));
                }
                else{
                    Log.e(TAG + " - matToMatOfPoint2f", "point should have a size of 2 but is " + point.length );
                }
            }
        }
        else if(matrix.size(1) == 2){ // matrix has size N*2 with one channel (corners detected by wechat)
            for(int i=0; i<matrix.size(0); i++){
                double pointU = matrix.get(i, 0)[0];
                double pointV = matrix.get(i, 1)[0];
                pointList.add(new Point(pointU, pointV));
            }
        }
        resultMat.fromList(pointList);
        return resultMat;
    }

    public static void logMat(String logTag, Mat mat){
        String result = "\n";
        for(int i=0; i<mat.size(0); i++){
            for(int j=0; j< mat.size(1); j++){
                result += ("|" + mat.get(i, j)[0]);
            }
            result += "|\n";
        }
        Log.i(logTag, "size " + mat.size() +"\n"+ result);
    }

    public static double distanceBetweenPoints(Point p1, Point p2){
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y- p2.y, 2));
    }

    /**
     * converts a Bitmap to byte[] array.
     * @param bitmap : bitmap to convert.
     * @return : the resulting byte[].
     */
    public static byte[] getBytesFromBitmap(Bitmap bitmap) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
        return stream.toByteArray();
    }

    /**
     * Convert a byte[] array to a Bitmap.
     * @param bytes : the byte[] to convert.
     * @return : the resulting bitmap.
     */
    public static Bitmap getBitmapFromBytes(byte[] bytes) {
        Bitmap bitmap = null;
        try {
            bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
        } catch (Exception e) {
            Log.e(TAG,"Erreur pendant la conversion du bitmap en byte[] : "+e);
        }
        return bitmap;
    }

    /**
     * converts an android.media.Image to a cv Mat .
     * @param image : the image to convert.
     * @return : the resulting cv Mat.
     */
    public static Mat imageToMat(Image image) {
        ByteBuffer bb = image.getPlanes()[0].getBuffer();
        byte[] buf = new byte[bb.remaining()];
        bb.get(buf);

        Mat mMat = new Mat();
        // byte array to mat
        mMat = Imgcodecs.imdecode(new MatOfByte(buf), Imgcodecs.IMREAD_UNCHANGED);
        return mMat;
    }

    /**
     * converts an android.media.Image to a Bitmap .
     * @param image : the image to convert.
     * @return : the resulting Bitmap.
     */
    public static Bitmap imageToBitmap(Image image)
    {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.capacity()];
        buffer.get(bytes);
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.length, null);
    }

    /**
     * converts an android.media.Image to a byte array .
     * @param image : the image to convert.
     * @return : the resulting byte array.
     */
    public static byte[] imageToBytes(Image image)
    {
        ByteBuffer buffer;
        int rowStride;
        int pixelStride;
        int width = image.getWidth();
        int height = image.getHeight();
        int offset = 0;

        Image.Plane[] planes = image.getPlanes();
        byte[] data = new byte[image.getWidth() * image.getHeight() * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8];
        byte[] rowData = new byte[planes[0].getRowStride()];

//        Log.i("coucou", "converting image to mat" );//
        for (int i = 0; i < planes.length; i++) {
            buffer = planes[i].getBuffer();
            rowStride = planes[i].getRowStride();
            pixelStride = planes[i].getPixelStride();
            int w = (i == 0) ? width : width / 2;
            int h = (i == 0) ? height : height / 2;
            for (int row = 0; row < h; row++) {
                int bytesPerPixel = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8;
                if (pixelStride == bytesPerPixel) {
                    int length = w * bytesPerPixel;
                    buffer.get(data, offset, length);

                    // Advance buffer the remainder of the row stride, unless on the last row.
                    // Otherwise, this will throw an IllegalArgumentException because the buffer
                    // doesn't include the last padding.
                    if (h - row != 1) {
                        buffer.position(buffer.position() + rowStride - length);
                    }
                    offset += length;
                } else {

                    // On the last row only read the width of the image minus the pixel stride
                    // plus one. Otherwise, this will throw a BufferUnderflowException because the
                    // buffer doesn't include the last padding.
                    if (h - row == 1) {
                        buffer.get(rowData, 0, width - pixelStride + 1);
                    } else {
                        buffer.get(rowData, 0, rowStride);
                    }

                    for (int col = 0; col < w; col++) {
                        data[offset++] = rowData[col * pixelStride];
                    }
                }
            }
        }

        return data;
    }


    /**
     * converts a cv Mat to a byte array in JPG format.
     * @param mat : the mat to convert.
     * @return : the resulting byte array.
     */
    public static byte[] matToBytes(Mat mat, boolean conv2RGB)
    {
        try {
            // convert color to RGB
            if(conv2RGB)
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);

            //Mat to bitmap
            Bitmap image = Bitmap.createBitmap(mat.cols(),
                    mat.rows(), Bitmap.Config.RGB_565);
            org.opencv.android.Utils.matToBitmap(mat, image);
            image = Bitmap.createScaledBitmap(image, mat.width(), mat.height(), false);
            //bitmap to byte array
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
            return byteArrayOutputStream.toByteArray();
        }
        catch (Exception e)
        {
            Log.w(TAG, "Error converting mat to byte " + Log.getStackTraceString(e));
            return new byte[mat.width()*mat.height()];
        }
    }

    /**
     * converts a cv Mat to a bitmap and resize.
     * @param mat : the mat to convert.
     * @param sizeX : resolution X
     * @param sizeY : resolution Y
     * @return : the resulting bitmap
     */
    public static  Bitmap matToBitmapAndResize(Mat mat, int sizeX, int sizeY)
    {
        Mat resizedFrame = new Mat();
        Imgproc.resize(mat, resizedFrame, new Size(sizeX,sizeY));
        Bitmap bitmapImagefull = Bitmap.createBitmap(resizedFrame.cols(), resizedFrame.rows(), Bitmap.Config.ARGB_8888);
        org.opencv.android.Utils.matToBitmap(resizedFrame, bitmapImagefull);
        return bitmapImagefull;
    }


    /**
     * from opencv forum : converts YUV image to a cv Mat.
     * @param image : the image to convert.
     * @return : the resulting Mat.
     */
    public static Mat convertYuv420888ToMat(Image image, boolean isGreyOnly) {
        int width = image.getWidth();
        int height = image.getHeight();

        Image.Plane yPlane = image.getPlanes()[0];
        int ySize = yPlane.getBuffer().remaining();

        if (isGreyOnly) {
            byte[] data = new byte[ySize];
            yPlane.getBuffer().get(data, 0, ySize);

            Mat greyMat = new Mat(height, width, CvType.CV_8UC1);
            greyMat.put(0, 0, data);

            return greyMat;
        }

        Image.Plane uPlane = image.getPlanes()[1];
        Image.Plane vPlane = image.getPlanes()[2];

        // be aware that this size does not include the padding at the end, if there is any
        // (e.g. if pixel stride is 2 the size is ySize / 2 - 1)
        int uSize = uPlane.getBuffer().remaining();
        int vSize = vPlane.getBuffer().remaining();

        byte[] data = new byte[ySize + (ySize / 2)];

        yPlane.getBuffer().get(data, 0, ySize);

        ByteBuffer ub = uPlane.getBuffer();
        ByteBuffer vb = vPlane.getBuffer();

        int uvPixelStride = uPlane.getPixelStride(); //stride guaranteed to be the same for u and v planes
        if (uvPixelStride == 1) {
            uPlane.getBuffer().get(data, ySize, uSize);
            vPlane.getBuffer().get(data, ySize + uSize, vSize);

            Mat yuvMat = new Mat(height + (height / 2), width, CvType.CV_8UC1);
            yuvMat.put(0, 0, data);
            Mat mrgbMat = new Mat(height, width, CV_8UC3);
            Imgproc.cvtColor(yuvMat, mrgbMat, Imgproc.COLOR_YUV2RGB_I420, 3);
            yuvMat.release();
            return mrgbMat;
        }

        // if pixel stride is 2 there is padding between each pixel
        // converting it to NV21 by filling the gaps of the v plane with the u values
        vb.get(data, ySize, vSize);
        for (int i = 0; i < uSize; i += 2) {
            data[ySize + i + 1] = ub.get(i);
        }

        Mat yuvMat = new Mat(height + (height / 2), width, CvType.CV_8UC1);
        yuvMat.put(0, 0, data);
        Mat rgbMat = new Mat(height, width, CV_8UC3);
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21, 3);
        yuvMat.release();
        return rgbMat;
    }


    /**
     * BFR implementation of 2D point
     */
    public static class bfr2DPoint
    {
        public double x;
        public double y;

        public bfr2DPoint(double x, double y)
        {
            this.x = x;
            this.y = y;
        }

        // Method used to display X and Y coordinates
        // of a point

    }

    /**
     * Compute the coords of the intersection between 2 lines, givent the coords of 4 points
     * @param a : coord of one the point.
     * @param b : coord of one the point.
     * @param c : coord of one the point.
     * @param d : coord of one the point.
     * @return : the point of intersection.
     */
    public static bfr2DPoint lineLineIntersection(bfr2DPoint a, bfr2DPoint b, bfr2DPoint c, bfr2DPoint d)
    {
        // Line AB represented as a1x + b1y = c1
        double a1 = b.y - a.y;
        double b1 = a.x - b.x;
        double c1 = a1*(a.x) + b1*(a.y);

        // Line CD represented as a2x + b2y = c2
        double a2 = d.y - c.y;
        double b2 = c.x - d.x;
        double c2 = a2*(c.x)+ b2*(c.y);

        double determinant = a1*b2 - a2*b1;

        if (determinant == 0)
        {
            // The lines are parallel. This is simplified
            // by returning a pair of FLT_MAX
            return new bfr2DPoint(Double.MAX_VALUE, Double.MAX_VALUE);
        }
        else
        {
            double x = (b2*c1 - b1*c2)/determinant;
            double y = (a1*c2 - a2*c1)/determinant;
            return new bfr2DPoint(x, y);
        }
    }

}
