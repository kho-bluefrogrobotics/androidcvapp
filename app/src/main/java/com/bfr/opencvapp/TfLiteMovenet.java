package com.bfr.opencvapp;

import static com.bfr.opencvapp.utils.Utils.modelsDir;
import static org.opencv.core.CvType.CV_8UC3;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

/** Movenet tflite implementation from official models https://www.kaggle.com/models/google/movenet
 * The models output a map of 17,3 =  17 keypoints x (x, y, score) , with x, y in the range [0;1]
 * The order of the 17 keypoint joints is:
 * [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]
 * */
public class TfLiteMovenet {

    private final String TAG = "TfLiteMovenet";

    //Params for TFlite interpreter
    private final boolean IS_QUANTIZED = false;
    private final int[] INPUT_SIZE = {256,256};
    private final int[] OUTPUT_SIZE = {17,3};
    private final int BATCH_SIZE = 1;
    private final int PIXEL_SIZE = 3;
    private final int NUM_THREADS = 4;
    private boolean WITH_NNAPI = true;
    private boolean WITH_GPU = true;
    private boolean WITH_DSP = false;
    //Face embedding
    private float[][][][] embeedings;

    //model file
    private final String MODEL_NAME = "Movenet_singlepose_thunder.tflite";

    private Interpreter tfLite;
    private HexagonDelegate hexagonDelegate;

    public TfLiteMovenet(Context context){

        try{
            Interpreter.Options options = (new Interpreter.Options());
            CompatibilityList compatList = new CompatibilityList();

            options.setNumThreads(NUM_THREADS);

            if (WITH_GPU) {
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                delegateOptions.setQuantizedModelsAllowed(false);
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(gpuDelegate);
                Log.i(TAG, "Interpreter on GPU");
            }
            else if (WITH_DSP){
                hexagonDelegate = new HexagonDelegate(context);
                options.addDelegate(hexagonDelegate);
                Log.i(TAG, "Interpreter on HEXAGONE");
            }
            else{
                options.setUseXNNPACK(true);
                WITH_NNAPI = false;
                Log.i(TAG, "Interpreter on CPU");
            }

            if (WITH_NNAPI) {
                NnApiDelegate nnApiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    nnApiDelegate = new NnApiDelegate();
                    options.addDelegate(nnApiDelegate);
                    options.setUseNNAPI(true);
                }
            }

            //Init interpreter
            File tfliteModel = new File(modelsDir+MODEL_NAME);
            tfLite = new Interpreter(tfliteModel, options );
        }
        catch (Exception e)
        {
            Log.e(TAG, "Error Creating the tflite model " + Log.getStackTraceString(e) );
        }

    }

    /**
     * Converts a Bitmap into a BytBuffer
     * @param bitmap original bitmap
     * @return ByteBuffer
     */
    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        if (IS_QUANTIZED) {
            byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * INPUT_SIZE[0] * INPUT_SIZE[1] * PIXEL_SIZE);
        }
        else{
            byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE[0] * INPUT_SIZE[1] * PIXEL_SIZE);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE[0] * INPUT_SIZE[1]];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE[0]; ++i) {
            for (int j = 0; j < INPUT_SIZE[1]; ++j) {
                final int val = intValues[pixel++];
                if (IS_QUANTIZED) {
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {

                    byteBuffer.putFloat(((val >> 16) & 0xFF) / 1.0f);
                    byteBuffer.putFloat(((val >> 8) & 0xFF) / 1.0f);
                    byteBuffer.putFloat((val & 0xFF) / 1.0f);
                }
            }
        }
        return byteBuffer;
    }


    /**
     * get the detected objects in the image
     * @param img original image
     * @return array of detections
     */
    public float[][][][] recognizeImage(Mat img) {

        Mat resizedWithScale = new Mat();

        // if input resolution NOK
        if(img.rows()!=256 && img.cols()!=256)
        {
            // resizing for Movenet model, with padding to keep ratio
            resizedWithScale = resizeWithPadding(img, 256, 256);
        }
        else
        {
            resizedWithScale = img.clone();
        }

        // convert to bitmap
        Bitmap bitmapImage = Bitmap.createBitmap(resizedWithScale.cols(), resizedWithScale.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(resizedWithScale, bitmapImage);
        //get buffer
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmapImage);

        Map<Integer, Object> outputMap = new HashMap<>();

        // Init Face embeedings (signature)
        embeedings = new float[1][1][OUTPUT_SIZE[0]][OUTPUT_SIZE[1]];

        outputMap.put(0, embeedings);

        Object[] inputArray = {byteBuffer};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        return embeedings;
    }


    /**
     * resize and add padding to keep scale
     * @param input the image to resize
     * @param desiredWidth the desired with for the output resized image
     * @param desiredHeight the desired height for the output resized image
     * @return a resized image
     */
    private Mat resizeWithPadding(Mat input, int desiredWidth, int desiredHeight)
    {

        int originalHeight = input.height();
        int originalWidth = input.width();

        // image with original size to be padded to keep ratio of resizing
        Mat paddedImage = input.clone();

        if ((float)originalHeight/(float)originalWidth > (float)desiredHeight/(float)desiredWidth) // if height of orig image is too large (=>need horizontal padding)
        {
            // width which respects required ratio
            int targetWidth =(int) ((float)originalHeight * (float)desiredWidth/(float)desiredHeight);

            paddedImage = new Mat( originalHeight,targetWidth, CV_8UC3, new Scalar(0, 0, 0));
            Rect ROI= new Rect(
                    0,
                    0,
                    input.cols(),
                    input.rows() );
            Mat roiInBlackMat = paddedImage.submat(ROI);
            input.copyTo(roiInBlackMat);

        }
        else if((float)originalHeight/(float)originalWidth < (float)desiredHeight/(float)desiredWidth)
        // if width of orig image is too large (=>need vertical padding)
        {
            // Height which respects required ratio
            int targetHeight =(int) ((float)originalWidth * (float)desiredHeight/(float)desiredWidth );

            paddedImage = new Mat( targetHeight, originalWidth, CV_8UC3, new Scalar(0, 0, 0));
            Rect ROI= new Rect(
                    0,
                    0,
                    input.cols(),
                    input.rows() );
            Mat roiInBlackMat = paddedImage.submat(ROI);
            input.copyTo(roiInBlackMat);

        }
        else // ratio is already correct
        {
            //do nothing, the image already has the right ratio
        }

        //finally resize to required size
        Mat resizedMat = new Mat();
        Imgproc.resize(paddedImage, resizedMat, new Size(desiredWidth, desiredHeight));


        return resizedMat;
    }

}
