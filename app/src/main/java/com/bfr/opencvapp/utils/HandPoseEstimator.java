package com.bfr.opencvapp.utils;


import static com.bfr.opencvapp.utils.Utils.Color.*;
import static com.bfr.opencvapp.utils.Utils.Color._BLACK;
import static com.bfr.opencvapp.utils.Utils.Color._BLUE;
import static com.bfr.opencvapp.utils.Utils.Color._GREEN;
import static com.bfr.opencvapp.utils.Utils.Color._RED;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;



import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/** Hand Face Human object detector based on a Mobilenetv2-SSD network*/
public class HandPoseEstimator {

    private final String TAG = "HandPose";

    //Params for TFlite interpreter
    private final boolean IS_QUANTIZED = false;
    private final Size INPUT_SIZE = new Size(224,224);
    // SSD outputs 4 maps corresponding to 50 object detections
    // 1) a fp32{1,50} map of the 50 scores of confidence
    // 2) a fp32{1,50,4} map of 50x4 values for xmin, ymin, xmax, ymax in [0;1]
    // 3) not used
    // 4) a fp32{1,50} map of the 50 labels of the detected class
    private final int[] OUTPUT_MAPS_SIZE = new int[]{50, 50, 1, 50};

    private final int BATCH_SIZE = 1;
    private final int PIXEL_SIZE = 3;
    private final String[] LABELS = {"Human", "Face", "Hand"};
    private final int NUM_THREADS =4;
    private boolean WITH_NNAPI = false;
    private boolean WITH_GPU = true;
    private boolean WITH_DSP = false;

    // for display
    public Mat displayMat;
    public boolean readyToDisplay=false;
    Point pt1 = new Point();
    Point pt2 = new Point();

    private int objId = 0;

    //where to find the models
//    final String MODEL_NAME = "hand_landmark_lite.tflite";
    final String MODEL_NAME = "hand_landmark_full.tflite";
    private final String MODELS_DIR = "/sdcard/Android/data/com.bfr.opencvapp/files/nnmodels/";

    private Interpreter tfLite;
    private HexagonDelegate hexagonDelegate;



    // confidence level of human detection for doublecheck with Movenet
    public float humanConfidence = 0.0f;

    public HandPoseEstimator(Context context){

        try{
            displayMat = new Mat();

            Interpreter.Options options = (new Interpreter.Options());
            CompatibilityList compatList = new CompatibilityList();

            options.setNumThreads(NUM_THREADS);

            if (WITH_GPU) {
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                delegateOptions.setQuantizedModelsAllowed(false);
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(gpuDelegate);
                Log.i(TAG, "Multidetector Interpreter on GPU");
            }
            else if (WITH_DSP){
                hexagonDelegate = new HexagonDelegate(context);
                options.addDelegate(hexagonDelegate);
                Log.i(TAG, "Multidetector Interpreter on HEXAGONE");
            }
            else{
                options.setUseXNNPACK(true);
                WITH_NNAPI = false;
                Log.i(TAG, "Multidetector Interpreter on CPU");
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
            File tfliteModel = new File(MODELS_DIR +MODEL_NAME);
            tfLite = new Interpreter(tfliteModel, options );

        }
        catch (Exception e)
        {
            Log.e(TAG, "Error Creating the MultiDetector " + Log.getStackTraceString(e) );
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
            byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * (int)INPUT_SIZE.height * (int)INPUT_SIZE.width * PIXEL_SIZE);
        }
        else{
            byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * (int)INPUT_SIZE.height * (int)INPUT_SIZE.width * PIXEL_SIZE);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[(int)INPUT_SIZE.height * (int)INPUT_SIZE.width];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < (int)INPUT_SIZE.width; ++i) {
            for (int j = 0; j < (int)INPUT_SIZE.height; ++j) {
                final int val = intValues[pixel++];
                if (IS_QUANTIZED) {
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {

                    byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                    byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                    byteBuffer.putFloat((val & 0xFF) / 255.0f);
                }
            }
        }
        return byteBuffer;
    }

    /**
     * get the detected objects in the image
     * @param frame original image in Mat format
     * @return array of detections
     */
    public float[][] recognizeImage(Mat frame) {

        Log.i(TAG, "Starting Hand pose estimation" );

        float[][] landmarks = null;

        boolean isReallyHuman = true;

        try
        {
            displayMat = frame.clone();

            // check input size
            Mat resizedFrame = new Mat();
            if(frame.rows()!=INPUT_SIZE.height || frame.cols()!=INPUT_SIZE.width)
                Imgproc.resize(frame, resizedFrame, new Size(INPUT_SIZE.width,INPUT_SIZE.height));
            else
                resizedFrame = frame.clone();

            //convert to bitmap
            Bitmap bitmapImg = Bitmap.createBitmap(resizedFrame.cols(), resizedFrame.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(resizedFrame, bitmapImg);
            // assigning memory of input
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmapImg);
            Object[] inputArray = {byteBuffer};

            // assigning output
            Map<Integer, Object> outputMap = new HashMap<>();

            //
            // 1) a fp32{1,63} map of the 21 landmarks * (x, y, z) coords in PIXEL , z takes the origin at the wrist
            outputMap.put(0, new float[1][63]);
            // 2) a fp32{1,1} map representing the probability of presence of a hand
            outputMap.put(1, new float[1][1]);
            // 3) a fp32{1,1} map representing the handedness  <0.5: Left hand , >0.5:Right hand
            outputMap.put(2, new float[1][1]);
            // 4) a fp32{1,63} map of the 21 landmarks * (x, y, z) coords in world coordinates
            outputMap.put(3, new float[1][63]);


            Log.d(TAG, "Inference NOW!");
            // Run inference
            tfLite.runForMultipleInputsOutputs(inputArray, outputMap);


            //explicit names for better readibility of output
            float[][]  handPresence= (float [][]) outputMap.get(1);
            float[][] handeness = (float[][]) outputMap.get(2);
            landmarks = (float[][]) outputMap.get(0);

            Log.d(TAG, "Inference done; confidence = " + handPresence[0][0] + " LorR="+ handeness[0][0]);
            Log.d(TAG, "tip index Point = " + landmarks[0][8*3] + ","+ landmarks[0][8*3 + 1]);

            //init for display only
            objId = 0;
            // for each detection
            for (int i = 0; i < OUTPUT_MAPS_SIZE[0]; i++)
            {

            } // next detection


        } catch (Exception e) {
            e.printStackTrace();
        }

        return landmarks;
    }




}
