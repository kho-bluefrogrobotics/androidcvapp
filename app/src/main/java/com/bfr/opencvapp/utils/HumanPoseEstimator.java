package com.bfr.opencvapp.utils;


import static com.bfr.opencvapp.GestureRecognition.IMG_HEIGHT;
import static com.bfr.opencvapp.utils.HumanPoseLandmarks.*;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
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
import java.util.List;


/** Hand Face Human object detector based on a Mobilenetv2-SSD network*/
public class HumanPoseEstimator {

    private final String TAG = "HumanPose";

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
//    final String MODEL_NAME = "hand_landmark_full.tflite";
//    final String MODEL_NAME = "hand_landmark_sparse.tflite";
    final String MODEL_NAME = "model_full_fp32.tflite";
    private final String MODELS_DIR = "/sdcard/Android/data/com.bfr.opencvapp/files/nnmodels/";

    private Interpreter tfLite;
    private HexagonDelegate hexagonDelegate;

    PoseLandmarker poseLandmarker;

    Context context;

    /** finger open or not*/
    enum FINGER{
        THUMB,
        INDEX,
        MIDDLE,
        RING,
        PINKIE
    }
    int[][] PHALANX_ID = new int[][]{{4,2}, {8,6}, {12,10}, {16,14}, {20, 18}};


    // confidence level of human detection for doublecheck with Movenet
    public float humanConfidence = 0.0f;

    public HumanPoseEstimator(Context context){

        try{
//            displayMat = new Mat();

            Interpreter.Options options = (new Interpreter.Options());
            CompatibilityList compatList = new CompatibilityList();

            options.setNumThreads(NUM_THREADS);

            if (WITH_GPU) {
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                delegateOptions.setQuantizedModelsAllowed(false);
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(gpuDelegate);
                Log.i(TAG, "Handpose Interpreter on GPU");
            }
            else{
                options.setUseXNNPACK(true);
                WITH_NNAPI = false;
                Log.i(TAG, "Handpose Interpreter on CPU");
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
//            tfLite = new Interpreter(tfliteModel, options );


            /** Mediapipe */

            BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder()
                    .setModelAssetPath("nnmodels/pose_landmarker_full.task")
                    .setDelegate(Delegate.GPU);
            BaseOptions baseOptions  = baseOptionsBuilder.build();

            PoseLandmarker.PoseLandmarkerOptions.Builder poseOptionsBuilder = PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMinPoseDetectionConfidence(0.5f)
                    .setMinTrackingConfidence(0.01f)
                    .setMinPosePresenceConfidence(0.01f)
                    .setNumPoses(1)
                    .setRunningMode(RunningMode.IMAGE);



            PoseLandmarker.PoseLandmarkerOptions poseOptions  = poseOptionsBuilder.build();

            poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions);

        }
        catch (Exception e)
        {
            Log.e(TAG, "Error Creating the Handpose " + Log.getStackTraceString(e) );
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
    public HumanPose recognizeImage(Mat frame) {

//        Log.i(TAG, "Starting Human pose estimation" );

        HumanPose humanPose = new HumanPose();

        //convert to bitmap
        Mat input = frame.clone();
//        Imgproc.resize(frame, resizedFrame, new Size(320,320));
        Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2BGR);
        Bitmap bitmapImagefull = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(input, bitmapImagefull);

//        try (FileOutputStream out = new FileOutputStream("/sdcard/Download/111.png")) {
//            bitmapImagefull.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
//            // PNG is a lossless format, the compression factor (100) is ignored
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        Imgcodecs.imwrite("/sdcard/Download/111.jpg", input);
        MPImage mpImage = new BitmapImageBuilder(bitmapImagefull).build() ;

        PoseLandmarkerResult poseLandmarkerResult =poseLandmarker.detect(mpImage);

        if (poseLandmarkerResult.landmarks().size()<=0)
        {
//            Log.w(TAG, "NO Human Pose DETECTED");
            return null;
        }


//        Log.i(TAG, "Result size ="+ poseLandmarkerResult.landmarks().get(0).size());

        humanPose.landmarks = poseLandmarkerResult.landmarks().get(0);

//        for(int k=0; k<20; k++)
//        {
//            int x = (int) (handLandmarkerResult.landmarks().get(0).get(k).x() * frame.cols());
//            int y = (int) (handLandmarkerResult.landmarks().get(0).get(k).y()* frame.rows());
//            Imgproc.circle(frame, new Point(x,y), 5, new Scalar(0,255,0), 5);
//        }



//
//        try
//        {
//            displayMat = frame.clone();
//
//            // check input size
//            Mat resizedFrame = new Mat();
//            if(frame.rows()!=INPUT_SIZE.height || frame.cols()!=INPUT_SIZE.width)
//                Imgproc.resize(frame, resizedFrame, new Size(INPUT_SIZE.width,INPUT_SIZE.height));
//            else
//                resizedFrame = frame.clone();
//
//            //convert to bitmap
//            Bitmap bitmapImg = Bitmap.createBitmap(resizedFrame.cols(), resizedFrame.rows(), Bitmap.Config.ARGB_8888);
//            Utils.matToBitmap(resizedFrame, bitmapImg);
//            // assigning memory of input
//            ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmapImg);
//            Object[] inputArray = {byteBuffer};
//
//            // assigning output
//            Map<Integer, Object> outputMap = new HashMap<>();
//
//            //
//            // 1) a fp32{1,63} map of the 21 landmarks * (x, y, z) coords in PIXEL , z takes the origin at the wrist
////            outputMap.put(0, new float[1][63]);
//            outputMap.put(0, new float[1][1]);
//            // 2) a fp32{1,1} map representing the probability of presence of a hand
////            outputMap.put(1, new float[1][1]);
//            outputMap.put(1, new float[1][63]);
//            // 3) a fp32{1,1} map representing the handedness  <0.5: Left hand , >0.5:Right hand
////            outputMap.put(2, new float[1][1]);
//            outputMap.put(2, new float[1][1]);
//            // 4) a fp32{1,63} map of the 21 landmarks * (x, y, z) coords in world coordinates
//            outputMap.put(3, new float[1][63]);
//
////            Log.d(TAG, "Inference NOW!");
//            // Run inference
//            tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
//
////            humanPose.landmarks = ((float[][]) outputMap.get(0))[0];
//            humanPose.landmarks = ((float[][]) outputMap.get(3))[0];
////            humanPose.handPresence = ((float [][]) Objects.requireNonNull(outputMap.get(1)))[0][0];
//            humanPose.handPresence = ((float [][]) Objects.requireNonNull(outputMap.get(0)))[0][0];
////            humanPose.handeness = ((float [][]) Objects.requireNonNull(outputMap.get(2)))[0][0];
//            humanPose.handeness = ((float [][]) Objects.requireNonNull(outputMap.get(2)))[0][0];
//
////            Log.d(TAG, "Inference done; confidence = " +  humanPose.handPresence + " LorR="+ humanPose.handeness);
//            Log.d(TAG, "tip index Point = " + humanPose.landmarks[8*3] + ","+ humanPose.landmarks[8*3 + 1]);
//
//            //init for display only
//            objId = 0;
//            // for each detection
//            for (int i = 0; i < OUTPUT_MAPS_SIZE[0]; i++)
//            {
//
//            } // next detection


//        } catch (Exception e) {
//            Log.e("ERROR", Log.getStackTraceString(e));
//        }

        return humanPose;
    }


    /**
    class returned by pose estimation
     containing
     - 63 landmarks *3 coords [x, y, z]; where x, y in PIXEL from the upper left corener of the input image, and z respective to the wrist
     - probability of hand presence
     - handedness: <0.5=left hand, >0.5 right hand

     */
    public class HumanPose{

        public float handPresence= 0.0f;
        public List<Category> handeness = new ArrayList<>();
        public List<NormalizedLandmark> landmarks = null;

        private boolean front = false;


        public int x(int landmark)
        {
            return (int)(landmarks.get(landmark).x()*1024);
        }
        public int y(int landmark)
        {
            return (int)(landmarks.get(landmark).y()*768);
        }
        public Mat display(Mat frame)
        {
            Mat displayMat=frame.clone();

//            Imgproc.circle(displayMat, new Point((int)(landmarks.get(LEFT_EYE).x()*frame.cols()), (int)(landmarks.get(LEFT_EYE).y()*frame.rows())), 10, new Scalar(0,255, 0),5);
            Imgproc.circle(displayMat, new Point(x(LEFT_EYE), y(LEFT_EYE)), 7, new Scalar(0,255, 0),5);
            Imgproc.circle(displayMat, new Point(x(RIGHT_EYE), y(RIGHT_EYE)), 7, new Scalar(0,255, 0),5);

            if(landmarks.get(LEFT_WRIST).visibility().get()>0.6f)
            {
                Imgproc.circle(displayMat, new Point(x(LEFT_WRIST), y(LEFT_WRIST)), 7, new Scalar(0,0, 255),5);
                Imgproc.circle(displayMat, new Point(x(LEFT_THUMB), y(LEFT_THUMB)), 7, new Scalar(0,0, 255),5);
                Imgproc.circle(displayMat, new Point(x(LEFT_INDEX), y(LEFT_INDEX)), 7, new Scalar(0,0, 255),5);
            }

            if(landmarks.get(RIGHT_WRIST).visibility().get()>0.6f) {
                Imgproc.circle(displayMat, new Point(x(RIGHT_WRIST), y(RIGHT_WRIST)), 7, new Scalar(255, 0, 0), 5);
                Imgproc.circle(displayMat, new Point(x(RIGHT_THUMB), y(RIGHT_THUMB)), 7, new Scalar(255, 0, 0), 5);
                Imgproc.circle(displayMat, new Point(x(RIGHT_INDEX), y(RIGHT_INDEX)), 7, new Scalar(255, 0, 0), 5);
            }
            return displayMat;
        }


        final float WRIST_VISIBILITY_THRES = 0.7f;

        public int isSigning()
        {
            if( landmarks.get(LEFT_WRIST).visibility().get() > WRIST_VISIBILITY_THRES && (
                            landmarks.get(LEFT_WRIST).y()<landmarks.get(LEFT_ELBOW).y()-0.05  || landmarks.get(LEFT_ELBOW).y()*IMG_HEIGHT<landmarks.get(LEFT_SHOULDER).y()*IMG_HEIGHT+15
            ))
            {
                return LEFT_WRIST;
            }
            else if(landmarks.get(RIGHT_WRIST).visibility().get() > WRIST_VISIBILITY_THRES && (
                    landmarks.get(RIGHT_WRIST).y()<landmarks.get(RIGHT_ELBOW).y()-0.05)  || landmarks.get(RIGHT_ELBOW).y()*IMG_HEIGHT<landmarks.get(RIGHT_SHOULDER).y()*IMG_HEIGHT+15 )
            {
                return RIGHT_WRIST;
            }
            else
                return -1;
        } //end isSigning


    } //end headpose class

}
