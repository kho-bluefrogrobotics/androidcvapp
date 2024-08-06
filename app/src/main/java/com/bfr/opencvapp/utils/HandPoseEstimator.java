package com.bfr.opencvapp.utils;


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
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;


/** Hand Face Human object detector based on a Mobilenetv2-SSD network*/
public class HandPoseEstimator {

    private final String TAG = "Gesture HandPose";

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

    HandLandmarker handLandmarker;

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

    public HandPoseEstimator(Context context){

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
                    .setModelAssetPath("nnmodels/hand_landmarker.task")
                    .setDelegate(Delegate.GPU);
            BaseOptions baseOptions  = baseOptionsBuilder.build();

            HandLandmarker.HandLandmarkerOptions.Builder handOptionsBuilder = HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setNumHands(1)
                    .setMinHandDetectionConfidence(0.5f)
                    .setMinTrackingConfidence(0.01f)
                    .setMinHandPresenceConfidence(0.01f)
                    .setRunningMode(RunningMode.IMAGE);



            HandLandmarker.HandLandmarkerOptions handOptions  = handOptionsBuilder.build();

            handLandmarker = HandLandmarker.createFromOptions(context, handOptions);

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
    public HandPose recognizeImage(Mat frame) {

        Log.i(TAG, "Starting Hand pose estimation" );

        HandPose handPose = new HandPose();

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

        HandLandmarkerResult handLandmarkerResult =handLandmarker.detect(mpImage);

        if (handLandmarkerResult.landmarks().size()<=0)
        {
            Log.w(TAG, "NO HAND DETECTED");
            return null;
        }


        Log.i(TAG, "Result size ="+ handLandmarkerResult.landmarks().get(0).size());

        handPose.landmarks = handLandmarkerResult.landmarks().get(0);
        handPose.handeness = handLandmarkerResult.handednesses().get(0);

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
////            handPose.landmarks = ((float[][]) outputMap.get(0))[0];
//            handPose.landmarks = ((float[][]) outputMap.get(3))[0];
////            handPose.handPresence = ((float [][]) Objects.requireNonNull(outputMap.get(1)))[0][0];
//            handPose.handPresence = ((float [][]) Objects.requireNonNull(outputMap.get(0)))[0][0];
////            handPose.handeness = ((float [][]) Objects.requireNonNull(outputMap.get(2)))[0][0];
//            handPose.handeness = ((float [][]) Objects.requireNonNull(outputMap.get(2)))[0][0];
//
////            Log.d(TAG, "Inference done; confidence = " +  handPose.handPresence + " LorR="+ handPose.handeness);
//            Log.d(TAG, "tip index Point = " + handPose.landmarks[8*3] + ","+ handPose.landmarks[8*3 + 1]);
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

        return handPose;
    }


    /**
    class returned by pose estimation
     containing
     - 63 landmarks *3 coords [x, y, z]; where x, y in PIXEL from the upper left corener of the input image, and z respective to the wrist
     - probability of hand presence
     - handedness: <0.5=left hand, >0.5 right hand

     */
    public class HandPose{

        public float handPresence= 0.0f;
        public List<Category> handeness = new ArrayList<>();
        public List<NormalizedLandmark> landmarks = null;

        private boolean front = false;

        // front: true = palm towards the camera, false= back of the hand towards the camera
        public void front()
        {
            // vector of the knucle line, from the base of the pinkie to the base of the index
            int[] knucleLine = new int[]{ (int)( (landmarks.get(17).x() - landmarks.get(5).x())  * 1024), (int)((landmarks.get(17).y() - landmarks.get(5).y())*768 )};
            // vector of the palm, from the wrist to the base of the index
            int[] plamLine = new int[]{ (int)( (landmarks.get(0).x() - landmarks.get(5).x()) *1024), (int)( (landmarks.get(0).y() - landmarks.get(5).y())*768 )};

            Log.w("vecto", "Result =" +  knucleLine[0]+"x"+plamLine[1] +"-"+ plamLine[0]+"x"+knucleLine[1] +"=" +(knucleLine[0]*plamLine[1]-plamLine[0]*knucleLine[1]));

            //z component of the cross product, not normalized
            int z=(knucleLine[0]*plamLine[1]-plamLine[0]*knucleLine[1]);

            // if left hand
            if (handeness.get(0).categoryName().toUpperCase().contains("LEFT")){
                if(z<0)
                    Log.w("sideH", "LEFT FRONT");
                else
                    Log.w("sideH", "LEFT BACK");
            }
            else{
                if(z>0)
                    Log.w("sideH", "RIGHT FRONT");
                else
                    Log.w("sideH", "RIGHT BACK");
            }

        }

        // front: true = palm towards the camera, false= back of the hand towards the camera
        public boolean isFront()
        {
            // vector of the knucle line, from the base of the index to the base of the pinkie
            int[] knucleLine = new int[]{ (int)( (landmarks.get(17).x() - landmarks.get(5).x())  * 1024), (int)((landmarks.get(17).y() - landmarks.get(5).y())*768 ), (int)((landmarks.get(17).z() - landmarks.get(5).z())*100 ) };
            // vector of the palm, from the wrist to the base of the index
            int[] plamLine = new int[]{ (int)( (landmarks.get(0).x() - landmarks.get(5).x()) *1024), (int)( (landmarks.get(0).y() - landmarks.get(5).y())*768 ), (int)( (landmarks.get(0).z() - landmarks.get(5).z())*100 )};

//            Log.w("vecto", "Result =" +  knucleLine[0]+"x"+plamLine[1] +"-"+ plamLine[0]+"x"+knucleLine[1] +"=" +(knucleLine[0]*plamLine[1]-plamLine[0]*knucleLine[1]));


            //the crosproduct represent the orthogonal vector to the knucleline and the vector index-wrist
            //https://en.wikipedia.org/wiki/Cross_product
            int x = (knucleLine[1]*plamLine[2]-knucleLine[2]*plamLine[1]);
            int y = (knucleLine[2]*plamLine[0]-knucleLine[0]*plamLine[2]);
            int z = (knucleLine[0]*plamLine[1]-knucleLine[1]*plamLine[0]);

            double norm = Math.sqrt(x*x+y*y+z*z);
            double normalizedZ = (double)z/norm;

            // angle from spherical coords
            //https://en.wikipedia.org/wiki/Spherical_coordinate_system#Modified_spherical_coordinates

            int signY =1;
            if(y>0)
                signY =1;
            else
                signY =-1;
            double rho = Math.acos(z/Math.sqrt(x*x+y*y+z*z));
            double phi = signY *Math.acos(x/Math.sqrt(x*x+y*y));
            Log.d(TAG, "IsFront: hand=" + handeness.get(0).categoryName() + " z=" + z
            +"\n"+ phi + "   " + rho);

            // if left hand
            if (handeness.get(0).categoryName().toUpperCase().contains("LEFT")){
//                if(z<-7000){
                if(rho>=3.1){
                    Log.w("sideH", "FRONT left z="+z);
                    this.front = true;
                }
                else{
                    Log.w("sideH", "BACK left z="+z);
                    this.front = false;
                }
            }
            //else Right hand
            else{
//                if(z>7000){
                if(rho<0.04){
                    Log.w("sideH", "FRONT right z="+z);
                    this.front = true;
                }
                else
                {
                    Log.w("sideH", "BACK right z="+z);
                    this.front = false;
                }
            }
            return this.front;
        }

        /** How to know a finger is opened : compute the hypotenuse  of the tip and 2nd phalanx
         * if the dist [tip of the finger to the wrist] < the dist [2nd phalanx to the wrist]  => the finger is open
         * https://github.com/opencv/opencv_zoo/blob/main/models/handpose_estimation_mediapipe/demo.py#L209
         * for a point (x1, y1) the dist is simply  = sqrt(x1^2 + y1^2)
         * for instance, the first finger tip has the id 8 , and the 2nd phalanx id 6
         * https://github.com/opencv/opencv_zoo/blob/main/models/handpose_estimation_mediapipe/demo.py#L205
         */
        public boolean isOpen(FINGER finger)
        {

            // For all fingers EXCEPT thumb
            if (finger != FINGER.THUMB)
            {
                int TIP = PHALANX_ID[finger.ordinal()][0];
                int SECOND_PHALANX = PHALANX_ID[finger.ordinal()][1];

//            int[] vec1 = new int[]{(int)(landmarks[TIP *3] -  landmarks[SECOND_PHALANX *3]), (int)(landmarks[TIP *3 +1] -  landmarks[SECOND_PHALANX *3+1]) };
//            Log.d("ccoucou", "vect=" + vec1[0] + "," + vec1[1] +"    " + landmarks[TIP *3] + "," + landmarks[TIP *3+1] );

                double distTip = Math.sqrt( (landmarks.get(TIP).x() -  landmarks.get(0).x())*(landmarks.get(TIP).x() -  landmarks.get(0).x())
                        + (landmarks.get(TIP).y() -  landmarks.get(0).y())*(landmarks.get(TIP).y() -  landmarks.get(0).y()) );

                double distPhalanx = Math.sqrt( (landmarks.get(SECOND_PHALANX).x() -  landmarks.get(0).x())*(landmarks.get(SECOND_PHALANX).x() -  landmarks.get(0).x())
                        + (landmarks.get(SECOND_PHALANX).y() -  landmarks.get(0).y())*(landmarks.get(SECOND_PHALANX).y() -  landmarks.get(0).y()) );

//            Log.d("ccoucou", "distTip=" + distTip );
//            Log.d("ccoucou", "distPhalanx=" + distPhalanx );
//            Log.d("ccoucou", "Interm Calc=" + (landmarks[TIP *3] -  landmarks[0]) + " + " + (landmarks[TIP*3 + 1] -  landmarks[1]) );

                if (distTip <= distPhalanx)
                    return  false;
                else
                    return true;
            }
            else // THUMB is an exception :
            // the open state of the thumb is obtained by comparing the dist of the tip to the base of the index finger the dist of the first two knuckles
            {
                double THRES_DIST_TIP_PHALANX = 0.1;
                int TIP = 4;
                int INDEX_BASE = 5;
                int MIDDLE_BASE = 9;

//            int[] vec1 = new int[]{(int)(landmarks[TIP *3] -  landmarks[SECOND_PHALANX *3]), (int)(landmarks[TIP *3 +1] -  landmarks[SECOND_PHALANX *3+1]) };
//            Log.d("ccoucou", "vect=" + vec1[0] + "," + vec1[1] +"    " + landmarks[TIP *3] + "," + landmarks[TIP *3+1] );

                double distTip = Math.sqrt( (landmarks.get(TIP).x() -  landmarks.get(INDEX_BASE).x())*(landmarks.get(TIP).x() -  landmarks.get(INDEX_BASE).x())
                        + (landmarks.get(TIP).y() -  landmarks.get(INDEX_BASE).y())*(landmarks.get(TIP).y() -  landmarks.get(INDEX_BASE).y()) );
                double distKnuckle = Math.sqrt( (landmarks.get(MIDDLE_BASE).x() -  landmarks.get(INDEX_BASE).x())*(landmarks.get(MIDDLE_BASE).x() -  landmarks.get(INDEX_BASE).x())
                        + (landmarks.get(MIDDLE_BASE).y() -  landmarks.get(INDEX_BASE).y())*(landmarks.get(MIDDLE_BASE).y() -  landmarks.get(INDEX_BASE).y()) );


//            Log.d("ccoucou", "distPhalanx=" + distPhalanx );
//            Log.d("ccoucou", "Interm Calc=" + (landmarks[TIP *3] -  landmarks[0]) + " + " + (landmarks[TIP*3 + 1] -  landmarks[1]) );

                // if tip of the thumb is close to the base of the middle finger
                if (distTip <= 2*distKnuckle)
                {
                    String.format("%1$,.2f", distTip);
                    Log.d("ccoucou", "distTip=" + String.format("%1$,.4f", distTip) + "distPhalanx=" + String.format("%1$,.4f", distKnuckle)  + "=> CLOSE");
                    return  false;
                }

                else
                {
                    Log.d("ccoucou", "distTip=" + String.format("%1$,.4f", distTip) + "distPhalanx=" + String.format("%1$,.4f", distKnuckle)  + "=> OPEN");
                    return true;
                }

            }

        } //end isOpen


        /**
         * returns the orientation of the finger as an angle in degrees [0-359]. 0 is horizontal, in anti-clockwise direction (so 90° si upward and -90° if downward)
         * @param finger
         * @return
         */
        public int fingerOrientation(FINGER finger)
        {
//            Log.d("ccoucou", "finger orientation");
            int TIP = PHALANX_ID[finger.ordinal()][0];
            int SECOND_PHALANX = PHALANX_ID[finger.ordinal()][1];

//            int[] vec1 = new int[]{(int)(landmarks[TIP *3] -  landmarks[SECOND_PHALANX *3]), (int)(landmarks[TIP *3 +1] -  landmarks[SECOND_PHALANX *3+1]) };
//            int[] vec2 = new int[]{1,0 };


            // dot product = x1*x2 + y1*y2
//            double dotProduct = (landmarks[TIP *3] -  landmarks[SECOND_PHALANX *3])* (landmarks[SECOND_PHALANX *3] -  landmarks[0]);
            double angleRad= Math.atan2( (landmarks.get(TIP).y() -  landmarks.get(SECOND_PHALANX).y()) ,  (landmarks.get(TIP).x() -  landmarks.get(SECOND_PHALANX).x()) );

//            double norm = Math.sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1]);
//            double dotProduct = ( vec1[0] * vec2[0]  + vec1[1]* vec2[1] )/ norm;


//            double angleRad = Math.atan2(vec1[1], vec1[0]);

            // image is oriented with y towards bottom -> invert sign
            return -(int)Math.toDegrees(angleRad);

//            Log.d("ccoucou", "TIP=" + (int)landmarks[TIP *3] + "," + (int)landmarks[TIP *3+1] + " PHALANX= " + (int)landmarks[SECOND_PHALANX *3] + "," + (int)landmarks[SECOND_PHALANX *3+1]);
//            Log.d("ccoucou", "vect1=" + vec1[0] + "," + vec1[1] + " dotproduct= " + dotProduct + "norm=" + norm + " ==>angle in rad = " + angleRad + " in deg = " + this.angle);

        } //end finger orientation


        /**
         * returns the orientation hand as an angle in degrees [0-359]. 0 is horizontal, in anti-clockwise direction (so 90° si upward and -90° if downward)
         * @return the hand rotation
         */
        public int handOrientation()
        {
//            Log.d("ccoucou", "finger orientation");
            int TIP = 17;
            int SECOND_PHALANX = 5;

//            int[] vec1 = new int[]{(int)(landmarks[TIP *3] -  landmarks[SECOND_PHALANX *3]), (int)(landmarks[TIP *3 +1] -  landmarks[SECOND_PHALANX *3+1]) };
//            int[] vec2 = new int[]{1,0 };


            // dot product = x1*x2 + y1*y2
//            double dotProduct = (landmarks[TIP *3] -  landmarks[SECOND_PHALANX *3])* (landmarks[SECOND_PHALANX *3] -  landmarks[0]);
//            double angleRad= Math.atan2( landmarks.get(TIP).y() -  landmarks.get(SECOND_PHALANX).y() ,  landmarks.get(TIP).x() -  landmarks.get(SECOND_PHALANX).x() );
            double angleRad= Math.atan2( Math.abs(landmarks.get(TIP).y() -  landmarks.get(SECOND_PHALANX).y() ) , Math.abs( landmarks.get(TIP).x() -  landmarks.get(SECOND_PHALANX).x() ) );

//            double norm = Math.sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1]);
//            double dotProduct = ( vec1[0] * vec2[0]  + vec1[1]* vec2[1] )/ norm;


//            double angleRad = Math.atan2(vec1[1], vec1[0]);

            // image is oriented with y towards bottom -> invert sign
            Log.d("ccoucou", "HandOrientation=" + ( ((int)Math.toDegrees(angleRad) ) %360) );
//

            return (int)Math.toDegrees(angleRad);

//            Log.d("ccoucou", "TIP=" + (int)landmarks[TIP *3] + "," + (int)landmarks[TIP *3+1] + " PHALANX= " + (int)landmarks[SECOND_PHALANX *3] + "," + (int)landmarks[SECOND_PHALANX *3+1]);
//            Log.d("ccoucou", "vect1=" + vec1[0] + "," + vec1[1] + " dotproduct= " + dotProduct + "norm=" + norm + " ==>angle in rad = " + angleRad + " in deg = " + this.angle);

        } //end finger orientation

    } //end headpose class

}
