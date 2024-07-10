package com.bfr.opencvapp.utils;

import static com.bfr.opencvapp.utils.Utils.Color._BLACK;
import static com.bfr.opencvapp.utils.Utils.Color._BLUE;
import static com.bfr.opencvapp.utils.Utils.Color._GREEN;
import static com.bfr.opencvapp.utils.Utils.Color._RED;
import static com.bfr.opencvapp.utils.Utils.MODELS_DIR;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;

import com.bfr.opencvapp.objdetect.Detection;

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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Hand Face Human object detector based on a Mobilenetv2-SSD network*/
public class MultiDetector {

    private final String TAG = "MultiDetector";

    //Params for TFlite interpreter
    private final boolean IS_QUANTIZED = false;
    private final Size INPUT_SIZE = new Size(320,320);
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
    final String MODEL_NAME = "ssd_3output_fp32.tflite";

    private Interpreter tfLite;
    private HexagonDelegate hexagonDelegate;

    // Pose estimation Movenet model, used to double check human silhouette, using the confidence score
    private TfLiteMovenet movenetDetector;

    // confidence level of human detection for doublecheck with Movenet
    public float humanConfidence = 0.0f;

    public MultiDetector()
    {
        this(null);
    }
    public MultiDetector(Context context){

        try{

            Runnable initTfMove = new Runnable() {
                @Override
                public void run() {
                    //movenet model to doublecheck human silouhette
                    movenetDetector = new TfLiteMovenet(context);
                }
            };

            Runnable initTflite = new Runnable() {
                @Override
                public void run() {
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
            };

            //
            ExecutorService executorService =
                    new ThreadPoolExecutor(1, 3, 0L, TimeUnit.MILLISECONDS,
                            new LinkedBlockingQueue<Runnable>());

            executorService.submit(initTfMove);
            executorService.submit(initTflite);

            //wait for end of tasks
            executorService.shutdown();
            try {
                executorService.awaitTermination(50000, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }



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
     * @param humanThres Threshold for human detection. Set very high >1.0 to exclude detection
     * @param faceThres Threshold for face detection. Set very high >1.0 to exclude detection
     * @param handThres Threshold for hand detection. Set very high >1.0 to exclude detection
     * @param doubleCheckThres Threshold for double checking a humandetection
     *                         doesn't work for face and hand detections.
     *                         to disable the double check, specify a value <=0.0F
     * @param withDisplay enable/disable the creation of the display mat to gain CPU ressources
     * @return array of detections
     */
    public ArrayList<Detection> recognizeImage(Mat frame, float humanThres, float faceThres, float handThres,
                                               float doubleCheckThres,
                                               boolean withDisplay) {

//        Log.i(TAG, "Starting Multidetector recognition" +
//                " humanThres=" + humanThres + " faceThres=" + faceThres + " handThes=" + handThres);

        ArrayList<Detection> detections = new ArrayList<Detection>();
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

            // SSD outputs 4 maps corresponding to 50 object detections
            // 1) a fp32{1,50} map of the 50 scores of confidence
            outputMap.put(0, new float[1][OUTPUT_MAPS_SIZE[0]]);
            // 2) a fp32{1,50,4} map of 50x4 values for xmin, ymin, xmax, ymax in [0;1]
            outputMap.put(1, new float[1][OUTPUT_MAPS_SIZE[1]][4]);
            // 3) not used
            outputMap.put(2, new float[1]);
            // 4) a fp32{1,50} map of the 50 labels of the detected class
            outputMap.put(3, new float[1][OUTPUT_MAPS_SIZE[3]]);

            // Run inference
            tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

            //explicit names for better readibility of output
            float[][]  out_score= (float [][]) outputMap.get(0);
            float[][][] bboxes = (float[][][]) outputMap.get(1);
            float[][] out_labels = (float[][]) outputMap.get(3);

            //init for display only
            objId = 0;
            // for each detection
            for (int i = 0; i < OUTPUT_MAPS_SIZE[0]; i++)
            {
                int detectedClass = (int) out_labels[0][i];
                float score = out_score[0][i];



                // filter by class
                if ( (detectedClass == 0 &&  score > humanThres)  // human detection
                        || (detectedClass == 1 &&  score > faceThres) //face detection
                        || (detectedClass == 2 &&  score > handThres) ) // hand detection
                {

                    Log.d(TAG, "Object detected : class=" + detectedClass + " score=" + score);
                    // position in % of the image
                    final float ymin = bboxes[0][i][0];
                    final float xmin = bboxes[0][i][1];
                    final float ymax = bboxes[0][i][2];
                    final float xmax = bboxes[0][i][3];

                    //dimension check
                    if( ymin < ymax && xmin < xmax){

                        // if human and need to double check
                        if(detectedClass == 0 && doubleCheckThres>0.0f)
                        {
                            // crop image around human detection
                            int left = (int)(xmin * frame.cols());
                            int top = (int)(ymin * frame.rows());
                            int right = (int)(xmax * frame.cols());
                            int bottom = (int)(ymax* frame.rows());

                            Rect toCrop = new Rect(
                                    Math.max(0,left),
                                    Math.max(0,top),
                                    Math.min(frame.cols()-left,right-left-1),
                                    Math.min(frame.rows()-top, bottom-top-1)
                            );

                            //crop
                            Log.d(TAG, "To crop "+left + " " + top + " " + (right-left) + " " + (bottom-top) );
                            Mat croppedTargetMat = frame.clone().submat(toCrop);

                            // double check if is really a human
                            isReallyHuman = doubleCheckHuman(movenetDetector, croppedTargetMat, doubleCheckThres);

                            if(isReallyHuman)
                            {
                                // add detection
                                detections.add(new Detection("" + i, LABELS[detectedClass], score, xmin, xmax, ymin, ymax, detectedClass));
                            }
                            // else ignore this detection

                        }
                        else // if not a human or no need to double check
                        {
                            detections.add(new Detection("" + i, LABELS[detectedClass], score, xmin, xmax, ymin, ymax, detectedClass));
                        }

                        /********* display*/
                        if (withDisplay)
                        {

                            if( ((detectedClass == 0 && isReallyHuman) || detectedClass >0 ) )
                            {
                                //left
                                pt1.x = (int) (xmin* displayMat.cols());
                                //top
                                pt1.y = (int) (ymin * displayMat.rows());
                                //right
                                pt2.x = (int) (xmax * displayMat.cols());
                                //bottom
                                pt2.y = (int) (ymax * displayMat.rows());

                                Scalar color = null;
                                switch (detectedClass){
                                    case 0: // human
                                        color = _GREEN;
                                        break;
                                    case 1: // face
                                        color = _RED;
                                        break;
                                    case 2: // hands
                                        color = _BLUE;
                                        break;
                                }

                                if( detectedClass == 0 && !isReallyHuman)
                                    color = _RED;

                                // Draw rectangle around detected object.
                                Imgproc.rectangle(displayMat, pt1, pt2,
                                        color, 2);
                                // Write class name or confidence.
                                Imgproc.putText(displayMat, "id:" + String.valueOf(objId)+ " [" + String.format(java.util.Locale.US,"%.3f", score)+"]" , pt1,
                                        1, 3, _BLACK, 7);
                                Imgproc.putText(displayMat, "id:" + String.valueOf(objId) + " [" + String.format(java.util.Locale.US,"%.3f", score)+"]", pt1,
                                        1, 3, color, 3);

                                readyToDisplay = true;
                                objId = objId+1;

                            } // end if correct detection
                        }

                    } //end if ymin < ymax && xmin < xmax

                } // //end if score OK
            } // next detection


        } catch (Exception e) {
            e.printStackTrace();
        }
        return detections;

    }




    /**
     * Doublechecks a human detection, by computing the average of confidence in Pose detection
     * @param movenet a movenet pose detector
     * @param detectionImg an image containing the supposed human, typically obtained from a bounding box of a human detector
     *                     Must be 256x256
     * @param thres a confidence threshold to accept a human (recommended value=0.3)
     * @return array of detections
     */
    public boolean doubleCheckHuman(TfLiteMovenet movenet, Mat detectionImg, float thres)
    {

        float[][][][] result = movenet.recognizeImage(detectionImg);

        //init
        humanConfidence = 0.0f;
        //for each keypoint including:
        // left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip
        for (int i = 5; i < 12; i++) {
            humanConfidence += result[0][0][i][2];
        }
        //computing average
        humanConfidence = humanConfidence/7;

        return (humanConfidence>=thres);
    }


}
