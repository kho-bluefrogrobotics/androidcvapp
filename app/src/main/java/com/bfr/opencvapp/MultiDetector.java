package com.bfr.opencvapp;


import static com.bfr.opencvapp.utils.Utils.Color.*;
import static com.bfr.opencvapp.utils.Utils.modelsDir;

import static org.opencv.core.CvType.CV_8UC3;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
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
public class MultiDetector {

    private final String TAG = "MultiDetector";

    //Params for TFlite interpreter
    private final boolean IS_QUANTIZED = false;
    private final int[] INPUT_SIZE = new int[]{320,320};
    private final int[] OUTPUT_WIDTH_SSD = new int[]{50, 50, 50};
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
    final String MODEL_NAME = "MobileNetSSD_3classes_fp32.tflite";

    private Interpreter tfLite;
    private HexagonDelegate hexagonDelegate;

    // Pose estimation Movenet model, used to double check human silhouette, using the confidence score
    private TfLiteMovenet movenetDetector;

    // confidence level of human detection for doublecheck with Movenet
    public float humanConfidence = 0.0f;

    public MultiDetector(Context context){

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
            File tfliteModel = new File(modelsDir+MODEL_NAME);
            tfLite = new Interpreter(tfliteModel, options );


            //movenet model to doublecheck human silouhette
            movenetDetector = new TfLiteMovenet(context);
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
     *
     * @return array of detections
     */
    public ArrayList<Recognition> recognizeImage(Mat frame, float humanThres, float faceThres, float handThres,
                                                 float doubleCheckThres,
                                                 boolean withDisplay) {

        Log.i(TAG, "Starting Multidetector recognition" +
                " humanThres=" + humanThres + " faceThres=" + faceThres + " handThes=" + handThres);

        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        boolean isReallyHuman = true;

        try
        {
            displayMat = frame.clone();

            //convert to bitmap
            Mat resizedFrame = new Mat();
            Imgproc.resize(frame, resizedFrame, new Size(INPUT_SIZE[0],INPUT_SIZE[1]));
            Bitmap bitmapImg = Bitmap.createBitmap(resizedFrame.cols(), resizedFrame.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(resizedFrame, bitmapImg);

            ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmapImg);

            Map<Integer, Object> outputMap = new HashMap<>();

            outputMap.put(0, new float[1][OUTPUT_WIDTH_SSD[0]]);
            outputMap.put(1, new float[1][OUTPUT_WIDTH_SSD[1]][4]);
            outputMap.put(2, new float[1]);
            outputMap.put(3, new float[1][OUTPUT_WIDTH_SSD[1]]);

            Object[] inputArray = {byteBuffer};
            tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

            float[][]  out_score= (float [][]) outputMap.get(0);
            float[][][] bboxes = (float[][][]) outputMap.get(1);
            float[][] out_labels = (float[][]) outputMap.get(3);

            //init
            objId = 0;

            // for each detection
            for (int i = 0; i < OUTPUT_WIDTH_SSD[0];i++)
            {
                int detectedClass = (int) out_labels[0][i];
                final float score = out_score[0][i];

                Log.d(TAG, "Object detected : class=" + detectedClass + " score=" + score);

                // filter by class
                if ( (detectedClass == 0 &&  score > humanThres)  // human detection
                        || (detectedClass == 1 &&  score > faceThres) //face detection
                        || (detectedClass == 2 &&  score > handThres) ) // hand detection
                {

                    // position in % of the image
                    final float ymin = bboxes[0][i][0];
                    final float xmin = bboxes[0][i][1];
                    final float ymax = bboxes[0][i][2];
                    final float xmax = bboxes[0][i][3];

                    if( ymin < ymax && xmin < xmax){

                        // if human and need to double check
                        if(detectedClass == 0 && doubleCheckThres>0.0f)
                        {
                            // crop image around human detection
                            // for display only
                            int cols = frame.cols();
                            int rows = frame.rows();

                            int left = (int)(xmin * cols);
                            int top = (int)(ymin * rows);
                            int right = (int)(xmax * cols);
                            int bottom = (int)(ymax* rows);

                            Rect toCrop = new Rect(
                                    Math.max(0,left),
                                    Math.max(0,top),
                                    Math.min(cols-left,right-left-1),
                                    Math.min(rows-top, bottom-top-1)
                            );

                            Log.d(TAG, "To crop "+left + " " + top + " " + (right-left) + " " + (bottom-top) );

                            Mat croppedTargetMat = frame.clone().submat(toCrop);

                            // double check if is really a human
                            isReallyHuman = doubleCheckHuman(movenetDetector, croppedTargetMat, doubleCheckThres);

                            if(isReallyHuman)
                            {
                                // add detection
                                detections.add(new Recognition("" + i, LABELS[detectedClass], score, xmin, xmax, ymin, ymax, detectedClass));
                            }
                            // else ignore this detection

                        }
                        else // if not a human or no need to double check
                        {
                            detections.add(new Recognition("" + i, LABELS[detectedClass], score, xmin, xmax, ymin, ymax, detectedClass));
                        }

                        // ********* display
                        if (withDisplay)
                        {
//                            if( (detectedClass == 0 && isReallyHuman) || detectedClass >0)
                            if( (detectedClass == 0) || detectedClass >0)
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
                                        color = _BLUE;
                                        break;
                                    case 2: // hands
                                        color = _RED;
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

//            Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_WholeDetect.jpg",
//                    displayMat);

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




    // return object by tflite interpreter
    public class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        public final Float confidence;

        /**
         * Optional location within the source image for the location of the recognized object.
         */
        private RectF location;

        private int detectedClass;

        public Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public Recognition(final String id, final String title, final Float confidence, final RectF location, int detectedClass) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
            this.detectedClass = detectedClass;
        }

        public Recognition(final String id, final String title, final Float confidence, float left, float right, float top, float bottom, int detectedClass) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
            this.detectedClass = detectedClass;

            this.left= left;
            this.right=right;
            this.top=top;
            this.bottom=bottom;
        }

        public float left, right, top, bottom=0;

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        public int getDetectedClass() {
            return detectedClass;
        }

        public void setDetectedClass(int detectedClass) {
            this.detectedClass = detectedClass;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }


}
