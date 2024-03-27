package com.bfr.opencvapp.utils;

import static com.bfr.opencvapp.utils.Utils.modelsDir;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

/** TFLite implementation of a YoloX detector for Huma, Head, and hands*/
public class TfLiteYoloXHumanHeadHands {

    private final String TAG = "TfLiteYOLOX";

    //Params for TFlite interpreter
    private final boolean IS_QUANTIZED = false;
    private final int[] INPUT_SIZE = {320,320};
    private final int[] OUTPUT_SIZE = {60,7}; // 20 bounding box max per class -> 20x3 = 60,
                                            // where each bbox is batch_no, class_id, score, w1, y1, x2, y2 -> 7 floats
    private static final float[] IMAGE_MEAN = {0, 0, 0};
    private static final float[] IMAGE_STD = {1, 1, 1};
    private final int NUM_CLASSES = 1;
    private final int BATCH_SIZE = 1;
    private final int PIXEL_SIZE = 3;
    private final int NUM_THREADS = 4;

    Mat displayMat;

    /**
     * NB: Topformer doesn't work well with GPU (some actually is executed on the CPU)
     * Also the values differ from GPU to CPU (more robust on CPU)
     *
     * Topformer doesn't support the NNAPI delegate
     */
    private boolean WITH_NNAPI = false;
    private boolean WITH_GPU = true ;
    private boolean WITH_DSP = false;

//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x512x512_float32.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x512x512_float16.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x288x480_float32.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x480x640_float16.tflite";
    private final String MODEL_NAME = "yolox_t_body_head_hand_post_0299_0.4522_1x3x320x320_float32.tflite";
//    private final String MODEL_NAME = "yolox_s_body_head_hand_post_0299_0.4983_1x3x320x320_float32.tflite";
//    private final String MODEL_NAME = "TopFormer-S_512x512_2x8_160k_argmax.tflite";

    private Interpreter tfLite;
    private HexagonDelegate hexagonDelegate;


    // Input
    private Bitmap inputBitmap;
    // Output
    /** Output probability TensorBuffer. */
    private TensorBuffer outputProbabilityBuffer;
    private ByteBuffer outputBuffer;

    public TfLiteYoloXHumanHeadHands(Context context){

        try{
            Interpreter.Options options = (new Interpreter.Options());


            /**
             * The following commented code is for experiment only.
             * Topformer doesn't work well with GPU (some actually is executed on the CPU)
             * Also the values differ from GPU to CPU (more robust on CPU)
             *
             * Topformer doesn't support the NNAPI delegate
             */
            CompatibilityList compatList = new CompatibilityList();
            if (WITH_GPU) {
                GpuDelegateFactory.Options delegateOptions = compatList.getBestOptionsForThisDevice();
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
            else if (WITH_NNAPI) {
                options.setUseXNNPACK(false);
                NnApiDelegate nnApiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    nnApiDelegate = new NnApiDelegate();
                    options.addDelegate(nnApiDelegate);
                    options.setUseNNAPI(true);
                }
            }
            else{
                options.setNumThreads(NUM_THREADS);
                options.setUseXNNPACK(true);
                WITH_NNAPI = false;
                Log.i(TAG, "Interpreter on CPU");
            }



            //Init interpreter
            File tfliteModel = new File(modelsDir+MODEL_NAME);
            tfLite = new Interpreter(tfliteModel, options );
        }
        catch (Exception e)
        {
            Log.e(TAG, "Error Creating the tflite model " + Log.getStackTraceString(e) );
        }

        // allocating memory for output
        outputBuffer = ByteBuffer.allocateDirect(1 * OUTPUT_SIZE[0] * OUTPUT_SIZE[1] * NUM_CLASSES * 4);
        // Creates the output tensor and its processor.
        int[] probabilityShape =
                tfLite.getOutputTensor(0).shape();
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, DataType.FLOAT32);


        // displya results image
        displayMat = new Mat();
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
                    // red
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    // blue
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    // green
                    byteBuffer.put((byte) (val & 0xFF));
                } else {
                    // red
                    byteBuffer.putFloat( ( (float)(val >> 16 & 0xFF) - 127.5f) / IMAGE_STD[0]);
                    // blue
                    byteBuffer.putFloat( ((float)(val >> 8 & 0xFF) - IMAGE_MEAN[1]) / IMAGE_STD[1]);
                    // green
                    byteBuffer.putFloat( ((float)(val & 0xFF) - IMAGE_MEAN[2]) / IMAGE_STD[2]);
                }
            }
        }
        byteBuffer.rewind();
        return byteBuffer;
    }



    /**
     * Get a 64x64 flatten array of semantic segmentations
     * @param frame original image to process in OpenCV Mat format
     * @return array of detections
     */
    public ArrayList<Recognition> recognizeImage(Mat frame, float humanThres, float headThres, float handThres) {

        //convert to bitmap
        Bitmap bitmapImage = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame.clone(), bitmapImage);

        // inference
        ArrayList<Recognition> listOfDetections = recognizeImage(bitmapImage, humanThres, headThres, handThres);

        displayMat = displayResult(frame, listOfDetections);

        return listOfDetections;

    }



    /**
     * Get a 64x64 flatten array of semantic segmentations
     * @param bitmap original image to process in bitmapZ
     * @return array of detections
     */
    public ArrayList<Recognition> recognizeImage(Bitmap bitmap, float humanThres, float headThres, float handThres) {

        // save for display
        this.inputBitmap = bitmap.copy(bitmap.getConfig(), true);

        ByteBuffer byteBuffer;

        //Check size
        if (bitmap.getWidth() == INPUT_SIZE[0] && bitmap.getHeight() == INPUT_SIZE[1]) // if correct input size
        {
            byteBuffer = convertBitmapToByteBuffer(bitmap);
        }
        else // resize
        {
            byteBuffer = convertBitmapToByteBuffer(
                    Bitmap.createScaledBitmap(bitmap, INPUT_SIZE[0], INPUT_SIZE[1], false));
        }

        // Input
        Object[] inputArray = {byteBuffer.rewind()};
        // Output
        outputBuffer.rewind();
        //inference
//        tfLite.run(byteBuffer, outputBuffer);

        tfLite.run(byteBuffer.rewind(), outputProbabilityBuffer.getBuffer().rewind());

        // local vars for more readibility
        int width = OUTPUT_SIZE[0];
        int height = OUTPUT_SIZE[1];
        int outArray[] = new int[ width* height];

        outputBuffer.rewind();

        ArrayList<Recognition> listOfDetections = new ArrayList<Recognition>();
        String[] LABELS = {"Human", "Face", "Hand"};

        float[] floatOutput = outputProbabilityBuffer.getFloatArray();
        //for each dimension
        for (int i =0; i<floatOutput.length; i+=7) {
            int detectedClass = (int) floatOutput[i+1];
            float score = floatOutput[i+2];

            if (  (detectedClass == 0 && score > humanThres)  // human
            || (detectedClass == 1 && score > headThres) // head
            || (detectedClass == 2 && score > handThres) // hands
        ){
                // position in % of the image
                final float x1 = floatOutput[i+3]/INPUT_SIZE[0];
                final float y1 = floatOutput[i+4]/INPUT_SIZE[1];
                final float x2 = floatOutput[i+5]/INPUT_SIZE[0];
                final float y2 = floatOutput[i+6]/INPUT_SIZE[1];
                Log.d(TAG, "detection n."+ (i) + " class="+ detectedClass   + " " + score + " " + x1 + "," + y1  );
//                        + " " + x1 + "," + y1 + "," + x2 + "," + y2);
                if( y1 < y2 && x1 < x2){
                    listOfDetections.add(new Recognition("" + i, LABELS[detectedClass], score, x1, x2, y1, y2, detectedClass));
                } //end check

            } //end check score
        } //next detection



        return  listOfDetections;






//        // local vars for more readibility
//        int width = OUTPUT_SIZE[0];
//        int height = OUTPUT_SIZE[1];
//        float outArray[] = new float[ width* height];
//        Map<Integer, Object> outputMap = new HashMap<>();
//
//        outputMap.put(0, new float[width][height]);
////        outputMap.put(1, new float[height]);
//
//        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
//
//        float[][] detection = (float[][]) outputMap.get(0);
//
//        ArrayList<TfLiteYoloX.Recognition> listOfDetections = new ArrayList<TfLiteYoloX.Recognition>();
//
//        String[] LABELS = {"Human", "Face", "Hand"};
//
//        for (int i = 0; i < OUTPUT_SIZE[0];i++){
//            int detectedClass = (int) detection[i][1];
//            float score = detection[i][2];
////            Log.w(TAG, i + " " + detectedClass + " " + score ); // + " " + x1 + "," + y1 + "," + x2 + "," + y2);
//            if ( score > 0.0){
//                // position in % of the image
//                final float x1 = detection[i][3];
//                final float y1 = detection[i][4];
//                final float x2 = detection[i][5];
//                final float y2 = detection[i][6];
//
//                if( y1 < y2 && x1 < x2){
//                    listOfDetections.add(new TfLiteYoloX.Recognition("" + i, LABELS[detectedClass], score, x1, x2, y1, y2, detectedClass));
//                }
//
//            }
//        }
//
//        return listOfDetections;


    } // end recognizeImage


    /**
     * Display tracked bounding box
     * @param frame the original frame
    //     * @param score the confidence score of the tracker
     * @return a Mat dislpaying the tracked bounding box
     */
    public Mat displayResult(Mat frame,  ArrayList<Recognition> listOfDetections)
    {
        // Display results
        Mat todisplay = frame.clone();

        // for each detection
        for (int k=0; k< listOfDetections.size(); k++)
        {
            float score= listOfDetections.get(k).confidence;
            int x1 = (int) (listOfDetections.get(k).left * todisplay.cols());
            int y1 = (int) (listOfDetections.get(k).top * todisplay.rows());
            int x2 = (int) (listOfDetections.get(k).right * todisplay.cols());
            int y2 = (int) (listOfDetections.get(k).bottom * todisplay.rows());
            int classId = listOfDetections.get(k).getDetectedClass();

//            Log.w(TAG, i + " " + classId + " " + score + " coords=" + x1 + "," + y1 + "," + x2 + "," + y2);

            Scalar color = null;
            switch (classId){
                case 0:
                    color = new Scalar(255,0,0);
                    break;
                case 1:
                    color = new Scalar(0,255,0);
                    break;

                case 2:
                    color = new Scalar(0,0,255);
                    break;
            }
            Imgproc.rectangle(todisplay, new Point(x1, y1), new Point(x2, y2), color, 4);
            Imgproc.putText(todisplay, String.valueOf(score), new Point(x1, y1-10), 1, 2, new Scalar(0,0,0), 5 );
            Imgproc.putText(todisplay, String.valueOf(score), new Point(x1, y1-10), 1, 2, color, 2 );

        }//next detection

        Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_WholeDetectYOLOX.jpg",
                todisplay);


        return todisplay;

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

