package com.bfr.opencvapp.utils;

import static java.lang.Math.min;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
//import org.tensorflow.lite.support.common.TensorOperator;
//import org.tensorflow.lite.support.common.ops.NormalizeOp;
//import org.tensorflow.lite.support.image.TensorImage;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Face recognizer based on a Mobilenet-Facenet trained with Sigmoid loss - int8 quantized*/
public class TfLiteTopFormer {

    private final String TAG = "TfLiteMidas";

    //Params for TFlite interpreter
    private final boolean IS_QUANTIZED = false;
    private final int[] INPUT_SIZE = {512,512};
    private final int[] OUTPUT_SIZE = {64,64};
//    private static final float IMAGE_MEAN = 0.450f;
//    private static final float IMAGE_STD = 0.225f;
    private static final float[] IMAGE_MEAN = {127.5f, 127.5f, 127.5f};
    private static final float[] IMAGE_STD = {127.5f, 127.5f, 127.5f};
    private final int NUM_CLASSES = 1;
    private final int BATCH_SIZE = 1;
    private final int PIXEL_SIZE = 3;
    private final float THRES = 0.75f;
    private final String[] LABELS = {"Human", "Face", "Hand"};
    private final int NUM_THREADS = 6;
    private boolean WITH_NNAPI = false;
    private boolean WITH_GPU = false;
    private boolean WITH_DSP = false;
    //Face embedding
    private float[] embeedings;

    //where to find the models
    private final String DIR = "/sdcard/Android/data/com.bfr.opencvapp/files/nnmodels/";
//    private final String MODEL_NAME = "pyDNet__256x320_float16_quant.tflite";
//    private final String MODEL_NAME = "Midas_float32.tflite";
    private final String MODEL_NAME = "TopFormer-T_512x512_2x8_160k_float16_quant_argmax.tflite";
//    private final String MODEL_NAME = "Fastdepth_512x512_float32.tflite";
//    private final String MODEL_NAME = "fastdepth_256x256_float16_quant.tflite";
//    private final String MODEL_NAME = "pydnet_256x320.tflite";

    private Interpreter tfLite;
    private HexagonDelegate hexagonDelegate;


    public TfLiteTopFormer(Context context){

        try{
            Interpreter.Options options = (new Interpreter.Options());
            CompatibilityList compatList = new CompatibilityList();

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
            else if (WITH_NNAPI) {
                options.setUseXNNPACK(false);
                NnApiDelegate nnApiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
//                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                if (true) {
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

//            if (WITH_NNAPI) {
//                options.setUseXNNPACK(false);
//                NnApiDelegate nnApiDelegate = null;
//                // Initialize interpreter with NNAPI delegate for Android Pie or above
//                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//                    nnApiDelegate = new NnApiDelegate();
//                    options.addDelegate(nnApiDelegate);
//                    options.setUseNNAPI(true);
//                }
//            }

            //Init interpreter
            File tfliteModel = new File(DIR+MODEL_NAME);
            tfLite = new Interpreter(tfliteModel, options );
        }
        catch (Exception e)
        {
            Log.e(TAG, "Error Creating the tflite model " + Log.getStackTraceString(e) );
        }


        // Creates the input tensor.
//        inputImageBuffer = new TensorImage(DataType.FLOAT32);
//        int[] imageShape = tfLite.getInputTensor(0).shape(); // {1, height, width, 3}
//        if(imageShape[1] != imageShape[2]) {
//            imageSizeY = imageShape[2];
//            imageSizeX = imageShape[3];
////            imageSizeY = 256;
////            imageSizeX = 320;
//        } else {
//            imageSizeY = imageShape[1];
//            imageSizeX = imageShape[2];
////            imageSizeY = 256;
////            imageSizeX = 320;
//        }

        imageSizeX = INPUT_SIZE[0];
        imageSizeY = INPUT_SIZE[1];
//
//        int[] probabilityShape = {1,OUTPUT_SIZE[0],OUTPUT_SIZE[1],1};
////                tfLite.getOutputTensor(0).shape(); // {1, NUM_CLASSES}
//
//        Log.w("coucou", "init output size: " + probabilityShape[0] + " " + probabilityShape[1] + " " + probabilityShape[2] );
        // Creates the output tensor and its processor.

        outputBuffer = ByteBuffer.allocateDirect(1 * OUTPUT_SIZE[0] * OUTPUT_SIZE[1] * NUM_CLASSES * 4);

    }



    /** Input image TensorBuffer. */
//    private TensorImage inputImageBuffer;
    /** Output probability TensorBuffer. */
    private ByteBuffer outputBuffer;

    /** Image size along the x axis. */
    private final int imageSizeX;
    /** Image size along the y axis. */
    private final int imageSizeY;
//    protected TensorOperator getPreprocessNormalizeOp() {
//        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
//    }


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
        float mean = 127.5F;
        float std = 127.5F;
        for (int i = 0; i < INPUT_SIZE[0]; ++i) {
            for (int j = 0; j < INPUT_SIZE[1]; ++j) {
                final int val = intValues[pixel++];
//                Log.w("coucou", "i,j:" +i + ","+j + " :"  +val);
                if (IS_QUANTIZED) {
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {
                    byteBuffer.putFloat( ( (float)(val >> 16 & 0xFF) - 127.5f) / IMAGE_STD[0]);
                    byteBuffer.putFloat( ((float)(val >> 8 & 0xFF) - IMAGE_MEAN[1]) / IMAGE_STD[1]);
                    byteBuffer.putFloat( ((float)(val & 0xFF) - IMAGE_MEAN[2]) / IMAGE_STD[2]);
                }
            }
        }
        byteBuffer.rewind();
        return byteBuffer;
    }



    /**
     * get the detected objects in the image
     * @param bitmap original image in bitmap format
     * @return array of detections
     */
    public int[] recognizeImage(Bitmap bitmap) {

        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);

//        ArrayList<Recognition> detections = new ArrayList<Recognition>();
//        Map<Integer, Object> outputMap = new HashMap<>();

        // Init Face embeedings (signature)
//        embeedings = new float[1][OUTPUT_SIZE[0]][OUTPUT_SIZE[1]][1];
//        embeedings = new float[1][OUTPUT_SIZE[0]][OUTPUT_SIZE[1]];
//        embeedings = new float[1][16][16][48];
        // Assign to Facenet output
//        outputMap.put(0, embeedings);
//        outputMap.put(1, new float[1][16]);
//        outputMap.put(2, new float[1][16]);
//        outputMap.put(3, new float[1][48]);

        Object[] inputArray = {byteBuffer};
        outputBuffer.rewind();
        tfLite.run(byteBuffer, outputBuffer);

        int width = OUTPUT_SIZE[0];
        int height = OUTPUT_SIZE[1];
        int outArray[] = new int[width*height];
        //
        outputBuffer.rewind();
        for (int y =0; y<height; y++) {
            for (int x = 0; x < width; x++) {
//                Log.w("coucou", (y * width  + x) +": outbytebuffer value = " + outputBuffer.get((y * width  + x)*4 ));
                outArray[(y * width  + x)] =  outputBuffer.get((y * width  + x)*4); //*4 because the output are INT32
            }
        }

        return outArray;

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
