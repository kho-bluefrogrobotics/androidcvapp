package com.bfr.opencvapp.utils;

import static java.lang.Math.min;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
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

    private final String TAG = "TfLiteTopformer";

    //Params for TFlite interpreter
    private final boolean IS_QUANTIZED = false;
    private final int[] INPUT_SIZE = {512,512};
    private final int[] OUTPUT_SIZE = {64,64};
    private static final float[] IMAGE_MEAN = {127.5f, 127.5f, 127.5f};
    private static final float[] IMAGE_STD = {127.5f, 127.5f, 127.5f};
    private final int NUM_CLASSES = 1;
    private final int BATCH_SIZE = 1;
    private final int PIXEL_SIZE = 3;
    private final int NUM_THREADS = 4;
    /**
     * NB: Topformer doesn't work well with GPU (some actually is executed on the CPU)
     * Also the values differ from GPU to CPU (more robust on CPU)
     *
     * Topformer doesn't support the NNAPI delegate
     */
    private boolean WITH_NNAPI = false;
    private boolean WITH_GPU = false;
    private boolean WITH_DSP = false;

    //where to find the models
    private final String DIR = "/sdcard/Android/data/com.bfr.opencvapp/files/nnmodels/";

//    private final String MODEL_NAME = "TopFormer-T_512x512_2x8_160k_float16_quant_argmax.tflite";
    private final String MODEL_NAME = "TopFormer-T_512x512_2x8_160k_float16_quant_argmax.tflite";
//    private final String MODEL_NAME = "TopFormer-S_512x512_2x8_160k_argmax.tflite";

    private Interpreter tfLite;
    private HexagonDelegate hexagonDelegate;

    /** Output */
    private ByteBuffer outputBuffer;


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
            File tfliteModel = new File(DIR+MODEL_NAME);
            tfLite = new Interpreter(tfliteModel, options );
        }
        catch (Exception e)
        {
            Log.e(TAG, "Error Creating the tflite model " + Log.getStackTraceString(e) );
        }

        // allocating memory for output
        outputBuffer = ByteBuffer.allocateDirect(1 * OUTPUT_SIZE[0] * OUTPUT_SIZE[1] * NUM_CLASSES * 4);

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
     * get the detected objects in the image
     * @param bitmap original image in bitmap format
     * @return array of detections
     */
    public int[] recognizeImage(Bitmap bitmap) {

        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);

        outputBuffer.rewind();
        tfLite.run(byteBuffer, outputBuffer);

        int width = OUTPUT_SIZE[0];
        int height = OUTPUT_SIZE[1];
        int outArray[] = new int[width*height];
        //
        outputBuffer.rewind();
        for (int y =0; y<height; y++) {
            for (int x = 0; x < width; x++) {
              outArray[(y * width  + x)] =  outputBuffer.get((y * width  + x)*4); //*4 because the output are INT32,
                // then we read on a single byte with .get
            }
        }

        return outArray;

    }


}
