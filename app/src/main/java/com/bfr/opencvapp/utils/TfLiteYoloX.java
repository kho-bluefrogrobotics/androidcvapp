package com.bfr.opencvapp.utils;

import static com.bfr.opencvapp.utils.Utils.modelsDir;
import static org.opencv.core.CvType.CV_8UC3;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.lite.DelegateFactory;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/** TFLite implementation of TopFormer trained on ADE20k
 * See the complete list of classes at the end of the file*/
public class TfLiteYoloX {

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
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x512x512_float32.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x288x480_float32.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x480x640_float16.tflite";
//    private final String MODEL_NAME = "yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.tflite";
//    private final String MODEL_NAME = "yolox_s_body_head_hand_post_0299_0.4983_1x3x320x320_float32.tflite";
//    private final String MODEL_NAME = "yolox_t_body_head_hand_post_0299_0.4522_1x3x192x416_float32.tflite";
    private final String MODEL_NAME = "yolox_t_body_head_hand_post_0299_0.4522_1x3x320x320_float32.tflite";
//    private final String MODEL_NAME = "TopFormer-S_512x512_2x8_160k_argmax.tflite";

    private Interpreter tfLite;
    private HexagonDelegate hexagonDelegate;


    // Input
    private Bitmap inputBitmap;
    // Output
    /** Output probability TensorBuffer. */
    private TensorBuffer outputProbabilityBuffer;
    private ByteBuffer outputBuffer;

    public TfLiteYoloX(Context context){

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
    public int[] segmentFloorFromMat(Mat frame) {

        //convert to bitmap
        Bitmap bitmapImage = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame.clone(), bitmapImage);

        // inference
        return segmentFloorFromBitmap(bitmapImage);
    }


    /**
     * Get a 64x64 flatten array of semantic segmentations
     * @param frame original image to process in OpenCV Mat format
     * @return array of detections
     */
    public ArrayList<TfLiteYoloX.Recognition> runInference(Mat frame) {

        //convert to bitmap
        Bitmap bitmapImage = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame.clone(), bitmapImage);

        // inference
        return runInference(bitmapImage);
    }



    /**
     * Get a 64x64 flatten array of semantic segmentations
     * @param bitmap original image to process in bitmap
     * @return array of detections
     */
    public ArrayList<TfLiteYoloX.Recognition> runInference(Bitmap bitmap) {

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

        ArrayList<TfLiteYoloX.Recognition> listOfDetections = new ArrayList<TfLiteYoloX.Recognition>();
        String[] LABELS = {"Human", "Face", "Hand"};

        float[] floatOutput = outputProbabilityBuffer.getFloatArray();
        //for each dimension
        for (int i =0; i<floatOutput.length; i+=7) {
            int detectedClass = (int) floatOutput[i+1];
            float score = floatOutput[i+2];

            if (  (detectedClass == 0 && score > 0.3)  // human
            || (detectedClass == 1 && score > 0.3) // head
            || (detectedClass == 2 && score > 0.3) // hands
        ){
                // position in % of the image
                final float x1 = floatOutput[i+3]/INPUT_SIZE[0];
                final float y1 = floatOutput[i+4]/INPUT_SIZE[1];
                final float x2 = floatOutput[i+5]/INPUT_SIZE[0];
                final float y2 = floatOutput[i+6]/INPUT_SIZE[1];
                Log.w(TAG, (i) + " " + floatOutput[i+0] + " " + detectedClass + " " + score );
//                        + " " + x1 + "," + y1 + "," + x2 + "," + y2);
                if( y1 < y2 && x1 < x2){
                    listOfDetections.add(new TfLiteYoloX.Recognition("" + i, LABELS[detectedClass], score, x1, x2, y1, y2, detectedClass));
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
     * Get a 64x64 flatten array of semantic segmentations
     * @param bitmap original image to process in bitmap
     * @return array of detections
     */
    public int[] segmentFloorFromBitmap(Bitmap bitmap) {

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

        outputBuffer.rewind();
        //inference
        tfLite.run(byteBuffer, outputBuffer);

        // local vars for more readibility
        int width = OUTPUT_SIZE[0];
        int height = OUTPUT_SIZE[1];
        int outArray[] = new int[ width* height];

        outputBuffer.rewind();
        //for each dimension
        for (int y =0; y<height; y++) {
            for (int x = 0; x < width; x++) {
              outArray[(y * width  + x)] =  outputBuffer.get((y * width  + x)*4); //*4 because the output are INT32,
                // then we read on a single byte with .get
            }
        }

        return outArray;

    } // end recognizeImage


    /**
     * Get a WidthxHeight mask image of the floor (in bitmap format). typically from the result of recognizeImage()
     * @param array the array containing the segmented objects
     * @param color the color to display the floor : should be ANDROID_GREEN, ANDROID_RED, ANDROID_BLUE, ANDROID_WHITE or any declinaison using  Color.rgb()
     * @param width the desired width of the image
     * @param height the desired height of the image
     * @return bitmap of the mask image
     */
    public Bitmap getFloorMaskBitmap(int array[], int color, int width, int height)
    {

        Bitmap maskBitmap = Bitmap.createBitmap(OUTPUT_SIZE[0], OUTPUT_SIZE[1], Bitmap.Config.RGB_565);
        // for each pixel
        for (int jj = 0; jj < OUTPUT_SIZE[1]; jj++)//pass the screen pixels in 2 directions
        {
            for (int ii = 0; ii < OUTPUT_SIZE[0]; ii++)
            {
                // if class is something floor-like
                if (array[jj*OUTPUT_SIZE[0] + ii] == 3 //floor
                        || array[jj*OUTPUT_SIZE[0] + ii] == 6 //road
                        || array[jj*OUTPUT_SIZE[0] + ii] == 11 //sidewalk
                        || array[jj*OUTPUT_SIZE[0] + ii] == 13 // earth
                        || array[jj*OUTPUT_SIZE[0] + ii] == 28 // rug, carpet
                        || array[jj*OUTPUT_SIZE[0] + ii] == 52 // path
                        || array[jj*OUTPUT_SIZE[0] + ii] == 54 // runway
                        || array[jj*OUTPUT_SIZE[0] + ii] == 94 // land
                )
                {
                    //colorize pixel in white
                    maskBitmap.setPixel(ii, jj, color);
                } //end if class is floor
                else// not the floor
                {
                    //colorize pixel in black
                    maskBitmap.setPixel(0, 0, 0);
                } // end if floor

            }// next jj
        }//next ii

        return  Bitmap.createScaledBitmap(maskBitmap, width, height, false);
    } // end getFloorMaskBitmap


    /**
     * Get a WidthxHeight image of the original image, blended with the masked floor (in bitmap format). typically from the result of recognizeImage()
     * Obviously this function is to be called AFTER segmentFloorFromBitmap()
     * @param array the array containing the segmented objects
     * @param color the color to display the floor : should be ANDROID_GREEN, ANDROID_RED, ANDROID_BLUE, ANDROID_WHITE or any declinaison using  Color.rgb()
     * @param width the desired width of the image
     * @param height the desired height of the image
     * @return bitmap of the blended image
     */
    public Bitmap getOverlayBitmap(int array[], int color, int width, int height) {
        Bitmap maskBitmap = Bitmap.createBitmap(OUTPUT_SIZE[0], OUTPUT_SIZE[1], Bitmap.Config.RGB_565);
        // for each pixel
        for (int jj = 0; jj < OUTPUT_SIZE[1]; jj++)//pass the screen pixels in 2 directions
        {
            for (int ii = 0; ii < OUTPUT_SIZE[0]; ii++)
            {
                // if class is something floor-like
                if (array[jj*OUTPUT_SIZE[0] + ii] == 3 //floor
                        || array[jj*OUTPUT_SIZE[0] + ii] == 6 //road
                        || array[jj*OUTPUT_SIZE[0] + ii] == 11 //sidewalk
                        || array[jj*OUTPUT_SIZE[0] + ii] == 13 // earth
                        || array[jj*OUTPUT_SIZE[0] + ii] == 28 // rug, carpet
                        || array[jj*OUTPUT_SIZE[0] + ii] == 52 // path
                        || array[jj*OUTPUT_SIZE[0] + ii] == 54 // runway
                        || array[jj*OUTPUT_SIZE[0] + ii] == 94 // land
                )
                {
                    //colorize pixel in white
                    maskBitmap.setPixel(ii, jj, color);
                } //end if class is floor
                else// not the floor
                {
                    //colorize pixel in black
                    maskBitmap.setPixel(0, 0, 0);
                } // end if floor

            }// next jj
        }//next ii

        /**** To OpenCV for easier matrix operations **/
        // display fusion with original image

        //Mask
        Mat maskMat = new Mat();
        Utils.bitmapToMat(maskBitmap, maskMat);
        //resize
        Imgproc.resize(maskMat, maskMat, new Size(width,height));

        //original image
        Mat origImage = new Mat();
        if (this.inputBitmap==null) // null object management
            //init to black image
            origImage = new Mat(height,width, CV_8UC3, new Scalar(0, 0, 0));
        else
            Utils.bitmapToMat(this.inputBitmap, origImage);

        //superposition of original image and mask
        Mat blendedMat = new Mat();
        Core.add(origImage, maskMat, blendedMat);

        //convert to bitmap
        Bitmap blendedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(blendedMat, blendedBitmap);

        return blendedBitmap;

    } // end getOverlayBitmap


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


/*  List of recognized classes
wall,0,0,255
building;edifice,120,120,180
sky,230,230,6
floor;flooring,255,0,0
tree,3,200,4
ceiling,80,120,120
road;route,255,0,0
bed,255,5,204
windowpane;window,0,0,255
grass,7,250,4
cabinet,255,5,224
sidewalk;pavement,255,0,0
person;individual;someone;somebody;mortal;soul,61,5,150
earth;ground,255,0,0
door;double;door,0,0,255
table,82,6,255
mountain;mount,140,255,143
plant;flora;plant;life,4,255,204
curtain;drape;drapery;mantle;pall,0,0,255
chair,3,70,204
car;auto;automobile;machine;motorcar,200,102,0
water,250,230,61
painting;picture,51,6,255
sofa;couch;lounge,255,102,11
shelf,71,7,255
house,224,9,255
sea,230,7,9
mirror,220,220,220
rug;carpet;carpeting,255,0,0
field,255,9,112
armchair,214,255,8
seat,224,255,7
fence;fencing,6,184,255
desk,71,255,10
rock;stone,10,41,255
wardrobe;closet;press,255,255,7
lamp,8,255,224
bathtub;bathing;tub;bath;tub,255,8,102
railing;rail,6,61,255
cushion,7,194,255
base;pedestal;stand,8,122,255
box,20,255,0
column;pillar,41,8,255
signboard;sign,153,5,255
chest;of;drawers;chest;bureau;dresser,255,51,6
counter,255,12,235
sand,20,150,160
sink,255,163,0
skyscraper,140,140,140
fireplace;hearth;open;fireplace,15,10,250
refrigerator;icebox,0,255,20
grandstand;covered;stand,0,255,31
path,255,0,0
stairs;steps,0,224,255
runway,255,0,0
case;display;case;showcase;vitrine,255,0,0
pool;table;billiard;table;snooker;table,0,71,255
pillow,255,235,0
screen;door;screen,255,173,0
stairway;staircase,255,0,31
river,200,200,11
bridge;span,0,82,255
bookcase,245,255,0
blind;screen,255,61,0
coffee;table;cocktail;table,112,255,0
toilet;can;commode;crapper;pot;potty;stool;throne,133,255,0
flower,0,0,255
book,0,163,255
hill,0,102,255
bench,0,255,194
countertop,255,143,0
stove;kitchen;stove;range;kitchen;range;cooking;stove,0,255,51
palm;palm;tree,255,82,0
kitchen;island,41,255,0
computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system,173,255,0
swivel;chair,255,0,10
boat,0,255,173
bar,153,255,0
arcade;machine,0,92,255
hovel;hut;hutch;shack;shanty,255,0,255
bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle,245,0,255
towel,102,0,255
light;light;source,0,173,255
truck;motortruck,20,0,255
tower,184,184,255
chandelier;pendant;pendent,255,31,0
awning;sunshade;sunblind,61,255,0
streetlight;street;lamp,255,71,0
booth;cubicle;stall;kiosk,204,0,255
television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box,194,255,0
airplane;aeroplane;plane,82,255,0
dirt;track,255,10,0
apparel;wearing;apparel;dress;clothes,255,112,0
pole,255,0,51
land;ground;soil,255,0,0
bannister;banister;balustrade;balusters;handrail,255,122,0
escalator;moving;staircase;moving;stairway,163,255,0
ottoman;pouf;pouffe;puff;hassock,0,153,255
bottle,10,255,0
buffet;counter;sideboard,0,112,255
poster;posting;placard;notice;bill;card,0,255,143
stage,255,0,0
van,0,255,163
ship,0,235,255
fountain,170,184,8
conveyer;belt;conveyor;belt;conveyer;conveyor;transporter,255,0,133
canopy,92,255,0
washer;automatic;washer;washing;machine,255,0,184
plaything;toy,31,0,255
swimming;pool;swimming;bath;natatorium,255,184,0
stool,255,214,0
barrel;cask,112,0,255
basket;handbasket,0,255,92
waterfall;falls,255,224,0
tent;collapsible;shelter,255,224,112
bag,160,184,70
minibike;motorbike,255,0,163
cradle,255,0,153
oven,0,255,71
ball,163,0,255
food;solid;food,0,204,255
step;stair,143,0,255
tank;storage;tank,235,255,0
trade;name;brand;name;brand;marque,0,255,133
microwave;microwave;oven,235,0,255
pot;flowerpot,255,0,245
animal;animate;being;beast;brute;creature;fauna,122,0,255
bicycle;bike;wheel;cycle,0,245,255
lake,212,190,10
dishwasher;dish;washer;dishwashing;machine,0,255,214
screen;silver;screen;projection;screen,255,204,0
blanket;cover,255,0,20
sculpture,0,255,255
hood;exhaust;hood,255,153,0
sconce,255,41,0
vase,204,255,0
traffic;light;traffic;signal;stoplight,255,0,41
tray,0,255,41
ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin,255,0,173
fan,255,245,0
pier;wharf;wharfage;dock,255,0,71
crt;screen,255,0,122
plate,184,255,0
monitor;monitoring;device,255,92,0
bulletin;board;notice;board,0,255,184
shower,255,133,0
radiator,0,214,255
glass;drinking;glass,194,194,25
clock,0,255,102
flag,255,0,92

 */