//package com.bfr.opencvapp;
//
//import android.content.Context;
//import android.content.res.AssetManager;
//import android.graphics.Bitmap;
//import android.os.Bundle;
//import android.os.Environment;
//import android.os.RemoteException;
//import android.util.Log;
//import android.view.View;
//import android.view.WindowManager;
//import android.widget.Button;
//import android.widget.Toast;
//
//import com.bfr.opencvapp.cnn.CNNExtractorService;
//import com.bfr.opencvapp.cnn.impl.CNNExtractorServiceImpl;
//
//import org.opencv.android.BaseLoaderCallback;
//import org.opencv.android.CameraActivity;
//import org.opencv.android.CameraBridgeViewBase;
//import org.opencv.android.LoaderCallbackInterface;
//import org.opencv.android.OpenCVLoader;
//import org.opencv.android.Utils;
//import org.opencv.core.Mat;
//import org.opencv.core.Point;
//import org.opencv.core.Rect;
//import org.opencv.core.Scalar;
//import org.opencv.dnn.Net;
//import org.opencv.imgproc.Imgproc;
//import org.opencv.tracking.TrackerKCF;
//import org.opencv.video.Tracker;
//
//import com.bfr.usbservice.BodySensorData;
//import com.bfr.usbservice.HeadSensorData;
//import com.bfr.usbservice.IUsbAidlCbListner;
//import com.bfr.usbservice.MotorHeadData;
//import com.bfr.usbservice.MotorMotionData;
//import com.google.android.gms.tasks.OnSuccessListener;
//import com.google.android.gms.tasks.Task;
//import com.google.android.gms.tasks.Tasks;
//import com.google.mlkit.vision.common.InputImage;
//import com.google.mlkit.vision.face.Face;
//import com.google.mlkit.vision.face.FaceDetection;
//import com.google.mlkit.vision.face.FaceDetector;
//import com.google.mlkit.vision.face.FaceDetectorOptions;
//import com.google.mlkit.vision.face.FaceLandmark;
//
//import java.io.BufferedInputStream;
//import java.io.File;
//import java.io.FileOutputStream;
//import java.io.IOException;
//import java.text.SimpleDateFormat;
//import java.util.Collections;
//import java.util.Date;
//import java.util.List;
//import java.util.Vector;
//import java.util.concurrent.ExecutionException;
//import java.util.function.Consumer;
//
//
//public class DetectorActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
//
//    private static final String TAG = MainActivity.class.getName();
//
//    private static final String IMAGENET_CLASSES = "imagenet_classes.txt";
//    private static final String MODEL_FILE = "pytorch_mobilenet.onnx";
//
//    private CameraBridgeViewBase mOpenCvCameraView;
//    private Net opencvNet;
//
//    private CNNExtractorService cnnService;
//
//    // Neural net for detection
//    private Net net;
//    private boolean isnetloaded = false;
//
//    // Face detector
//    private FaceDetector faceDetector;
//
//    //button to start tracking
//    private Button initBtn;
//
//    //classes
//    private static final String[] classNames = {"background",
//            "aeroplane", "bicycle", "bird", "boat",
//            "bottle", "bus", "car", "cat", "chair",
//            "cow", "diningtable", "dog", "horse",
//            "motorbike", "person", "pottedplant",
//            "sheep", "sofa", "train", "tvmonitor"};
//
//
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
//        setContentView(R.layout.activity_main);
//
//        // initialize implementation of CNNExtractorService
//        this.cnnService = new CNNExtractorServiceImpl();
//        // configure camera listener
//        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
//        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
//        mOpenCvCameraView.setCvCameraViewListener(this);
//
//        // link to UI
//        initBtn = findViewById(R.id.initButton);
//
//        // Load model
//        // directory where the files are saved
//        String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();
//
//        String proto = dir + "/MobileNetSSD_deploy.prototxt";
//        String weights = dir + "/MobileNetSSD_deploy.caffemodel";
//        Toast.makeText(this, dir , Toast.LENGTH_SHORT).show();
////        net = Dnn.readNetFromCaffe(proto, weights);
////        net = cnnService.getConvertedNet("", TAG);
//        Log.i(TAG, "Network loaded successfully");
//
//        //start tracking button
//        initBtn.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Log.i("Tracking", "Tracking is starting");
//            }
//        });
//
//        // Real-time contour detection of multiple faces
//        FaceDetectorOptions options =
//                new FaceDetectorOptions.Builder()
//                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
//                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
//                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
//                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
//                        .enableTracking()
//                        .build();
//        FaceDetector detector = FaceDetection.getClient(options);
//        faceDetector = detector;
//
//    }
//
//    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
//        @Override
//        public void onManagerConnected(int status) {
//            switch (status) {
//                case LoaderCallbackInterface.SUCCESS: {
//                    Log.i(TAG, "OpenCV loaded successfully!");
//                    mOpenCvCameraView.enableView();
//                }
//                break;
//                default: {
//                    super.onManagerConnected(status);
//                }
//                break;
//            }
//        }
//    };
//
//    @Override
//    public void onResume() {
//        super.onResume();
//        // OpenCV manager initialization
//        OpenCVLoader.initDebug();
//        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
//    }
//
//
//    @Override
//    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
//        return Collections.singletonList(mOpenCvCameraView);
//    }
//
//    public void onCameraViewStarted(int width, int height) {
//        // obtaining converted network
//        String onnxModelPath = getPath(MODEL_FILE, this);
//        if (onnxModelPath.trim().isEmpty()) {
//            Log.i(TAG, "Failed to get model file");
//            return;
//        }
////        opencvNet = cnnService.getConvertedNet(onnxModelPath, TAG);
//
//        // Tracker init
////        mytracker = TrackerKCF.create();
//
//    }
//
//    // frame captured by camera
//    Mat frame;
//    // conversion to MLKit
//    InputImage inputImage;
//    Bitmap bitmapImage = null ;
//    // Face detector
//    Task result;
//    // List of detected faces
//    Vector<DetectedFace> foundFaces = new Vector<DetectedFace>();
//
//    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
//
//        // if not loaded
//        if (!isnetloaded)
//        {
//            // Load model
//            // directory where the files are saved
//            String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();
//
//            String proto = dir + "/MobileNetSSD_deploy.prototxt";
//            String weights = dir + "/MobileNetSSD_deploy.caffemodel";
////            runOnUiThread(new Runnable() {
////                @Override
////                public void run() {
////                    Toast.makeText(this, "coucou" , Toast.LENGTH_SHORT).show();
////                }
////            });
//
////            net = Dnn.readNetFromCaffe(proto, weights);
//            //net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
////            net.setPreferableTarget(Dnn.DNN_TARGET_OPENCL_FP16);
//
//            isnetloaded = true;
//        }
//
//
//        // Caapture Frame from camera
//        frame = inputFrame.rgba();
//        Log.i("MLKit", String.valueOf(System.currentTimeMillis()) + " Taking picture: ");
//        //Detecting Face
//
//        //convert to bitmap
//        bitmapImage = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(frame, bitmapImage);
//        //convert to InputImage
//        inputImage = InputImage.fromBitmap(bitmapImage, 0);
//        result = faceDetector.process(inputImage)
//                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
//                    @Override
//                    public void onSuccess(List<Face> faces) {
//                        Log.i("MLKit", "Found faces : " + String.valueOf(faces.size()) );
//                        // adjusting size
//                        if (foundFaces.size()<faces.size())
//                        {   // add elements
//                            do {
//                                foundFaces.add(new DetectedFace());
//                            } while (foundFaces.size()<faces.size());
//                        } // end if foundface size < i+1
//                        else if (foundFaces.size()>faces.size())
//                        {   // remove elements
//                            do {
//                                foundFaces.remove(0);
//                            } while (foundFaces.size()>faces.size());
//                        } // end if foundface size < i+1
//
//                        // for each detected face
//                        for (int i=0; i < faces.size(); i++) {
//                               // assiging values
//                            foundFaces.get(i).x1 = faces.get(i).getBoundingBox().left;
//                            foundFaces.get(i).y1 = faces.get(i).getBoundingBox().top;
//                            foundFaces.get(i).x2 = faces.get(i).getBoundingBox().right;
//                            foundFaces.get(i).y2 = faces.get(i).getBoundingBox().bottom;
//                            // tracking id
//                            foundFaces.get(i).trackingId = faces.get(i).getTrackingId();
//                            //classifications
//                            foundFaces.get(i).smilingProbability = faces.get(i).getSmilingProbability();
//                        Log.i("MLKit", String.valueOf(System.currentTimeMillis()) + " Face detected : "
//                                + String.valueOf(foundFaces.get(i).x1) +
//                                    " " + String.valueOf(foundFaces.get(i).y1)
//                                + " " + String.valueOf(foundFaces.get(i).x2)
//                                + " " + String.valueOf(foundFaces.get(i).y2) );
//
//                            } // next face
//
//                    } // end onSucess
//                }); // end process image
//
//        try {
//            Tasks.await(result);
//        } catch (ExecutionException e) {
//            e.printStackTrace();
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//
//        frame_count +=1;
//
//        // draw a rectangle around face
//        Log.i("MLKit", "Found faces : " + String.valueOf(foundFaces.size()) );
//        // for each found face
//        for (int i=0; i<foundFaces.size(); i++)
//        {
//            Log.i("MLKit", String.valueOf(System.currentTimeMillis()) + "Drawing face : "
//                    + String.valueOf(foundFaces.get(i).x1) +
//                    " " + String.valueOf(foundFaces.get(i).y1)
//                    + " " + String.valueOf(foundFaces.get(i).x2)
//                    + " " + String.valueOf(foundFaces.get(i).y2) );
//            Imgproc.rectangle(frame, new Point(foundFaces.get(i).x1, foundFaces.get(i).y1),
//                    new Point(foundFaces.get(i).x2,foundFaces.get(i).y2),
//                    new Scalar(255, 10, 10));
//            Imgproc.putText(frame, String.valueOf(foundFaces.get(i).trackingId) + "  "
//                    + String.valueOf(foundFaces.get(i).smilingProbability),
//                    new Point(foundFaces.get(i).x1, foundFaces.get(i).y1),2, 0.8, new Scalar(255,0 , 0));
//        }
////        Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2,y2), new Scalar(255, 10, 10));
////        Imgproc.putText(frame, String.valueOf(trackId) + "  " + String.valueOf(smilingProba), new Point(x1, y1),2, 0.8, new Scalar(255,0 , 0));
//
//
////        if (false)
////        {
////            // draw a rectangle
////            Imgproc.rectangle(frame, new Point(600, 100), new Point(700,200), new Scalar(255, 10, 10));
////
////            final int IN_WIDTH = 300;
////            final int IN_HEIGHT = 300;
////            final float WH_RATIO = (float)IN_WIDTH / IN_HEIGHT;
////            final double IN_SCALE_FACTOR = 0.007843;
////            final double MEAN_VAL = 127.5;
////            final double THRESHOLD = 0.75;
////
////
////            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
////
////            // atempt to find a person
////            // Forward image through network.
////            Mat blob = Dnn.blobFromImage(frame, IN_SCALE_FACTOR,
////                    new org.opencv.core.Size(IN_WIDTH, IN_HEIGHT),
////                    new Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), /*swapRB*/false, /*crop*/false);
////            net.setInput(blob);
////            Mat detections = net.forward();
////            int cols = frame.cols();
////            int rows = frame.rows();
////            detections = detections.reshape(1, (int)detections.total() / 7);
////            for (int i = 0; i < detections.rows(); ++i) {
////                double confidence = detections.get(i, 2)[0];
////                if (confidence > THRESHOLD) {
////                    int classId = (int)detections.get(i, 1)[0];
////                    int left   = (int)(detections.get(i, 3)[0] * cols);
////                    int top    = (int)(detections.get(i, 4)[0] * rows);
////                    int right  = (int)(detections.get(i, 5)[0] * cols);
////                    int bottom = (int)(detections.get(i, 6)[0] * rows);
////                    // Draw rectangle around detected object.
////                    Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
////                            new Scalar(0, 255, 0));
////                    String label = classNames[classId] + ": " + confidence;
////                    int[] baseLine = new int[1];
////                    org.opencv.core.Size labelSize = Imgproc.getTextSize(label, 2, 0.5, 1, baseLine);
////                    // Draw background for label.
////                    //Imgproc.rectangle(frame, new Point(left, top - labelSize.getHeight()),
////                    //        new Point(left + labelSize.getWidth(), top + baseLine[0]),
////                    //        new Scalar(255, 255, 255), Imgproc.FILLED);
////                    Imgproc.rectangle(frame, new Point(left, top - labelSize.height),
////                            new Point(left + labelSize.width, top + baseLine[0]),
////                            new Scalar(255, 255, 255));
////                    // Write class name and confidence.
//////                Imgproc.putText(frame, label, new Point(left, top),
////                    Imgproc.putText(frame, String.valueOf(classId), new Point(left, top),
////                            2, 0.8, new Scalar(255,0 , 0));
////                }   // end if confidence OK
////            } // next detection
////
////
////            // if found person
////            if (foundperson)
////            {
////                // if not tracking yet
////                if (!istracking)
////                {
////                    // init
////                    // init tracker on drawn rect
////                    Rect initBB = new Rect(600, 100, 100, 100);
////                    mytracker.init(frame, initBB);
////
////                    foundperson = false;
////                    istracking = true;
////                }
////                else // is tracking
////                {
////
////                }
////
////            }
////
////        } // end if modulo 10
////        else // rest of the frames
////        {
////            // if is tracking
////            if (istracking)
////            {
////                //Update tracker
////                mytracker.update(frame, tracked);
////
////                // draw a rectangle
////                Imgproc.rectangle(frame, new Point(tracked.x, tracked.y), new Point(tracked.x+tracked.width,tracked.y+tracked.height), new Scalar(0, 0, 255));
////            }
////        }
//
//
//        return frame;
//    } // end function
//
//    public void onCameraViewStopped() {
//    }
//
//    private static String getPath(String file, Context context) {
//        AssetManager assetManager = context.getAssets();
//        BufferedInputStream inputStream;
//        try {
//            // read the defined data from assets
//            inputStream = new BufferedInputStream(assetManager.open(file));
//            byte[] data = new byte[inputStream.available()];
//            inputStream.read(data);
//            inputStream.close();
//            // Create copy file in storage.
//            File outFile = new File(context.getFilesDir(), file);
//            FileOutputStream os = new FileOutputStream(outFile);
//            os.write(data);
//            os.close();
//            // Return a path to file which may be read in common way.
//            return outFile.getAbsolutePath();
//        } catch (IOException ex) {
//            Log.i(TAG, "Failed to upload a file");
//        }
//        return "";
//    } // end getpath
//
//
//
//
//}