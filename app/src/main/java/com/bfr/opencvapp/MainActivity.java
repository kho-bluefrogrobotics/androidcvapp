package com.bfr.opencvapp;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.Toast;

import com.bfr.opencvapp.cnn.CNNExtractorService;
import com.bfr.opencvapp.cnn.impl.CNNExtractorServiceImpl;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.TrackerKCF;
import org.opencv.video.Tracker;
import org.opencv.video.TrackerMIL;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;


public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getName();

    private static final String IMAGENET_CLASSES = "imagenet_classes.txt";
    private static final String MODEL_FILE = "pytorch_mobilenet.onnx";

    private CameraBridgeViewBase mOpenCvCameraView;
    private Net opencvNet;

    private CNNExtractorService cnnService;

    // Neural net for detection
    private Net net;
    private boolean isnetloaded = false;

    //button to start tracking
    private Button initBtn;
    private CheckBox trackingCheckbox;


    //classes
    private static final String[] classNames = {"background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};


    Tracker mytracker;
    Rect tracked = new Rect();
    private int frame_count = 0;
    private boolean foundperson =false;
    private boolean istracking = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // initialize implementation of CNNExtractorService
        this.cnnService = new CNNExtractorServiceImpl();
        // configure camera listener
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        // link to UI
        initBtn = findViewById(R.id.initButton);
        trackingCheckbox = findViewById(R.id.trackingBox) ;


        // Load model
        // directory where the files are saved
        String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();

        String proto = dir + "/MobileNetSSD_deploy.prototxt";
        String weights = dir + "/MobileNetSSD_deploy.caffemodel";
        Toast.makeText(this, dir , Toast.LENGTH_SHORT).show();
//        net = Dnn.readNetFromCaffe(proto, weights);
//        net = cnnService.getConvertedNet("", TAG);
        Log.i(TAG, "Network loaded successfully");

        initBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.i("Tracking", "Tracking is starting");
                foundperson = true;
            }
        });

    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully!");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    public void onResume() {
        super.onResume();
        // OpenCV manager initialization
        OpenCVLoader.initDebug();
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }


    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onCameraViewStarted(int width, int height) {
        // obtaining converted network
        String onnxModelPath = getPath(MODEL_FILE, this);
        if (onnxModelPath.trim().isEmpty()) {
            Log.i(TAG, "Failed to get model file");
            return;
        }
//        opencvNet = cnnService.getConvertedNet(onnxModelPath, TAG);

        // Tracker init
        mytracker = TrackerKCF.create();

    }


    Mat blob, detections;
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {


        // if not loaded
        if (!isnetloaded)
        {
            // Load model
            // directory where the files are saved
            String dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).toString();

            String proto = dir + "/opencv_face_detector.pbtxt";
            String weights = dir + "/opencv_face_detector_uint8.pb";
            net = Dnn.readNetFromTensorflow(weights, proto);

            isnetloaded = true;
        }
        Mat frame = inputFrame.rgba();

        frame_count +=1;

        // every xxx frame
        if (frame_count%5 == 0) {
            // draw a rectangle
            Imgproc.rectangle(frame, new Point(600, 100), new Point(700, 200), new Scalar(255, 10, 10));

            final int IN_WIDTH = 300;
            final int IN_HEIGHT = 300;
            final float WH_RATIO = (float) IN_WIDTH / IN_HEIGHT;
            final double IN_SCALE_FACTOR = 0.007843;
            final double MEAN_VAL = 127.5;
            final double THRESHOLD = 0.75;


            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

            blob = Dnn.blobFromImage(frame, 1.0,
                    new org.opencv.core.Size(300, 300),
                    new Scalar(104, 117, 123), /*swapRB*/true, /*crop*/false);
            net.setInput(blob);
            detections = net.forward();
            int cols = frame.cols();
            int rows = frame.rows();
            detections = detections.reshape(1, (int) detections.total() / 7);

            // If found faces
            if (detections.rows() > 0) {
                // only if currently tracking something
                if (istracking) {
                    // find id of closest face
                    int id_closest=0;
                    int max_dist = 999999;
                    int dist;
                    // for each face
                    for (int i = 0; i < detections.rows(); ++i) {
                        double confidence = detections.get(i, 2)[0];
                        if (confidence > THRESHOLD) {
                            int left = (int) (detections.get(i, 3)[0] * cols);
                            int top = (int) (detections.get(i, 4)[0] * rows);
                            int right = (int) (detections.get(i, 5)[0] * cols);
                            int bottom = (int) (detections.get(i, 6)[0] * rows);

                            // if dist min
                            dist = Math.abs(left - tracked.x) + Math.abs(top - tracked.y);
                            if (dist < max_dist) {
                                // update
                                max_dist = dist;
                                id_closest = i;
                            }

                        } // end if confidence OK
                    } // next face

                    // Init tracker on closest face
                    // Init tracker on first face
                    int left = (int) (detections.get(id_closest, 3)[0] * cols);
                    int top = (int) (detections.get(id_closest, 4)[0] * rows);
                    int right = (int) (detections.get(id_closest, 5)[0] * cols);
                    int bottom = (int) (detections.get(id_closest, 6)[0] * rows);
                    Rect bbox = new Rect((int) left,
                            top,
                            right-left,
                            bottom-top
                    );
                    Log.i("Tracking", "New Init on " +  bbox.x + " " + bbox.y);
                    mytracker.init(frame, bbox);

                    //DRAW
                    for (int i = 0; i < detections.rows(); ++i) {
                        double confidence = detections.get(i, 2)[0];
                        if (confidence > THRESHOLD) {
                            int classId = (int) detections.get(i, 1)[0];
                            left = (int) (detections.get(i, 3)[0] * cols);
                            top = (int) (detections.get(i, 4)[0] * rows);
                            right = (int) (detections.get(i, 5)[0] * cols);
                            bottom = (int) (detections.get(i, 6)[0] * rows);
                            // Draw rectangle around detected object.
                            Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                                    new Scalar(0, 255, 0));
                            String label = classNames[classId] + ": " + confidence;
                            int[] baseLine = new int[1];
                            org.opencv.core.Size labelSize = Imgproc.getTextSize(label, 2, 0.5, 1, baseLine);
                        } // end if confidence
                    } // next face
                    // end DRAW


                } // end if is tracking
                else // Not tracking yet
                {
                    // Init tracker on first face
                    int left = (int) (detections.get(0, 3)[0] * cols);
                    int top = (int) (detections.get(0, 4)[0] * rows);
                    int right = (int) (detections.get(0, 5)[0] * cols);
                    int bottom = (int) (detections.get(0, 6)[0] * rows);
                    Rect bbox = new Rect((int) left,
                            top,
                            right-left,
                            bottom-top
                    );
                    Log.i("Tracking", "First Init on " +  bbox.x + " " + bbox.y);
                    mytracker.init(frame, bbox);
                    //set status
                    istracking = true;
                }
            } // end if face found

        } // end if everyxxx frame
        else
        {
            // if is tracking
            if (istracking)
//            if (false)
            {
                Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
                // update the tracker
                Log.i("Tracking", "channels "+ String.valueOf(frame.channels()) );
                //Update tracker
//                Rect bbox = new Rect(600, 100, 100, 100);
                mytracker.update(frame, tracked);

//                //record
//                tracked.x = bbox.x;
//                tracked.y = bbox.y;
//                tracked.width = bbox.width;
//                tracked.height = bbox.height;

                Log.i("Tracking", "Tracker updated " + tracked.x + " " + tracked.y);
                // draw a rectangle
                Imgproc.rectangle(frame, new Point(tracked.x, tracked.y), new Point(tracked.x+10,tracked.y+10), new Scalar(0, 0, 255));
            } // end if is tracking
        } // end rest of the frames


//
//
//                if (trackingCheckbox.isChecked()) {
//                    foundperson = true;
//                    trackingCheckbox.setChecked(false);
//                }
//
//
//
//
//            }   // end if confidence OK
//
//        } // next detection
//
//        // set
//
//
//
//
//        if (istracking)
//        {
//            Log.i("Tracking", "channels "+ String.valueOf(frame.channels()) );
//            //Update tracker
//            Rect bbox = new Rect();
//            mytracker.update(frame, bbox);
//            Log.i("Tracking", "Tracker updated " + bbox.x + " " + bbox.y);
//
//            // draw a rectangle
//            Imgproc.rectangle(frame, new Point(bbox.x, bbox.y), new Point(bbox.x+bbox.width,bbox.y+bbox.height), new Scalar(0, 0, 255));
//        }
//        else // not tracking yet
//        {
//            // if found person
//            if (detections.rows()>0)
//            {
//                // init tracker on drawn rect
//                Rect initBB = new Rect(600, 100, 100, 100);
//                mytracker.init(frame, initBB);
//
//                foundperson = false;
//                istracking = true;
//            }
//        }
//
////
//
//        } // end if modulo 10


        return frame;
    } // end function

    public void onCameraViewStopped() {
    }

    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        try {
            // read the defined data from assets
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

}