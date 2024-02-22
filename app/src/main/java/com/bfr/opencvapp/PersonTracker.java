package com.bfr.opencvapp;

import static com.bfr.opencvapp.utils.Utils.Color.*;
import static com.bfr.opencvapp.utils.Utils.matToBitmapAndResize;
import static com.bfr.opencvapp.utils.Utils.modelsDir;

import android.graphics.Bitmap;
import android.util.Log;


import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseDetector;
import com.google.mlkit.vision.pose.PoseLandmark;
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.KAZE;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.legacy_TrackerMOSSE;
import org.opencv.video.TrackerVit;
import org.opencv.video.TrackerVit_Params;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PersonTracker {

    private final String TAG = "_VISION Person tracker";

    // Person detector
    MultiDetector detector;

    TfLiteBlazePose poseEstimator;

    // List of detected person or faces or hands
    ArrayList<MultiDetector.Recognition> detections = new ArrayList<MultiDetector.Recognition>();
    // Blob
    Mat blob;
    Size frameResize = new Size(250,250);
    Scalar frameMeanPers = new Scalar(127.5, 127.5, 127.5);
    private int frameRows, frameCols;
    // detection result
    double confidence = 0;
    // thres for detection
    public double THRESHOLD=0.6;
    public int frameCount =0;
    private int _FRAME_DETECT = 0;
    private boolean ShouldDetectFace = false;
    private int INTERVAL_MIN = 0;
    private int INTERVAL_MAX = 0;


    //
//    private boolean isTracking = false;
    private double maxDist, dist;
    private int classOfClosestDetection;

    // Tracker
    TrackerVit vitTracker;
    TrackerVit_Params vitTrackerparams ;
    // Human Tracking
    final String VITTRACKMODEL = modelsDir + "object_tracking_vittrack_2023sep.onnx";
    //Mosse Tracker
    legacy_TrackerMOSSE mosseTracker;
    Rect2d mosseTracked=new Rect2d();
    // resize to be faster
    Mat smallFrame;
//    Size smallSize = new Size(320, 240);
    Size smallSize = new Size(1024, 768);

    // Tracked target
    public TrackedObject tracked = new TrackedObject() ;
//    public Rect tracked=new Rect();
    public boolean trackingSuccess = false;
//    public float targetLeftPos, targetRightPos, targetTopPos, targetBottomPos = 0.0f;

    // to display
    public Mat displayMat;
    public boolean readyToDisplay = false;
    private Point pt1 = new Point(0, 0);
    private Point pt2 = new Point(0, 0);

    String saveFolder="";

    float IOU_THRES = 0.7f;
    float OVERLAPRATIO_THRES = 0.8f;

    // hard limit for human height to track (in pixel)
    int MIN_HUMAN_HEIGHT = 120;

    // log debug
    private boolean debugLog =true;
    ArrayList<Float> scoreHistory = new ArrayList<Float>();
    final float NUM_OF_SCORE_HISTORY = 5;
    float avscore = 0.0f;

    PoseDetectorOptions poseDetectoptions;
    PoseDetector poseDetector;
    Pose mypose;

    public PersonTracker(MultiDetector personDetector, TfLiteBlazePose blazePose){
        // Load model for human detector
        detector = personDetector;

        poseEstimator = blazePose;

        Log.d(TAG, "Person detector model created"  ) ;
        displayMat = new Mat();
        smallFrame = new Mat();
        frameCols = (int)smallSize.width;
        frameRows = (int)smallSize.height;//

        //init ViT Tracker
        vitTrackerparams = new TrackerVit_Params();
        vitTrackerparams.set_net(VITTRACKMODEL);
        vitTracker = TrackerVit.create(vitTrackerparams);

        // Base pose detector with streaming frames, when depending on the pose-detection sdk
        poseDetectoptions =
                new PoseDetectorOptions.Builder()
                        .setDetectorMode(PoseDetectorOptions.SINGLE_IMAGE_MODE)
                        .setPreferredHardwareConfigs(PoseDetectorOptions.CPU_GPU)
                        .build();

        poseDetector = PoseDetection.getClient(poseDetectoptions);

    }

    double fps = 0.0;

    public void visualTracking(Mat frame, boolean fastTracking, boolean constructVisualizationImage)
    {

//        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
        try {

            // Resize for better performances
//            Imgproc.resize(frame, smallFrame, smallSize);
            smallFrame = frame.clone();
            // scale coeff reduction

            double xScale = (double)frame.width() / smallFrame.width();
            double yScale =  (double)frame.height()/ smallFrame.height();

            if (fastTracking)
            {     // dectect every xxx frame
                _FRAME_DETECT = 25;
            }
            else
            {// dectect every xxx frame
                _FRAME_DETECT = 30000000;
            }


//            if(debugLog)
//                Log.d(TAG, "\n \n                   FRAME NUMBER: " + frameCount ) ;

            /*** First,  init on object detection*/
            if(frameCount == 0)
            {
                if(debugLog)
                    Log.d(TAG, "\n\n****************************************************\n" +
                            "Init tracker " ) ;

                // Detection
                detections = detector.recognizeImage(
                        matToBitmapAndResize(frame, 320, 320),
                        0.5f, 0.6f, 99.0f, frame);

                if (detections.size() > 0) {
                    /*** Look for first detected face */
                    int detectedFaceId = -1;
                    // for each detection
                    for (int i = 0; i < detections.size(); ++i) {

                        if ((detections.get(i).getDetectedClass()==1)) { // if is a face
                            // save index
                            detectedFaceId = i;
                            break;
                        } // end if confidence OK
                    } // next detection

                    // By default, init on first detected face
                    if(detectedFaceId>=0) {
                        Log.w(TAG, "Init on first face: " + detectedFaceId ) ;
                        tracked.box = initTracker(vitTracker, detections.get(detectedFaceId));
                        tracked.objectClass = detections.get(detectedFaceId).getDetectedClass(); // 0:human, 1:face
                        tracked.score = vitTracker.getTrackingScore();
                    }
                    else // no face found -> init on first detection, hopefully a human silouhette
                    {
                        Log.w("coucou", "Init on first detection: " + detectedFaceId ) ;
                        tracked.box = initTracker(vitTracker, detections.get(0));
                        tracked.objectClass = detections.get(0).getDetectedClass();
                        tracked.score = vitTracker.getTrackingScore();
                    }

                    // set
                    trackingSuccess = true;
                    //increment frame count
                    frameCount +=1;

                }
                else // nothing detected in front of the camera
                    Log.w(TAG, "Nothing detected");
            }

            /*** Else: is already tracking */
            else
            {
                // Reset tracking only every xxx frames of in case of tracking lost
//                if (frameCount %_FRAME_DETECT == 0 || !trackingSuccess ) {
                if (false ) {

                    if(debugLog)
                        Log.d(TAG, "\n\n****************************************************\n" +
                                "Checking if need to reset  trackingsucess =" + trackingSuccess + " frameNum="+frameCount ) ;

                    //reset in any case
                    frameCount = 1;
//                    trackingSuccess = false;
                    long mlkitTime = System.currentTimeMillis();


//                    //convert to bitmap
//                    Bitmap bitmapImage = Bitmap.createBitmap(smallFrame.cols(), smallFrame.rows(), Bitmap.Config.ARGB_8888);
//                    Utils.matToBitmap(smallFrame, bitmapImage);
//
//                    InputImage inputImage = InputImage.fromBitmap(bitmapImage, 0);
//
//
//                    Task<Pose> result =
//                            poseDetector.process(inputImage)
//                                    .addOnSuccessListener(
//                                            new OnSuccessListener<Pose>() {
//                                                @Override
//                                                public void onSuccess(Pose pose) {
//                                                    // Task completed successfully
//                                                    // ...
//
//                                                    mypose = pose;
//                                                }
//                                            })
//                                    .addOnFailureListener(
//                                            new OnFailureListener() {
//                                                @Override
//                                                public void onFailure(@NonNull Exception e) {
//                                                    // Task failed with an exception
//                                                    // ...
//                                                }
//                                            });
//
//                    try {
//                        Tasks.await(result);
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                    }
//
//                    // Get all PoseLandmarks. If no person was detected, the list will be empty
//                    List<PoseLandmark> allPoseLandmarks = mypose.getAllPoseLandmarks();
//
//                    PoseLandmark nose = mypose.getPoseLandmark(PoseLandmark.NOSE);
//                    PoseLandmark leftEar = mypose.getPoseLandmark(PoseLandmark.LEFT_EAR);
//                    PoseLandmark rightEar = mypose.getPoseLandmark(PoseLandmark.RIGHT_EAR);
//                    PoseLandmark leftShoulder = mypose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER);
//                    PoseLandmark rightShoulder = mypose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER);
//                    PoseLandmark leftHip = mypose.getPoseLandmark(PoseLandmark.LEFT_HIP);
//                    PoseLandmark rightHip = mypose.getPoseLandmark(PoseLandmark.RIGHT_HIP);
//
//                    Log.w("coucouMLKit", "MLKit elapsed time : "+ (System.currentTimeMillis()-mlkitTime)
//                    // display position in pixels
//                            +"\n Position=" + nose.getPosition().x + "," + nose.getPosition().y );
//
//                    int left = (int) Math.min (leftShoulder.getPosition().x, rightShoulder.getPosition().x);
//                    int right = (int) Math.max(leftShoulder.getPosition().x, rightShoulder.getPosition().x);
//                    int top = (int) leftShoulder.getPosition().y;
//                    int bottom = (int) leftHip.getPosition().y;
//                    int height = Math.abs(bottom-top);
//                    int width = Math.abs(right-left);
//
//                    Log.i(TAG, "Target : " + left + " " + top + " "+
//                            Math.max(1,
//                                    top - (int)(height/4)) + " " + width + " " + height);
//                    top = Math.max(1,
//                            top - (int)(height/2));
//
//                    //
//
//                    Rect targetBox = new Rect(
//                            left,
//                            top,
//                            width , // width
//                            height); // height


//                    checkAndResetTracking(vitTracker, tracked, targetBox);
                    checkAndResetTracking(vitTracker, tracked, null);
                    trackingSuccess = true;
                    frameCount +=1;

                }
                else  // other frames -> only tracking
                {

                    // if is tracking
                    if (true)
                    {
                        //Update tracker
                            if (fastTracking) {
                                trackingSuccess = mosseTracker.update(smallFrame, mosseTracked);
                            }
                            else{
//                            Log.w(TAG, "UPDATE VIT tracker ") ;
                                vitTracker.update(smallFrame, tracked.box);
                                tracked.score = vitTracker.getTrackingScore();
                                if(computeTrackingScore(scoreHistory)>=0.4f)
                                    trackingSuccess=true;
                                else
                                    trackingSuccess = false;
                            }


                            int left = tracked.box.x;
                            int top = tracked.box.y;
                            int right = tracked.box.x+tracked.box.width;
                            int bottom = tracked.box.y+tracked.box.height;
                            Rect toCrop = new Rect(
                                    left,
                                    top,
                                    right-left-5,
                                    bottom-top-10
                            );
                            //convert to bitmap
                            Mat croppedTargetMat = frame.submat(toCrop);
//                            Imgproc.resize(croppedTargetMat, croppedTargetMat, new Size(256,256));
                            Bitmap bitmapImage = Bitmap.createBitmap(croppedTargetMat.cols(), croppedTargetMat.rows(), Bitmap.Config.ARGB_8888);
                            Utils.matToBitmap(croppedTargetMat, bitmapImage);




                        try (FileOutputStream out = new FileOutputStream("/sdcard/Download/coucou.jpg")) {
                            bitmapImage.compress(Bitmap.CompressFormat.JPEG, 100, out); // bmp is your Bitmap instance
                            Imgcodecs.imwrite("/sdcard/Download/coucoumat.jpg", croppedTargetMat);
                            // PNG is a lossless format, the compression factor (100) is ignored
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

//
//                        /*** MLKit for pose*/
//                    InputImage inputImage = InputImage.fromBitmap(bitmapImage, 0);
//                    Task<Pose> result =
//                            poseDetector.process(inputImage)
//                                    .addOnSuccessListener(
//                                            new OnSuccessListener<Pose>() {
//                                                @Override
//                                                public void onSuccess(Pose pose) {
//                                                    // Task completed successfully
//                                                    // ...
//
//                                                    mypose = pose;
//                                                }
//                                            })
//                                    .addOnFailureListener(
//                                            new OnFailureListener() {
//                                                @Override
//                                                public void onFailure(@NonNull Exception e) {
//                                                    // Task failed with an exception
//                                                    // ...
//                                                }
//                                            });
//
//                    try {
//                        Tasks.await(result);
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                    }
//
//                    // Get all PoseLandmarks. If no person was detected, the list will be empty
//                    List<PoseLandmark> allPoseLandmarks = mypose.getAllPoseLandmarks();
//
//                    PoseLandmark nose = mypose.getPoseLandmark(PoseLandmark.NOSE);
//                    PoseLandmark leftEye = mypose.getPoseLandmark(PoseLandmark.LEFT_EYE);
//                    PoseLandmark rightEye = mypose.getPoseLandmark(PoseLandmark.RIGHT_EYE);
//                    PoseLandmark leftEar = mypose.getPoseLandmark(PoseLandmark.LEFT_EAR);
//                    PoseLandmark rightEar = mypose.getPoseLandmark(PoseLandmark.RIGHT_EAR);
//                    PoseLandmark leftShoulder = mypose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER);
//                    PoseLandmark rightShoulder = mypose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER);
//                    PoseLandmark leftHip = mypose.getPoseLandmark(PoseLandmark.LEFT_HIP);
//                    PoseLandmark rightHip = mypose.getPoseLandmark(PoseLandmark.RIGHT_HIP);
//
//                    Log.w("coucouMLKit", "MLKit elapsed time : "+ (System.currentTimeMillis())
//                    // display position in pixels
//                            +"\n Position=" + nose.getPosition().x + "," + nose.getPosition().y );
//
//                            int noseX = (int) nose.getPosition().x;
//                            int noseY = (int) nose.getPosition().x;
//                            int leftEyeX = (int) leftEye.getPosition().x;
//                            int leftEyeY = (int) leftEye.getPosition().y;
//                            int rightEyeX = (int) rightEye.getPosition().x;
//                            int rightEyeY = (int) rightEye.getPosition().y;
//                            int leftShoulderX = (int) leftShoulder.getPosition().x;
//                            int leftShoulderY = (int) leftShoulder.getPosition().y;
//                            int rightShoulderX = (int) rightShoulder.getPosition().x;
//                            int rightShoulderY = (int) rightShoulder.getPosition().y;
//                            int leftHipX = (int) leftHip.getPosition().x;
//                            int leftHipY = (int) leftHip.getPosition().y;
//                            int rightHipX = (int) rightHip.getPosition().x;
//                            int rightHipY = (int) rightHip.getPosition().y;
//                            //Log.w(TAG, "Coords : "+ shoulderX + " " +  shoulderY );
//
//                            Imgproc.circle(frame, new Point(
//                                    left + noseX, top + noseY), 5, new Scalar(0,255,0), 10);
//                            Imgproc.circle(frame, new Point(
//                                    left + leftEyeX, top + leftEyeY), 5, new Scalar(0,255,0), 10);
//                            Imgproc.circle(frame, new Point(
//                                    left + rightEyeX, top + rightEyeY), 5, new Scalar(0,255,0), 10);
//
//                            Imgproc.circle(frame, new Point(
//                                    left + leftShoulderX, top + leftShoulderY), 5, new Scalar(0,255,0), 10);
//                            Imgproc.circle(frame, new Point(
//                                    left + rightShoulderX, top + rightShoulderY), 5, new Scalar(0,255,0), 10);
//
//                            Imgproc.circle(frame, new Point(
//                                    left + leftHipX, top + leftHipY), 5, new Scalar(0,255,0), 10);
//                            Imgproc.circle(frame, new Point(
//                                    left + rightHipX, top + rightHipY), 5, new Scalar(0,255,0), 10);



//                            float[][] result =poseEstimator.recognizeImage(bitmapImage);
//
//                            // Blaze pose returns the x, y coords as values [0:255] independent from the input resolution
//                            int noseX = (int) (result[0][0*5] * (right-left)/255);
//                            int noseY = (int) (result[0][0*5+1] * (bottom-top)/255);
//                            int leftEyeX = (int) (result[0][2*5] * (right-left)/255);
//                            int leftEyeY = (int) (result[0][2*5+1] * (bottom-top)/255);
//                            int rightEyeX = (int) (result[0][5*5] * (right-left)/255);
//                            int rightEyeY = (int) (result[0][5*5+1] * (bottom-top)/255);
//                            int leftShoulderX = (int) (result[0][11*5] * (right-left)/255);
//                            int leftShoulderY = (int) (result[0][11*5+1] * (bottom-top)/255);
//                            int rightShoulderX = (int) (result[0][12*5] * (right-left)/255);
//                            int rightShoulderY = (int) (result[0][12*5+1] * (bottom-top)/255);
//                            int leftHipX = (int) (result[0][23*5] * (right-left)/255);
//                            int leftHipY = (int) (result[0][23*5+1] * (bottom-top)/255);
//                            int rightHipX = (int) (result[0][24*5] * (right-left)/255);
//                            int rightHipY = (int) (result[0][24*5+1] * (bottom-top)/255);
//                            //Log.w(TAG, "Coords : "+ shoulderX + " " +  shoulderY );
//
//                            Imgproc.circle(frame, new Point(
//                                    left + noseX, top + noseY), 5, new Scalar(0,255,0), 10);
//                            Imgproc.circle(frame, new Point(
//                                    left + leftEyeX, top + leftEyeY), 5, new Scalar(0,255,0), 10);
//                            Imgproc.circle(frame, new Point(
//                                    left + rightEyeX, top + rightEyeY), 5, new Scalar(0,255,0), 10);
//
//                            Imgproc.circle(frame, new Point(
//                                    left + leftShoulderX, top + leftShoulderY), 5, new Scalar(0,255,0), 10);
//                            Imgproc.circle(frame, new Point(
//                                    left + rightShoulderX, top + rightShoulderY), 5, new Scalar(0,255,0), 10);
//
//                            Imgproc.circle(frame, new Point(
//                                    left + leftHipX, top + leftHipY), 5, new Scalar(0,255,0), 10);
//                            Imgproc.circle(frame, new Point(
//                                    left + rightHipX, top + rightHipY), 5, new Scalar(0,255,0), 10);


//                        if(debugLog)
//                            Log.d("Tracking", "Tracker updated " + tracked.box.x + " " + tracked.box.y + "score=" + tracked.score );
//
                        // if tracking successfull
                        if (trackingSuccess)
                        {
//                            // register result
//                            if (fastTracking) {
//                                targetLeftPos = Math.max(0.0f, (float)mosseTracked.x/smallFrame.width());
//                                targetRightPos = Math.min(1.0f,(float)(mosseTracked.x + mosseTracked.width)/smallFrame.width());
//                                targetTopPos = Math.max(0.0f,(float)mosseTracked.y/smallFrame.height());
//                                targetBottomPos =  Math.min(1.0f,(float)(mosseTracked.y + mosseTracked.height)/smallFrame.height());
//                            }
//                            else{
//                                targetLeftPos = Math.max(0.0f, (float)tracked.box.x/smallFrame.width());
//                                targetRightPos = Math.min(1.0f,(float)(tracked.box.x + tracked.box.width)/smallFrame.width());
//                                targetTopPos = Math.max(0.0f,(float)tracked.box.y/smallFrame.height());
//                                targetBottomPos =  Math.min(1.0f,(float)(tracked.box.y + tracked.box.height)/smallFrame.height());
//                            }

//                            if(debugLog)
//                                Log.d(TAG, "opencv Tracking result:  " +  trackingSuccess
//                                        +" "+tracked.box.x +"," + tracked.box.y
//                                        + " " + tracked.box.height + "x"+ tracked.box.width );

//                                        + "\n" +  targetLeftPos + " "+ targetRightPos +" "+ targetTopPos+ " "+targetBottomPos
//                                        + "\n" + (float) tracked.box.x/frame.width()
//                                        + " "+ (float)(tracked.box.x + tracked.box.width)/frame.width()
//                                        +" "+ (float)tracked.box.y/frame.height()
//                                        + " "+(float)(tracked.box.y + tracked.box.height)/frame.height());

//                            if(constructVisualizationImage && !readyToDisplay) {

                        }

                        //increment frame count
                        frameCount +=1;

                    } // end if is tracking
                } // end rest of the frames



            } // end if First image or Already initialized

            if(true) {

                displayMat = displayResult(frame, tracked);
                // set flag for handshake with displaying service
                readyToDisplay = true;
            }

        }
        catch (Exception e) {
            Log.d(TAG, "tracking ERROR "+ Log.getStackTraceString(e) ) ;
            trackingSuccess = false;
        }


    } //end visual Tracking




    /**
     * Display tracked bounding box
     * @param frame the original frame
//     * @param score the confidence score of the tracker
     * @return a Mat dislpaying the tracked bounding box
     */
    public Mat displayResult(Mat frame, TrackedObject tracked)
    {
        try
        {
//            if (debugLog)
//                Log.d(TAG, "preparing display mat ");
            Mat displayMat = frame.clone();

            // draw a rectangle around Target
            pt1.x = (int) (tracked.box.x);
            pt1.y = (int) (tracked.box.y);
            pt2.x = (int) ( (tracked.box.x + tracked.box.width));
            pt2.y = (int) ( (tracked.box.y + tracked.box.height) );


            Imgproc.rectangle(displayMat, pt1, pt2,
                    _WHITE, 4);
            Imgproc.rectangle(displayMat, pt1, pt2,
                    _RED, 2);
            Imgproc.putText(displayMat, "Tracking [" + String.format(java.util.Locale.US, "%.3f", avscore) + "]", pt1,
                    2, 1, _BLACK, 5);
            Imgproc.putText(displayMat, "Tracking [" + String.format(java.util.Locale.US, "%.3f", avscore) + "]", pt1,
                    2, 1, _GREEN, 2);


//            Imgproc.putText(displayMat, "COUCOU", new Point(300, 300),
//                    2, 10, _RED, 20);

//            if(debugLog)
//                Log.d(TAG, "Found an overlapping object with same class IoU="+iou+"\n"
//                        +  detectionBbox.x
//                        + " " + detectionBbox.y
//                        + " " + detectionBbox.height
//                        + " " + detectionBbox.width);

            return displayMat;
        } catch (Exception e) {
            Log.e(TAG, "ERROR TRACKING: " + Log.getStackTraceString(e));
            return  null;
        }


    }

    /**
     * init tracker on a bounding box. In our case the bounding box corresponds to a human detection from the SSD model (cf MultiDetector)
     * @param tracker the used tracker
     * @param detection the bounding box the tracker has to track from now on
     * @return a Mat dislpaying the tracked bounding box
     */
    private Rect initTracker(Object tracker,MultiDetector.Recognition detection)
    {
        Rect tracked = new Rect();

        // Bbox of the detection
        Rect detectionBbox = new Rect((int) (detection.left * frameCols), (int)(detection.top*frameRows),
                (int) ((detection.right-detection.left) * frameCols) , // width
                (int) ((detection.bottom-detection.top) * frameRows)); // height

//        // Bounding box to track
//        tracked.x = (int) (detection.left * frameCols);
//        tracked.y= (int) (detection.top * frameRows);
//        tracked.width =  (int) (detection.right * frameCols) - (int) (detection.left * frameCols);
//        tracked.height = (int) (detection.bottom * frameRows) - (int) (detection.top * frameRows);
//
//        // Adjusting whether we init on a face or a human silhouette
//        if (detection.getDetectedClass() == 0) // Human
//        {
//            // crop extra area
//            tracked.x = (int) (detection.left * frameCols + tracked.width/4 );
//            tracked.width = (int)(tracked.width- (tracked.width/2) );
//            tracked.y= (int) (detection.top * frameRows + tracked.height/16);
//            // track the 2/3 upper part  or minimum arbitrary value
//            tracked.height = Math.max( (int) (0.5 * tracked.height),  70 );
//        }
//        else // Face
//        {
//            //TODO : see if necessary
//            // crop extra area
//            tracked.y = (int)(tracked.y + 0.1*tracked.height);
//            //tracked.height = Math.max( (int) (0.75 * tracked.height),  10 );
//        }


        // crop extra area wether it is a face or a human
        tracked = cropExtraArea(detectionBbox, detection.getDetectedClass());

        if(BuildConfig.DEBUG)
            Log.d("coucou", "Init the tracker class= " + detection.getDetectedClass() + "\n"
                    + " x=" + tracked.x
                    + " y=" + tracked.x
                    + " height=" + tracked.height
                    + " width=" + tracked.width) ;

        // Init the tracker on the detection
        // if the tracking method is MOSSE (very fast medium accurate)
        if( tracker instanceof legacy_TrackerMOSSE)
        {
            if(BuildConfig.DEBUG)
                Log.d(TAG, "Init MOSSETracker");

            mosseTracker = legacy_TrackerMOSSE.create();
            mosseTracker.init(smallFrame, new Rect2d((double)tracked.x,
                    (double)tracked.y,
                    (double)tracked.width,
                    (double)tracked.height
            ));
        }
        else if(tracker instanceof TrackerVit) // if the tracking method is VisualTransformer-based (fast & accurate)
        {
            if(BuildConfig.DEBUG)
                Log.d("coucou", "Init ViTTracker");

            vitTracker.init(smallFrame, tracked);
        }
        else if(tracker instanceof KAZE) // if the tracking method is VisualTransformer-based (fast & accurate)
        {

        }
        else
        {
            //Throw error
        }

        trackingSuccess = true;

        return tracked;

    }// end Init tracker


    /**
     Get closest detection to the current tracked bbox
     */
    private int getClosestDetection(Rect tracked, ArrayList<MultiDetector.Recognition> detections)
    {
        int idClosest = 0;

        Point trackedCentroid = getCentroid(tracked.x, // upper left corner x
                tracked.y, // upper left corner y
                tracked.height, // height
                tracked.width); // width

        // for each detection
        for (int i = 0; i < detections.size(); ++i) {

            if (detections.get(i).getConfidence() > THRESHOLD // if object detected with enough confidence
                    || (detections.get(i).getDetectedClass()==1 && detections.get(i).getConfidence()>=0.5)) { // or is a face so that we lower the confidence
//                //left
//                pt1.x = (int) (detections.get(i).left * frameCols);
//                //top
//                pt1.y = (int) (detections.get(i).top * frameRows);

                Point detectionCentroid = getCentroid((int) (detections.get(i).left * frameCols), // upper left corner x
                        (int) (detections.get(i).top * frameCols), // upper left corner y
                        (int) ((detections.get(i).bottom-detections.get(i).right) * frameRows) , // height
                        (int) ((detections.get(i).right-detections.get(i).left) * frameCols)); // width
                //if istracking whole body, check if face detected inside the target
//                if (isTrackingAPerson)
                if (false)
                {
                    //if detection is a face
                    if(detections.get(i).getDetectedClass()==1) {
                        // check if inside the tracked boundingbox
                        if ((detections.get(i).left * frameCols) >= tracked.x && (detections.get(i).right * frameCols <= tracked.x + tracked.width)) {
                            idClosest = i;
                            classOfClosestDetection = detections.get(i).getDetectedClass();
                            break;
                        }//end if inside tracked bbox
                    } // end if detection is a face
                } // endif is tracking a whole body

                // find the closest detection to the tracking position
                // L1 distance to optimize computing time
                dist =  (Math.abs(pt1.x - tracked.x) + Math.abs(pt1.y - tracked.y));
                if (dist < maxDist) {
                    // update
                    maxDist = dist;
                    idClosest = i;
                    classOfClosestDetection = detections.get(i).getDetectedClass();
                }

            } // end if confidence OK
        } // next detection

        return idClosest;
    } //end getClosest


    /**
     * Crop out the extra area to be easier to track
     * @param tracked
     * @param detectedClass
     * @return a smaller area to track
     */
    private Rect cropExtraArea(Rect tracked, int detectedClass)
    {
        Rect croppedArea = tracked;


        // if we deal with a human silhouette
        // Adjusting whether we init on a face or a human silhouette
        if (detectedClass == 0) // Human
        {
            // crop extra area
//            croppedArea.x = (int) (tracked.x + tracked.width/4 );
//            croppedArea.width = (int)(tracked.width- (tracked.width/2) );
//            croppedArea.y= (int) (tracked.y + tracked.height/16);
//            // track the 2/3 upper part  or minimum arbitrary value
//            croppedArea.height = Math.max( (int) (0.5 * tracked.height), MIN_HUMAN_HEIGHT  );

        }
        else if (detectedClass == 1) // Face
        {
            //TODO : see if necessary
            // crop extra area
            croppedArea.y = (int)(tracked.y + 0.1*tracked.height);
            //tracked.height = Math.max( (int) (0.75 * tracked.height),  10 );
        }
        else
        {
            croppedArea = tracked;
        }
        // else, a face


        return croppedArea;
    }
    /**
     Get the centroid of a bbox (from upper left corner coordinates and height/width)
     */
    private Point getCentroid(int x, int y, int height, int width)
    {
        Point centroid = new Point();

        centroid.x = x + (int)(width/2);
        centroid.y = y + (int)(height/2);

        return centroid;
    } //end getCentroid

    /**
     Get area of Overlap between two bbox
     */
    private double getAreaOfOverlap(Rect a, Rect b)
    {
        double areaOfOverlap = 0;
//        // only if there is actually an overlap
//        if (  ((a.x+a.width > b.x ) || (b.x+b.width > a.x ) ) && ( (a.y+a.height>b.y) || (b.y+b.height>a.y) ) )
//            areaOfOverlap = Math.abs(Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x , b.x)) * //width
//                Math.abs(Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y , b.y)); // height

        // Length of intersecting part i.e
        // start from max(l1.x, l2.x) of
        // x-coordinate and end at min(r1.x,
        // r2.x) x-coordinate by subtracting
        // start from end we get required
        // lengths

        int x_dist = (Math.min(a.x+a.width, b.x+b.width)
                - Math.max(a.x, b.x));
        int y_dist = (Math.min(a.y+a.height, b.y+b.height)
                - Math.max(a.y, b.y));

        if( x_dist > 0 && y_dist > 0 )
        {
            areaOfOverlap = x_dist * y_dist;
        }

        return areaOfOverlap;
    }

    /**
     Get Intersection over Union between two bbox
     */
    private float getIOU(Rect a, Rect b)
    {
        double areaOfOverlap = getAreaOfOverlap(a, b);

        double areaOfUnion = a.height*a.width - areaOfOverlap + b.height*b.width;

        return (float) (areaOfOverlap/areaOfUnion);
    } //end get iou

    /**
     reset if tracking is NOK
     @params : tracker
     @params : tracked:  the current tracked object
     @params : detections
     */
    private void checkAndResetTracking(Object tracker, TrackedObject tracked, Rect target)
    {
        try{
            if(debugLog)
                Log.d(TAG, "Current tracked Bbox:  "
                        +  tracked.box.x
                        + " " + tracked.box.y
                        + " " + tracked.box.height
                        + " " + tracked.box.width);

            // dimension check
            if(tracked.box.x<=0
                    || tracked.box.y<=0
                    || tracked.box.x+tracked.box.width>smallFrame.cols()
                    || tracked.box.y+tracked.box.height>smallFrame.rows() )
                return;

            /**** Todebug: record images of tracked and where to reset*/
//            Mat trackMat = smallFrame.submat(tracked.box);
//            Imgcodecs.imwrite("/sdcard/Download/"+System.currentTimeMillis()+"0Tracked.jpg", trackMat);

            // Human-face Detection
            detections = detector.recognizeImage(
                            matToBitmapAndResize(smallFrame, 320, 320),
                            0.7f, 0.4f, 99.0f, smallFrame);


            // If detected something
            if (detections.size() > 0) {

                if(debugLog)
                    Log.d(TAG, "Total of detected object = " + detections.size());
                if(detections.size()<=1)
                    Imgcodecs.imwrite("/storage/emulated/0/Download/"+System.currentTimeMillis()+"_baddetection.jpg",
                            smallFrame);


                //get closest human
                Point trackedCentroid = getCentroid(tracked.box.x, // upper left corner x
                        tracked.box.y, // upper left corner y
                        tracked.box.height, // height
                        tracked.box.width); // width

                // init
                int numOfDetectedHumans = 0;
                Double dist = 0.0;
                Double maxDist = Double.POSITIVE_INFINITY;
                int idClosest = -1;
                // for each detection
                for (int i=0; i<detections.size(); i++)
                {
                    // only if human
                    if(detections.get(i).getDetectedClass()==0)
                    {

                        // todeleteafterdebug
                        int left = (int) (detections.get(i).left * frameCols);
                        int top =  (int) (detections.get(i).top * frameCols);
                        int right = (int) ((detections.get(i).right-detections.get(i).left) * frameCols);
                        int bottom = (int) ((detections.get(i).bottom-detections.get(i).top) * frameRows);

                        Imgproc.rectangle(smallFrame, new Point(left, top ), new Point(right, bottom),
                                _WHITE, 4);
                        //end todeleteafterdebug
                        numOfDetectedHumans+=1;

                        Point detectionCentroid = getCentroid((int) (detections.get(i).left * frameCols), // upper left corner x
                                (int) (detections.get(i).top * frameCols), // upper left corner y
                                (int) ((detections.get(i).right-detections.get(i).left) * frameCols) , // width
                                (int) ((detections.get(i).bottom-detections.get(i).top) * frameRows)); // height

                        // find the closest detection to the tracking position
                        // L1 distance to optimize computing time
                        dist =  (Math.abs(detectionCentroid.x - trackedCentroid.x) + Math.abs(detectionCentroid.y - trackedCentroid.y));
                        if (dist < maxDist) {
                            // update
                            maxDist = dist;
                            idClosest = i;
                        }
                    } //end if
                } //next detection

                if (numOfDetectedHumans==0)
                {
                    Log.w(TAG, "Tried to check and reset but found 0 human");
                }

                // if found a closest
                if (idClosest>=0)
                {
                    int left = (int)(detections.get(idClosest).left * smallFrame.cols());
                    int top = (int)(detections.get(idClosest).top * smallFrame.rows());
                    int right = (int)(detections.get(idClosest).right * smallFrame.cols());
                    int bottom = (int)(detections.get(idClosest).bottom* smallFrame.rows());

                    // todeleteafterdebug
                    Imgproc.rectangle(smallFrame, new Point(left, top ), new Point(right, bottom),
                            _RED, 3);
                    //end todeleteafterdebug

                    // pose estimation
                    Rect toCrop = new Rect(
                            left,
                            top,
                            right-left-5,
                            bottom-top-10
                    );
                    Log.i(TAG, "To crop "+left + " " + top + " " + (right-left) + " " + (bottom-top) );
                    Mat croppedTargetMat = smallFrame.submat(toCrop);
                    Imgproc.resize(croppedTargetMat, croppedTargetMat, new Size(256,256));
                    Bitmap bitmapImage = Bitmap.createBitmap(croppedTargetMat.cols(), croppedTargetMat.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(croppedTargetMat, bitmapImage);

                    float[][] result = poseEstimator.recognizeImage(bitmapImage);

                    // Blaze pose returns the x, y coords as values [0:255] independent from the input resolution
                    int noseX = (int) (result[0][0*5] * (right-left)/255);
                    int noseY = (int) (result[0][0*5+1] * (bottom-top)/255);
                    int leftEyeX = (int) (result[0][2*5] * (right-left)/255);
                    int leftEyeY = (int) (result[0][2*5+1] * (bottom-top)/255);
                    int rightEyeX = (int) (result[0][5*5] * (right-left)/255);
                    int rightEyeY = (int) (result[0][5*5+1] * (bottom-top)/255);
                    int leftShoulderX = (int) (result[0][11*5] * (right-left)/255);
                    int leftShoulderY = (int) (result[0][11*5+1] * (bottom-top)/255);
                    int rightShoulderX = (int) (result[0][12*5] * (right-left)/255);
                    int rightShoulderY = (int) (result[0][12*5+1] * (bottom-top)/255);
                    int leftHipX = (int) (result[0][23*5] * (right-left)/255);
                    int leftHipY = (int) (result[0][23*5+1] * (bottom-top)/255);
                    int rightHipX = (int) (result[0][24*5] * (right-left)/255);
                    int rightHipY = (int) (result[0][24*5+1] * (bottom-top)/255);
                    //Log.w(TAG, "Coords : "+ shoulderX + " " +  shoulderY );

                    Imgproc.circle(smallFrame, new Point(
                            left + noseX, top + noseY), 5, new Scalar(0,255,0), 10);
                    Imgproc.circle(smallFrame, new Point(
                            left + leftEyeX, top + leftEyeY), 5, new Scalar(0,255,0), 10);
                    Imgproc.circle(smallFrame, new Point(
                            left + rightEyeX, top + rightEyeY), 5, new Scalar(0,255,0), 10);

                    Imgproc.circle(smallFrame, new Point(
                            left + leftShoulderX, top + leftShoulderY), 5, new Scalar(0,255,0), 10);
                    Imgproc.circle(smallFrame, new Point(
                            left + rightShoulderX, top + rightShoulderY), 5, new Scalar(0,255,0), 10);

                    Imgproc.circle(smallFrame, new Point(
                            left + leftHipX, top + leftHipY), 5, new Scalar(0,255,0), 10);
                    Imgproc.circle(smallFrame, new Point(
                            left + rightHipX, top + rightHipY), 5, new Scalar(0,255,0), 10);

                    Imgcodecs.imwrite("sdcard/Download/" + System.currentTimeMillis() + "_poseest.jpg", smallFrame);

                    /** check if need to reset**/

                    //check if skeleton intersects with current tracked
                    // nose, eyes, ears shoulders are in the current tracked


                } // end if found a closest





                // if not do nothing

                Rect detectionBboxToReset = null;

                float bestOverlapRatio = 0.0f;
                float bestIou = 0.0f;

                // scan all detections
                for (int i = 0; i < detections.size(); ++i) {

                    // Bbox of the detection
                    Rect detectionBbox = new Rect((int) (detections.get(i).left * frameCols), (int)(detections.get(i).top*frameRows),
                            (int) ((detections.get(i).right-detections.get(i).left) * frameCols) , // width
                            (int) ((detections.get(i).bottom-detections.get(i).top) * frameRows)); // height


                    /**If currently tracking a face
                     // we'll try to stay on it (checking if the IoU with a face is OK)
                     // if a face is not available, we reset the tracker on the human with the best overlap*/
                    if(tracked.objectClass==1) //0:human, 1:face
                    {
                        // if detection is a face
                        if (detections.get(i).getDetectedClass() == 1) {

//                            // check IoU
//                            float iou = getIOU(tracked.box, detectionBbox);

                            // calculating the area of the current tracked face
                            double faceArea = tracked.box.height*tracked.box.width;

                            // ratio between the area of the face included in the human bbox and the total face area
                            float overlapRatio = (float)(getAreaOfOverlap(detectionBbox, tracked.box) / faceArea);

                            if(debugLog)
//                                Log.d(TAG, "Tracking a face and found a face with IoU="+iou+"\n"
                                Log.d(TAG, "Tracking a face and found a face with overlapratio="+overlapRatio+"\n"
                                        +  detectionBbox.x
                                        + " " + detectionBbox.y
                                        + " " + detectionBbox.height
                                        + " " + detectionBbox.width
                                        + " frame size =(" + smallFrame.size()+")");

                            // if IoU good enough
                            if (overlapRatio >= IOU_THRES)
                            {
                                if(debugLog)
                                    Log.d(TAG, "IoU OK!");

                                // declare tracking as OK
                                trackingSuccess = true;
                                return; // do nothing -> Exit the function to keep the current tracking object
                                //   NB: we do not reset on the detected face as there is a possibility to be another occluding face

                            } // end if IoU OK
                            else
                            {
                                Log.d(TAG, "IoU NOK :( ");
                                Mat currTracked = smallFrame.submat(tracked.box);
                                Imgcodecs.imwrite("/storage/emulated/0/Download/"+System.currentTimeMillis()+"_currTracking.jpg",
                                        currTracked);
                                Mat candidate = smallFrame.submat(detectionBbox);
                                Imgcodecs.imwrite("/storage/emulated/0/Download/"+System.currentTimeMillis()+"_faceCandidate.jpg",
                                        candidate);

                            }
                        } // end if detected object is a face

                        else // object detected is a human
                        {
                            // calculating the area of the face
                            double faceArea = tracked.box.height*tracked.box.width;

                            // ratio between the area of the face included in the human bbox and the total face area
                            float overlapRatio = (float)(getAreaOfOverlap(detectionBbox, tracked.box) / faceArea);

                            if(debugLog)
                                Log.d(TAG, "Tracking a face but found a human " +
                                        " with overlap="+getAreaOfOverlap(detectionBbox, tracked.box)
                                        + " facearea=" + faceArea
                                        + " overlapRatio="+ overlapRatio
                                        +"\n"
                                        +  detectionBbox.x
                                        + " " + detectionBbox.y
                                        + " " + detectionBbox.height
                                        + " " + detectionBbox.width
                                        + " frame size =(" + smallFrame.size()+")");

                            if (overlapRatio>OVERLAPRATIO_THRES)
                            {
                                if(debugLog)
                                    Log.d(TAG, "Overlap Ratio OK!");
                                // if best overlap ratio so far
                                if (overlapRatio> bestOverlapRatio)
                                {
                                    // remember this detection
                                    detectionBboxToReset = detectionBbox;
                                    bestOverlapRatio = overlapRatio;
                                }

                            } //end if overlap ratio good enough

                        } //end if object detected is a human
                    } //end if is currently tracking a face


                    else /** is currently tracking a human
                     // we'll try to find the respective face and reset the tracker on it
                     // if a face is not available, we reset the tracker on the human with the best IoU*/
                    {
                        // if detection is a face
                        if (detections.get(i).getDetectedClass() == 1) {

                            // calculating the area of the face
                            double faceArea = detectionBbox.height*detectionBbox.width;

                            // ratio between the area of the face included in the human bbox and the total face area
                            float overlapRatio = (float)(getAreaOfOverlap(detectionBbox, tracked.box) / faceArea);

                            if(debugLog)
                                Log.d(TAG, "Tracking a human but found a face " +
                                        " with overlap="+getAreaOfOverlap(detectionBbox, tracked.box)
                                        + " facearea=" + faceArea
                                        + " overlapRatio="+ overlapRatio
                                        +"\n"
                                        +  detectionBbox.x
                                        + " " + detectionBbox.y
                                        + " " + detectionBbox.height
                                        + " " + detectionBbox.width
                                        + " frame size =(" + smallFrame.size()+")");

                            if (overlapRatio>OVERLAPRATIO_THRES)
                            {
                                if(debugLog)
                                    Log.w(TAG, "Overlap Ratio OK!  Reseting on that face");
                                // reset on that face
                                resetTracker(vitTracker, detectionBbox, 1 );
                                if(debugLog)
                                    Log.d(TAG, "reset done: returning");
                                return; //exit once it is done

                            } //end if overlap ratio good enough

                        } // end if detected a face
                        else // detected a human
                        {

                            double trackedArea = tracked.box.height*tracked.box.width;
                            // ratio between the area of the face included in the human bbox and the total face area
                            float overlapRatio = (float)(getAreaOfOverlap(detectionBbox, tracked.box) / trackedArea);

                            // check IoU
                            float iou = getIOU(tracked.box, detectionBbox);

                            if(debugLog)
                                Log.d(TAG, "Tracking a human and found a human with IoU="+iou+" "
                                        +  "and overlapRatio=" + overlapRatio +"\n"
                                        +  detectionBbox.x
                                        + " " + detectionBbox.y
                                        + " " + detectionBbox.height
                                        + " " + detectionBbox.width
                                        + " frame size =(" + smallFrame.size()+")");

                            // if overlap good enough
//                            if (iou >= IOU_THRES)
                            if (overlapRatio >= OVERLAPRATIO_THRES)
                            {
                                if(debugLog)
                                    Log.d(TAG, "Overlap OK!");

                                // if best overlap ratio so far
                                if (iou> bestIou)
                                {
                                    // remember this detection
                                    detectionBboxToReset = detectionBbox;
                                    bestIou = iou;
                                }
                            } // end if IoU OK
                        } //end if detected a human

                    } //end if is currently tracking a human


                    //
                    // ********* in parallel, look for the closest object
                    //
                    trackedCentroid = getCentroid(tracked.box.x, // upper left corner x
                            tracked.box.y, // upper left corner y
                            tracked.box.height, // height
                            tracked.box.width); // width
                    Point detectionCentroid = getCentroid((int) (detections.get(i).left * frameCols), // upper left corner x
                            (int) (detections.get(i).top * frameCols), // upper left corner y
                            (int) ((detections.get(i).right-detections.get(i).left) * frameCols) , // width
                            (int) ((detections.get(i).bottom-detections.get(i).top) * frameRows)); // height

                    // find the closest detection to the tracking position
                    // L1 distance to optimize computing time
                    dist =  (Math.abs(detectionCentroid.x - trackedCentroid.x) + Math.abs(detectionCentroid.y - trackedCentroid.y));
                    if (dist < maxDist) {
                        // update
                        maxDist = dist;
                        idClosest = i;
                    }

                } //next detection

                /** ******** After scanning through all the detections : we have 3 possibilities
                 // 1) we're tracking a face but don't see it anymore and detect a human -> we reset on the best over overlapping human
                 // 2) we're tracking a human -> we reset on the best intersecting human
                 // 3)  we couldn't find an overlapping object -> we reset on the closest object
                 */

                // >>> reset on human
                if (detectionBboxToReset != null)
                {
                    if (BuildConfig.DEBUG)
                        Log.w("coucou", "Reset ViTTracker on Human \n"
                                +  detectionBboxToReset.x
                                + " " + detectionBboxToReset.y
                                + " " + detectionBboxToReset.height
                                + " " + detectionBboxToReset.width
                        );

                    // Adjusting on a human silhouette
                    // crop extra area
                    detectionBboxToReset = cropExtraArea(detectionBboxToReset, 0);

                    resetTracker(tracker, detectionBboxToReset, 0 );
                }
                else // >>> else init on closest object
                {

                    // reset on the bbox of the closest detection
                    detectionBboxToReset = new Rect((int) (detections.get(idClosest).left * frameCols), (int)(detections.get(idClosest).top*frameRows),
                            (int) ((detections.get(idClosest).right - detections.get(idClosest).left) * frameCols),
                            (int) ((detections.get(idClosest).bottom - detections.get(idClosest).top) * frameRows) );

                    if (BuildConfig.DEBUG)
                        Log.w("coucou", "Reset ViTTracker on closest detection ["
                                + detections.get(idClosest).getDetectedClass() + "]\n"
                                +  detectionBboxToReset.x
                                + " " + detectionBboxToReset.y
                                + " " + detectionBboxToReset.height
                                + " " + detectionBboxToReset.width
                        );

                    resetTracker(tracker, detectionBboxToReset, detections.get(idClosest).getDetectedClass() );

                } // end if found a human or reset on closest object


                //coucou todelete
                trackingSuccess = true;


            } //end if detection size >0

        } catch (Exception e) {
            Log.e(TAG, "ERROR During CheckReset " + Log.getStackTraceString(e));
        }


    }


    private void resetTracker(Object tracker, Rect bbox, int classOfTrack)
    {
        vitTracker.init(smallFrame, bbox);
        tracked.objectClass = classOfTrack;
        trackingSuccess = true;

    } // end of rest Tracker

    private float computeTrackingScore(ArrayList<Float> array)
    {
        //init to OK value
        float averageScore = 1.0f;

        // add last tracking score
        array.add(vitTracker.getTrackingScore());

        // wait to fill the array
        if (array.size()>NUM_OF_SCORE_HISTORY) {
            //remove oldest entry
            array.remove(0);

            // compute average
            float sum = 0.0f;
            for (int i = 0; i < NUM_OF_SCORE_HISTORY; i++) {
                sum += array.get(i);
            }

            averageScore = sum / array.size();
        }

        avscore = averageScore;
        return averageScore;
    }


    public class TrackedObject
    {
        public Rect box = new Rect();
        public int objectClass = -1;
        public float score = 0.0f;

    }
}
