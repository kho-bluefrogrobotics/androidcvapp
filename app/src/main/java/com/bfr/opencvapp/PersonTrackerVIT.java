package com.bfr.opencvapp;

import static com.bfr.opencvapp.MainActivity.speedLinearGrafcet;
import static com.bfr.opencvapp.utils.Utils.Color.*;
import static com.bfr.opencvapp.utils.Utils.MODELS_DIR;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;

import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.utils.TfLiteYoloXHumanHeadHands;
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
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.TrackerVit;
import org.opencv.video.TrackerVit_Params;

import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class PersonTrackerVIT {

    private final String TAG = "_VISION Person tracker";

    // Person detector for init
    MultiDetector detector;
    // Detector for verification
    TfLiteYoloXHumanHeadHands humanHeadHandsDetector;

    // Pose estimation for torso height
    Pose mypose;
    PoseDetectorOptions poseDetectoptions;
    PoseDetector poseDetector;
    // torso height to judge the distance from the person
    public float torsoHeight=0.0f;

    // List of detected person or faces or hands
    ArrayList<MultiDetector.Recognition> detections = new ArrayList<MultiDetector.Recognition>();
    ArrayList<TfLiteYoloXHumanHeadHands.Recognition> hhhDetections = new ArrayList<TfLiteYoloXHumanHeadHands.Recognition>();


    public int frameCount =0;
    private int _FRAME_DETECT = 50; // 15fps -> check/reset every 3-4 s
    // hard limit for human height to track (in pixel) and crop
    private int MIN_HUMAN_HEIGHT = 120;
    // hard limit for face/head size to track before switching to body
    private int MIN_HEAD_SIZE = 85; // recommended for FOLLOW ME: 85



    // Tracker
    TrackerVit vitTracker;
    TrackerVit_Params vitTrackerparams ;
    // Human Tracking
    final String VITTRACKMODEL = MODELS_DIR + "object_tracking_vittrack_2023sep.onnx";

    // copy of the input frame
    Mat frame;

    // Tracked target
    public TrackedObject tracked = new TrackedObject() ;

    // internal flag: if tracking NOK -> reset
    private boolean trackingSuccess = false;
    public boolean isTracking = false; // isTracking is false only when the tracker doesn't have any target
                                        // = is trying to init

    // to display
    public Mat displayMat;
    public boolean readyToDisplay = false;
    private Point pt1 = new Point(0, 0);
    private Point pt2 = new Point(0, 0);

    // Threshold for tracking verification with closest objects
    float IOU_THRES = 0.4f;
    float OVERLAPRATIO_THRES = 0.5f;

    // mean of Tracking confidence score
    ArrayList<Float> scoreHistory = new ArrayList<Float>();
    final float NUM_OF_SCORE_HISTORY = 30; //15fps -> deals with occlusion during ~2s
    // mean of torso height
    ArrayList<Float> torsoHeightHistory = new ArrayList<Float>();
    final float NUM_OF_TORSOHEIGHT_HISTORY = 3;
    float avscore = 0.0f;


    public PersonTrackerVIT(MultiDetector personDetector, TfLiteYoloXHumanHeadHands hhhdetector ){

        Log.d(TAG, "PersonTracker creation"  ) ;

        // Load model for human detector
        detector = personDetector;
        // detector for tracking reset
        humanHeadHandsDetector = hhhdetector;

        // init imgs
        displayMat = new Mat();
        frame = new Mat();

        //init ViT Tracker
        vitTrackerparams = new TrackerVit_Params();
        vitTrackerparams.set_net(VITTRACKMODEL);
        vitTracker = TrackerVit.create(vitTrackerparams);

        // pose estimator for torso height
        poseDetectoptions =
                new PoseDetectorOptions.Builder()
                        .setDetectorMode(PoseDetectorOptions.SINGLE_IMAGE_MODE)
                        .setPreferredHardwareConfigs(PoseDetectorOptions.CPU_GPU)
                        .build();

        poseDetector = PoseDetection.getClient(poseDetectoptions);

    }



    /**
     * Visual tracking of target in successive images
     * @param inputImg the img where to track the target
     * @param constructVisualizationImage enable/disable the construction of the display image (to gain some speed)
     * @return nothing
     */
    public void visualTracking(Mat inputImg, boolean constructVisualizationImage)
    {

        try {

            frame = inputImg.clone();

            // Rotate image with Buddy's head orientation (empiric tuning)
            double angle = 1.3* BuddySDK.Actuators.getYesPosition() *-(BuddySDK.Actuators.getNoPosition())/80;
            Mat mapMatrix = Imgproc.getRotationMatrix2D(new Point((int)frame.cols()/2, (int)frame.rows()/2)
                    , angle, 1.0);
            Imgproc.warpAffine(frame, frame, mapMatrix, new Size(frame.cols(), frame.rows()));


            /*** First,  init on object detection*/
            if(frameCount == 0)
            {
                Log.d(TAG, "\n\n****************************************************\n" +
                        "Init tracker " ) ;

                // reset
                isTracking = false;

                // Detection
                // Using the most robust model we have to avoid false detection at init
                detections = detector.recognizeImage(frame,
                        0.5f, 0.6f, 99.0f, 0.3f, true);

                Log.d(TAG, "Num. of detected objects for init : " + detections.size());

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
                        Log.w(TAG, "Init on first detection: " + detectedFaceId ) ;
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

                isTracking = true;

                // Reset tracking only every xxx frames of in case of tracking lost
                if (frameCount %_FRAME_DETECT == 0 || !trackingSuccess ) {

                    Log.d(TAG, "\n\n****************************************************\n" +
                            "Checking if need to reset  trackingsucess =" + trackingSuccess + " frameNum="+frameCount ) ;

                    //reset in any case
                    frameCount = 1;

                    // Detection
                    hhhDetections = humanHeadHandsDetector.recognizeImage(
                            this.frame, 0.6f, 0.5f, 99.0f );


                    checkAndResetTracking(tracked,  hhhDetections);

                }
                else  // other frames -> only tracking
                {

                    //Update tracker
                    vitTracker.update(this.frame, tracked.box);
                    tracked.score = vitTracker.getTrackingScore();
                    if(computeTrackingScore(scoreHistory)>=0.6f && tracked.box.width < this.frame.cols()/2)  // check size: Bbox must not be too big-> it means the person is very close to the camera and we should reset on the face)
                        trackingSuccess=true;
                    else {
                        trackingSuccess = false;
                        Log.e(TAG, "Tracking NOK : score history= " +scoreHistory + " current score=" + vitTracker.getTrackingScore());
                    }

                    //increment frame count
                    frameCount +=1;

                } // end rest of the frames


            } // end if First image or Already initialized


            // Create display image with result
            if(constructVisualizationImage) {
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

            Mat displayMat = frame.clone();

            // to debug: lmits in Follow-Me
            if(tracked.box.y<= speedLinearGrafcet.MAX_UPPER_LIMIT)
                Imgproc.rectangle(displayMat, new Point(10, speedLinearGrafcet.MAX_UPPER_LIMIT), new Point(1020, 750),
                        _BLUE, 5);
            else
                Imgproc.rectangle(displayMat, new Point(10, speedLinearGrafcet.MAX_UPPER_LIMIT), new Point(1020, 750),
                        _YELLOW, 2);

            if (frameCount == 0 ) // not tracking yet
            {

                Imgproc.putText(displayMat, "Not tracking",
                        new Point(500, 300),
                        2, 2, _WHITE, 8);
                Imgproc.putText(displayMat, "Not tracking",
                        new Point(500, 300),
                        2, 2, _BLACK, 5);
                Imgproc.putText(displayMat, "Not tracking",
                        new Point(500, 300),
                        2, 2, _BLUE, 2);

            }
            else // tracking : displaying rectangle
            {
                // draw a rectangle around Target
                pt1.x = (int) (tracked.box.x);
                pt1.y = (int) (tracked.box.y);
                pt2.x = (int) ( (tracked.box.x + tracked.box.width));
                pt2.y = (int) ( (tracked.box.y + tracked.box.height) );

                Imgproc.rectangle(displayMat, pt1, pt2,
                        _WHITE, 4);
                Imgproc.rectangle(displayMat, pt1, pt2,
                        _RED, 2);
                Imgproc.putText(displayMat, "Tracking",
                        new Point(pt1.x, pt1.y-30),
                        2, 1, _BLACK, 5);
                Imgproc.putText(displayMat, "[" + String.format(java.util.Locale.US, "%.3f", avscore) + "]",
                        new Point(pt1.x, pt1.y),
                        2, 1, _BLACK, 5);
                Imgproc.putText(displayMat, "Tracking",
                        new Point(pt1.x, pt1.y-30),
                        2, 1, _GREEN, 2);
                Imgproc.putText(displayMat, "[" + String.format(java.util.Locale.US, "%.3f", avscore) + "]",
                        new Point(pt1.x, pt1.y),
                        2, 1, _GREEN, 2);

                // to debug
//                Imgproc.putText(displayMat, "" + SpeedLinearGrafcet.step_num + " ("+torsoHeight+")",
//                        new Point(500, 100),
//                        2, 2, _WHITE, 10);
//                Imgproc.putText(displayMat, "" + SpeedLinearGrafcet.step_num + " ("+torsoHeight+")",
//                        new Point(500, 100),
//                        2, 2, _RED, 5);
//
//                Imgproc.putText(displayMat, "" + speedLinearGrafcet.linearSpeed,
//                        new Point(500, 150),
//                        2, 2, _WHITE, 10);
//                Imgproc.putText(displayMat, "" + speedLinearGrafcet.linearSpeed,
//                        new Point(500, 150),
//                        2, 2, _BLUE, 5);
//
//                int surf = tracked.box.height*tracked.box.width;
//                Imgproc.putText(displayMat, "[" + surf + "]",
//                        new Point(pt1.x, pt1.y+60),
//                        2, 1, _BLACK, 5);
//
//                Imgproc.putText(displayMat, "[" + surf + "]",
//                        new Point(pt1.x, pt1.y+60),
//                        2, 1, _YELLOW, 2);

            } //end if is tracking or not yet


            return displayMat;

        } catch (Exception e) {
            Log.e(TAG, "ERROR TRACKING: " + Log.getStackTraceString(e));
            return  null;
        }

    } //end displayResult




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
        Rect detectionBbox = new Rect((int) (detection.left * frame.cols()), (int)(detection.top*frame.rows()),
                (int) ((detection.right-detection.left) * frame.cols()) , // width
                (int) ((detection.bottom-detection.top) * frame.rows())); // height

        // crop extra area whether it is a face or a human
        tracked = cropExtraArea(detectionBbox, detection.getDetectedClass());


        Log.d("coucou", "Init the tracker class= " + detection.getDetectedClass() + "\n"
                + " x=" + tracked.x
                + " y=" + tracked.x
                + " height=" + tracked.height
                + " width=" + tracked.width) ;

        // Init the tracker on the detection
        vitTracker.init(frame, tracked);

        trackingSuccess = true;

        return tracked;

    }// end Init tracker


//    /**
//     Get closest detection to the current tracked bbox
//     */
//    private int getClosestDetection(Rect tracked, ArrayList<MultiDetector.Recognition> detections)
//    {
//        int idClosest = 0;
//
//        Point trackedCentroid = getCentroid(tracked.x, // upper left corner x
//                tracked.y, // upper left corner y
//                tracked.height, // height
//                tracked.width); // width
//
//        // for each detection
//        for (int i = 0; i < detections.size(); ++i) {
//
//            if (detections.get(i).getConfidence() > THRESHOLD // if object detected with enough confidence
//                    || (detections.get(i).getDetectedClass()==1 && detections.get(i).getConfidence()>=0.5)) { // or is a face so that we lower the confidence
////                //left
////                pt1.x = (int) (detections.get(i).left * frameCols);
////                //top
////                pt1.y = (int) (detections.get(i).top * frameRows);
//
//                Point detectionCentroid = getCentroid((int) (detections.get(i).left * frameCols), // upper left corner x
//                        (int) (detections.get(i).top * frameCols), // upper left corner y
//                        (int) ((detections.get(i).bottom-detections.get(i).right) * frameRows) , // height
//                        (int) ((detections.get(i).right-detections.get(i).left) * frameCols)); // width
//                //if istracking whole body, check if face detected inside the target
////                if (isTrackingAPerson)
//                if (false)
//                {
//                    //if detection is a face
//                    if(detections.get(i).getDetectedClass()==1) {
//                        // check if inside the tracked boundingbox
//                        if ((detections.get(i).left * frameCols) >= tracked.x && (detections.get(i).right * frameCols <= tracked.x + tracked.width)) {
//                            idClosest = i;
//                            classOfClosestDetection = detections.get(i).getDetectedClass();
//                            break;
//                        }//end if inside tracked bbox
//                    } // end if detection is a face
//                } // endif is tracking a whole body
//
//                // find the closest detection to the tracking position
//                // L1 distance to optimize computing time
//                dist =  (Math.abs(pt1.x - tracked.x) + Math.abs(pt1.y - tracked.y));
//                if (dist < maxDist) {
//                    // update
//                    maxDist = dist;
//                    idClosest = i;
//                    classOfClosestDetection = detections.get(i).getDetectedClass();
//                }
//
//            } // end if confidence OK
//        } // next detection
//
//        return idClosest;
//    } //end getClosest


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
            croppedArea.x = (int) (tracked.x + tracked.width/4 );
            croppedArea.width = (int)(tracked.width- (tracked.width/2) );
            // track the 2/3 upper part  or minimum arbitrary value
            croppedArea.height = Math.max( (int) (0.5 * tracked.height), MIN_HUMAN_HEIGHT  );


            //convert to bitmap
            Mat croppedResult = frame.submat(croppedArea);
            Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_newTrackingHuma.jpg",
                    croppedResult);

        }
        else if (detectedClass == 1) // Face
        {
            //TODO : see if really necessary to crop extra area
            croppedArea.y = (int)(tracked.y + 0.1*tracked.height);
            //tracked.height = Math.max( (int) (0.75 * tracked.height),  10 );
        }
        else
        {
            croppedArea = tracked;
        }

        return croppedArea;
    }

    /**
     Get the centroid of a bounding box (from upper left corner coordinates and height/width)
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
     @param tracked the current tracked object
     @param detections list of detections returned by YOLO
     */
    private void checkAndResetTracking(TrackedObject tracked, ArrayList<TfLiteYoloXHumanHeadHands.Recognition> detections)
    {
        try{
            Log.d(TAG, "Current tracked Bbox:  "
                    +  tracked.box.x
                    + " " + tracked.box.y
                    + " " + tracked.box.height
                    + " " + tracked.box.width);

            // dimension check & capping
            if(tracked.box.x<=0)
                tracked.box.x =1;
            if(tracked.box.y<=0)
                tracked.box.y =1;
            if(tracked.box.x+tracked.box.width> frame.cols() && tracked.box.x < frame.cols())
                tracked.box.width = frame.cols()-tracked.box.x;
            if(tracked.box.y+tracked.box.height> frame.rows() && tracked.box.y < frame.rows())
                tracked.box.height = frame.rows()-tracked.box.y;

            //safety check
            if(tracked.box.x<=0
                    || tracked.box.y<=0
                    || tracked.box.x+tracked.box.width> frame.cols()
                    || tracked.box.y+tracked.box.height> frame.rows() )
            {
                // declare tracking as NOK
                trackingSuccess = false;
                // reset frame num.
                frameCount =0;
                Log.e(TAG, "Tracking NOK : during reset, box out of range : "
                        + tracked.box.x +" "
                        + tracked.box.y +" "
                        + tracked.box.width +" "
                        + tracked.box.height +" "
                );
                return;
            }


            Mat currTracked = frame.submat(tracked.box);
            Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_currTracking_"+tracked.objectClass+".jpg",
                    currTracked);

            /**** Todebug: record images of tracked and where to reset*/
//            Mat trackMat = smallFrame.submat(tracked.box);
//            Imgcodecs.imwrite("/sdcard/Download/trackingdebug/"+System.currentTimeMillis()+"0Tracked.jpg", trackMat);

            // If detected something
            if (detections.size() > 0) {

                Log.d(TAG, "Total of detected object = " + detections.size());

                Rect detectionBboxToReset = null;

                Double dist = 0.0;
                Double maxDist = Double.POSITIVE_INFINITY;
                int idClosest = 0;
                float bestOverlapRatio = 0.0f;
                float bestIou = 0.0f;

                // scan all detections
                for (int i = 0; i < detections.size(); ++i) {

                    // Bbox of the detection
                    Rect detectionBbox = new Rect((int) (detections.get(i).left * frame.cols()),
                            (int)(detections.get(i).top*frame.rows()),
                            Math.min((int) ((detections.get(i).right-detections.get(i).left) * frame.cols()), frame.cols() -(int) (detections.get(i).left * frame.cols()) ) , // width
                            Math.min( (int) ((detections.get(i).bottom-detections.get(i).top) * frame.rows()), (int) frame.rows() - (int)(detections.get(i).top*frame.rows()))
                    ); // height


                    /**If currently tracking a face
                     // we'll try to stay on it (checking if the IoU with a face is OK)
                     // if a face is not available, we reset the tracker on the human with the best overlap*/
                    if(tracked.objectClass==1) //0:human, 1:face
                    {
                        // if detection is a face
                        if (detections.get(i).getDetectedClass() == 1) {

                            // check IoU
                            float iou = getIOU(tracked.box, detectionBbox);

                            Log.d(TAG, "Tracking a face and found a face with IoU="+iou+"\n"
                                    +  detectionBbox.x
                                    + " " + detectionBbox.y
                                    + " " + detectionBbox.height
                                    + " " + detectionBbox.width
                                    + " frame size =(" + frame.size()+")");


                            Mat candidate = frame.submat(detectionBbox);
                            Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_faceCandidate_"+iou+".jpg",
                                    candidate);

                            if( detectionBbox.height > MIN_HEAD_SIZE)
                            {
                                // if IoU good enough
                                if (iou >= IOU_THRES)
                                {
                                    Log.d(TAG, "IoU OK!");

                                    // declare tracking as OK
                                    trackingSuccess = true;
                                    // go on with tracking -> increment frame num.
                                    frameCount +=1;

                                    // reset score
                                    scoreHistory.clear();
                                    for(int id=0; id<NUM_OF_SCORE_HISTORY; id++)
                                    {
                                        scoreHistory.add(1.0f);
                                    }

                                    Mat newT = frame.submat(detectionBbox);
                                    Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_newTrackFace.jpg",
                                            newT);

                                    return; // do nothing -> Exit the function to keep the current tracking object
                                    //   NB: we do not reset on the detected face as there is a possibility to be another occluding face

                                } // end if IoU OK
                                else
                                {


                                }
                            } //end if head size big enough
                            else
                            {
                                Log.d(TAG, "But size not big enough "+ detectionBbox.height );
                            }

                        } // end if detected object is a face

                        else // object detected is a human
                        {
                            // calculating the area of the face
                            double faceArea = tracked.box.height*tracked.box.width;

                            // ratio between the area of the face included in the human bbox and the total face area
                            float overlapRatio = (float)(getAreaOfOverlap(detectionBbox, tracked.box) / faceArea);

                            Log.d(TAG, "Tracking a face but found a human " +
                                    " with overlap="+getAreaOfOverlap(detectionBbox, tracked.box)
                                    + " facearea=" + faceArea
                                    + " overlapRatio="+ overlapRatio
                                    +"\n"
                                    +  detectionBbox.x
                                    + " " + detectionBbox.y
                                    + " " + detectionBbox.height
                                    + " " + detectionBbox.width
                                    + " frame size =(" + frame.size()+")");


                            Mat candidate = frame.submat(detectionBbox);
                            Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_HumanCandidate_"+overlapRatio+".jpg",
                                    candidate);

                            if (overlapRatio>OVERLAPRATIO_THRES)
                            {
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

                            Log.d(TAG, "Tracking a human but found a face " +
                                    " with overlap="+getAreaOfOverlap(detectionBbox, tracked.box)
                                    + " facearea=" + faceArea
                                    + " overlapRatio="+ overlapRatio
                                    +"\n"
                                    +  detectionBbox.x
                                    + " " + detectionBbox.y
                                    + " " + detectionBbox.height
                                    + " " + detectionBbox.width
                                    + " frame size =(" + frame.size()+")");

                            Mat candidate = frame.submat(detectionBbox);
                            Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_faceCandidate_"+overlapRatio+".jpg",
                                    candidate);

                            if( detectionBbox.height > MIN_HEAD_SIZE)
                            {
                                if (overlapRatio>OVERLAPRATIO_THRES)
                                {
                                    Log.w(TAG, "Overlap Ratio OK!  Reseting on that face");
                                    // reset on that face
                                    resetTracker(detectionBbox, 1 );

                                    // declare tracking as OK
                                    trackingSuccess = true;
                                    // go on with tracking -> increment frame num.
                                    frameCount +=1;

                                    //convert to bitmap
                                    Mat croppedResult = frame.submat(detectionBbox);
                                    Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_newTrackingFace.jpg",
                                            croppedResult);

                                    Log.d(TAG, "reset done: returning");
                                    return; //exit once it is done

                                } //end if overlap ratio good enough
                            }
                            else // Head size not big enough
                            {
                                Log.d(TAG, "But size not big enough "+ detectionBbox.height );
                            } //Head size not big enough


                        } // end if detected a face
                        else // detected a human
                        {

                            double trackedArea = tracked.box.height*tracked.box.width;
                            // ratio between the area of the face included in the human bbox and the total face area
                            float overlapRatio = (float)(getAreaOfOverlap(detectionBbox, tracked.box) / trackedArea);

                            // check IoU
                            float iou = getIOU(tracked.box, detectionBbox);

                            Log.d(TAG, "Tracking a human and found a human with IoU="+iou+" "
                                    +  "and overlapRatio=" + overlapRatio +"\n"
                                    +  detectionBbox.x
                                    + " " + detectionBbox.y
                                    + " " + detectionBbox.height
                                    + " " + detectionBbox.width
                                    + " frame size =(" + frame.size()+")");

                            Mat candidate = frame.submat(detectionBbox);
                            Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_humanCandidate_"+overlapRatio+".jpg",
                                    candidate);

                            // if overlap good enough
//                            if (iou >= IOU_THRES)
                            if (overlapRatio >= OVERLAPRATIO_THRES)
                            {
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
                    Point trackedCentroid = getCentroid(tracked.box.x, // upper left corner x
                            tracked.box.y, // upper left corner y
                            tracked.box.height, // height
                            tracked.box.width); // width
                    Point detectionCentroid = getCentroid((int) (detections.get(i).left * frame.cols()), // upper left corner x
                            (int) (detections.get(i).top * frame.cols()), // upper left corner y
                            (int) ((detections.get(i).right-detections.get(i).left) * frame.cols()) , // width
                            (int) ((detections.get(i).bottom-detections.get(i).top) * frame.rows())); // height

                    // find the closest detection to the tracking position
                    // L1 distance to optimize computing time
                    dist =  (Math.abs(detectionCentroid.x - trackedCentroid.x) + Math.abs(detectionCentroid.y - trackedCentroid.y));
                    if ( (dist < maxDist)
                            &&  Math.abs(detections.get(i).bottom-detections.get(i).top) >= MIN_HEAD_SIZE) //to manage the case that the closest object is a small head
                    {
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
                    Log.d("coucou", "Reset ViTTracker on Human \n"
                            +  detectionBboxToReset.x
                            + " " + detectionBboxToReset.y
                            + " " + detectionBboxToReset.height
                            + " " + detectionBboxToReset.width
                    );

                    // Adjusting on a human silhouette
                    // crop extra area
                    detectionBboxToReset = cropExtraArea(detectionBboxToReset, 0);

                    resetTracker(detectionBboxToReset, 0 );

                    // declare tracking as OK
                    trackingSuccess = true;
                    // go on with tracking -> increment frame num.
                    frameCount +=1;
                }
                else // >>> else init on closest object
                {

                    // reset on the bbox of the closest detection
                    detectionBboxToReset = new Rect((int) (detections.get(idClosest).left * frame.cols()), (int)(detections.get(idClosest).top* frame.rows()),
                            (int) ((detections.get(idClosest).right - detections.get(idClosest).left) * frame.cols()),
                            (int) ((detections.get(idClosest).bottom - detections.get(idClosest).top) * frame.rows()) );

                    Log.d("coucou", "Reset ViTTracker on closest detection ["
                            + detections.get(idClosest).getDetectedClass() + "]\n"
                            +  detectionBboxToReset.x
                            + " " + detectionBboxToReset.y
                            + " " + detectionBboxToReset.height
                            + " " + detectionBboxToReset.width
                    );

                    // crop extra area wether it is a face or a human
                    detectionBboxToReset = cropExtraArea(detectionBboxToReset, detections.get(idClosest).getDetectedClass());
                    resetTracker(detectionBboxToReset, detections.get(idClosest).getDetectedClass() );

                    Mat candidate = frame.submat(detectionBboxToReset);
                    Imgcodecs.imwrite("/storage/emulated/0/Download/trackingdebug/"+System.currentTimeMillis()+"_newTrackClosest.jpg",
                            candidate);
                    // declare tracking as OK
                    trackingSuccess = true;
                    // go on with tracking -> increment frame num.
                    frameCount +=1;

                } // end if found a human or reset on closest object



            } //end if detection size >0
            else // nothing detected
            {
                // declare tracking as OK
                trackingSuccess = false;
                // reset : reinit from the start
                frameCount = 0;
                Log.e(TAG, "Tracking NOK : reset -> nothing detected");
            }

        } catch (Exception e) {
            Log.e(TAG, "ERROR During CheckReset " + Log.getStackTraceString(e));
        }

    }

    /**
     * reset tracker on provided bounding box
     * @param bbox a rect corresponding to the bounding box
     * @param classOfTrack the class of the object detected (human or face)
     */
    private void resetTracker(Rect bbox, int classOfTrack)
    {
        // init tracker
        vitTracker.init(frame, bbox);
        tracked.objectClass = classOfTrack;

        // init score
        scoreHistory.clear();
        for(int i=0; i<NUM_OF_SCORE_HISTORY; i++)
        {
            scoreHistory.add(1.0f);
        }

    } // end of reset Tracker


    /**
     * compute the average of the tracking score
     * @param array the history of the previous tracking scores
     */
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


    // Scheduler for motion detection
    private ScheduledExecutorService poseScheduler ;
    /**
     Runnable to estimate pose of tracked person
     @param tracked the current tracked object
     */
    private  Runnable poseRunnable = new Runnable() {
        @Override
        public void run() {
            try {

//                Mat framecpy = smallFrame.clone();
//
//                //crop around tracked region
//                int newLeft = tracked.box.x-tracked.box.width;
//                int newRight = tracked.box.x+2*tracked.box.width;
//                int newTop = tracked.box.y-20;
//
//                int newBottom = 0;
//                if(tracked.objectClass==0) //tracking a human
//                    newBottom = tracked.box.y + 2* tracked.box.height;
//                else // tracking a face
//                    newBottom = tracked.box.y + 4* tracked.box.height;
//
//                Rect adjustedROI= new Rect(
//                        Math.max(0,newLeft),
//                        Math.max(0,newTop),
//                        Math.min(framecpy.cols()-newLeft,newRight-newLeft),
//                        Math.min(framecpy.rows()-newTop, newBottom-newTop) );
//
//                //Crop around face
//                Mat croppedMat = framecpy.submat(adjustedROI);
//
//
//
//                // Incrust cropped image on black background
////                // init
////                Mat roiInDisplayMat = new Mat(framecpy.rows(),framecpy.cols(), CV_8UC3, new Scalar(0, 0, 0));
////                Rect displayROI= new Rect(
////                        Math.max(0,newLeft),
////                        Math.max(0,newTop),
////                        croppedMat.cols(),
////                        croppedMat.rows() );
////
////                croppedMat.copyTo(roiInDisplayMat.submat(displayROI));
//
//
//                long mlkitTime = System.currentTimeMillis();
//                Bitmap bitmapImage = Bitmap.createBitmap(croppedMat.cols(), croppedMat.rows(), Bitmap.Config.ARGB_8888);
//                Utils.matToBitmap(croppedMat, bitmapImage);
//
//                InputImage inputImage = InputImage.fromBitmap(bitmapImage, 0);
//
//                Task<Pose> result =
//                        poseDetector.process(inputImage)
//                                .addOnSuccessListener(
//                                        new OnSuccessListener<Pose>() {
//                                            @Override
//                                            public void onSuccess(Pose pose) {
//                                                // Task completed successfully
//                                                // ...
//
//                                                mypose = pose;
//                                            }
//                                        })
//                                .addOnFailureListener(
//                                        new OnFailureListener() {
//                                            @Override
//                                            public void onFailure(@NonNull Exception e) {
//                                                // Task failed with an exception
//                                                // ...
//                                            }
//                                        });
//
//                try {
//                    Tasks.await(result);
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//
//                // Get all PoseLandmarks. If no person was detected, the list will be empty
////                List<PoseLandmark> allPoseLandmarks = mypose.getAllPoseLandmarks();
//
//                PoseLandmark nose = mypose.getPoseLandmark(PoseLandmark.NOSE);
//                PoseLandmark leftEar = mypose.getPoseLandmark(PoseLandmark.LEFT_EAR);
//                PoseLandmark rightEar = mypose.getPoseLandmark(PoseLandmark.RIGHT_EAR);
//                PoseLandmark leftShoulder = mypose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER);
//                PoseLandmark rightShoulder = mypose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER);
//                PoseLandmark leftHip = mypose.getPoseLandmark(PoseLandmark.LEFT_HIP);
//                PoseLandmark rightHip = mypose.getPoseLandmark(PoseLandmark.RIGHT_HIP);
//
////                Log.w("coucouMLKit", "MLKit elapsed time : "+ (System.currentTimeMillis()-mlkitTime)
////                        // display position in pixels
////                        +"\n Position=" + nose.getPosition().x + "," + nose.getPosition().y );
//
//                Imgproc.circle(croppedMat, new Point(
//                        0 + nose.getPosition().x, 0 + nose.getPosition().y), 5, new Scalar(0,0,255), 10);
//                Imgproc.circle(croppedMat, new Point(
//                        0 + leftEar.getPosition().x, 0 + leftEar.getPosition().y), 5, new Scalar(0,0,255), 10);
//                Imgproc.circle(croppedMat, new Point(
//                        0 + leftShoulder.getPosition().x, 0 + leftShoulder.getPosition().y), 5, new Scalar(0,0,255), 10);
//                Imgproc.circle(croppedMat, new Point(
//                        0 + rightShoulder.getPosition().x, 0 + rightShoulder.getPosition().y), 5, new Scalar(0,0,255), 10);
//                Imgproc.circle(croppedMat, new Point(
//                        0 + leftHip.getPosition().x, 0 + leftHip.getPosition().y), 5, new Scalar(0,0,255), 10);
//                Imgproc.circle(croppedMat, new Point(
//                        0 + rightHip.getPosition().x, 0 + rightHip.getPosition().y), 5, new Scalar(0,0,255), 10);



//                torsoHeight = Math.abs(leftHip.getPosition().y - leftShoulder.getPosition().y);

                torsoHeight = getTorsoHeight();


            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    };

    /**
     * Get the torso height of the target for distance evaluation
     * Uses the BlazePose MLKit implementation for pose estimation
     * @return the torso height
     */
    public float getTorsoHeight()
    {
        //crop around tracked region
        // adding a few empiric margin to crop the entire target
        int newLeft = tracked.box.x-tracked.box.width;
        int newRight = tracked.box.x+2*tracked.box.width;
        int newTop = tracked.box.y-20;

        int newBottom = 0;
        if(tracked.objectClass==0) //tracking a human
            newBottom = tracked.box.y + 2* tracked.box.height;
        else // tracking a face
            newBottom = tracked.box.y + 5* tracked.box.height;

        Rect adjustedROI= new Rect(
                Math.max(0,newLeft),
                Math.max(0,newTop),
                Math.min(frame.cols()-newLeft,newRight-newLeft),
                Math.min(frame.rows()-newTop, newBottom-newTop) );

        //Crop around face
        Mat croppedMat = frame.submat(adjustedROI);

        //converting to bitmap then Image format
        Bitmap bitmapImage = Bitmap.createBitmap(croppedMat.cols(), croppedMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(croppedMat, bitmapImage);
        InputImage inputImage = InputImage.fromBitmap(bitmapImage, 0);

        // start inference
        Task<Pose> result =
                poseDetector.process(inputImage)
                        .addOnSuccessListener(
                                new OnSuccessListener<Pose>() {
                                    @Override
                                    public void onSuccess(Pose pose) {
                                        // Task completed successfully
                                        mypose = pose;
                                    }
                                })
                        .addOnFailureListener(
                                new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        Log.e(TAG, "Error during Pose estimation ofr torso Height: " + Log.getStackTraceString(e));
                                    }
                                });

        // sync task
        try {
            Tasks.await(result);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Get all PoseLandmarks
        PoseLandmark leftShoulder = mypose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER);
        PoseLandmark leftKnee = mypose.getPoseLandmark(PoseLandmark.LEFT_KNEE);


//                Log.w("coucouMLKit", "MLKit elapsed time : "+ (System.currentTimeMillis()-mlkitTime)
//                        // display position in pixels
//                        +"\n Position=" + rightAnkle.getPosition().x + "," + rightAnkle.getPosition().y );
//        mlkitTime = System.currentTimeMillis();
//
//        Imgproc.circle(croppedMat, new Point(
//                0 + nose.getPosition().x, 0 + nose.getPosition().y), 5, new Scalar(0,0,255), 10);
//        Imgproc.circle(croppedMat, new Point(
//                0 + leftEar.getPosition().x, 0 + leftEar.getPosition().y), 5, new Scalar(0,0,255), 10);
//        Imgproc.circle(croppedMat, new Point(
//                0 + leftShoulder.getPosition().x, 0 + leftShoulder.getPosition().y), 5, new Scalar(0,0,255), 10);
//        Imgproc.circle(croppedMat, new Point(
//                0 + rightShoulder.getPosition().x, 0 + rightShoulder.getPosition().y), 5, new Scalar(0,0,255), 10);
//        Imgproc.circle(croppedMat, new Point(
//                0 + leftHip.getPosition().x, 0 + leftHip.getPosition().y), 5, new Scalar(0,0,255), 10);
//        Imgproc.circle(croppedMat, new Point(
//                0 + rightHip.getPosition().x, 0 + rightHip.getPosition().y), 5, new Scalar(0,0,255), 10);
//        Imgproc.circle(croppedMat, new Point(
//                0 + leftKnee.getPosition().x, 0 + leftKnee.getPosition().y), 3, new Scalar(0,255,0), 5);
//        Imgproc.circle(croppedMat, new Point(
//                0 + rightKnee.getPosition().x, 0 + rightKnee.getPosition().y), 3, new Scalar(0,255,0), 5);
//        Imgproc.circle(croppedMat, new Point(
//                0 + leftAnkle.getPosition().x, 0 + leftAnkle.getPosition().y), 3, new Scalar(255,255,0), 5);
//        Imgproc.circle(croppedMat, new Point(
//                0 + rightAnkle.getPosition().x, 0 + rightAnkle.getPosition().y), 2, new Scalar(255,255,0), 3);
//
//
//        Imgcodecs.imwrite("/sdcard/todelete.jpg", croppedMat);


        // compute average of torso height
        float returnvalue=0.0f;
        torsoHeightHistory.add(Math.abs(leftKnee.getPosition().y - leftShoulder.getPosition().y));
        // wait to fill the array
        if (torsoHeightHistory.size()>NUM_OF_TORSOHEIGHT_HISTORY) {
            //remove oldest entry
            torsoHeightHistory.remove(0);

            // compute average
            float sum = 0.0f;
            for (int i = 0; i < NUM_OF_TORSOHEIGHT_HISTORY; i++) {
                sum += torsoHeightHistory.get(i);
            } //next value
            returnvalue = sum / torsoHeightHistory.size();
        } //end if

        return returnvalue;

    }

    /**
     * Start the torso height estimation of the target in a background task
     */
    public void startTorsoHeightEstimation()
    {
        if(poseScheduler==null || poseScheduler.isShutdown())
        {
            try{
                //init background worker
                poseScheduler = Executors.newScheduledThreadPool(1);
                poseScheduler.scheduleWithFixedDelay(poseRunnable, 0, 100, TimeUnit.MILLISECONDS);
            }
            catch (Exception e)
            {
                Log.e(TAG, "ERROR starting torso estimation: " + Log.getStackTraceString(e));
            }
        }//end if scheduler ready

    } //end start torso height estimation


    /**
     * Stop the background task for torso height estimation
     */
    public void stopTorsoHeightEstimation()
    {
        // stop scheduler
        try{
            poseScheduler.shutdown();
        }
        catch (Exception e)
        {
            Log.e(TAG, "ERROR stopping torso estimation: " + Log.getStackTraceString(e));
        }
    } //end stop torso estimation


    public class TrackedObject
    {
        public Rect box = new Rect();
        public int objectClass = -1;
        public float score = 0.0f;

    }

    /**
     * Set minimimun face/head size to track, before switiching to upper body
     * @param size size in pixel
     */
    public void setMinHeadSize(int size)
    {
        this.MIN_HEAD_SIZE = size;
    }

    /**
     * Set minimimun hard limit of body size to track, to avoid cropping too much from the person detection
     * @param size size in pixel
     */
    public void setMinBodySize(int size)
    {
        this.MIN_HUMAN_HEIGHT = size;
    }
}
