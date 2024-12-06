package com.bfr.opencvapp.utils;

import static com.bfr.opencvapp.utils.HandPoseLandmarks.*;

import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoWriter;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public class HandGestSeqRecognizer extends HandlerThread {

    String TAG = "HandIMGRecorder";

    @Override
    protected void onLooperPrepared()
    {
        initHandler();
    }
    void initHandler() {
        poseEstHandler = new Handler(getLooper()) {
            @Override
            public void handleMessage(Message msg)
            {}
        };
    }

    public HandGestSeqRecognizer(HandPoseEstimator handPoseEstimator){
        super("HandImageREcorder");
        this.handPoseEstimator = handPoseEstimator;
        handPose = new HandPose();
    }

    HandPoseEstimator handPoseEstimator;
    HandPose handPose = null;
    List<Mat> listOfMat = new ArrayList<>();
    int imgIdx = 0;
    int NUM_OF_IMG = 15;

    List<HandPose> listOfHandpose = new ArrayList<>();

    double elapsed_time=0;
    /** record for debog*/

    LocalDateTime myDateObj = LocalDateTime.now();
    DateTimeFormatter myFormatObj = DateTimeFormatter.ofPattern("yyMMddHHmmss");
    VideoWriter videoWriter;
    String formattedDate = myDateObj.format(myFormatObj);
    String debugFileName = "/storage/emulated/0/Download/" + formattedDate + "_trackingDebug.avi" ;
    int fourcc =-1;
    /****/

    Handler poseEstHandler = new Handler();
    //Element to display frame from Camera
    Mat imgToAdd;
    private  Runnable poseEstimation = new Runnable() {
        @Override
        public void run() {
            try {
                synchronized (listOfMat){
                    elapsed_time = System.currentTimeMillis();
                    Log.i(TAG, "handpose before " + (listOfMat.size()-1));
                    handPose = handPoseEstimator.recognizeImage(listOfMat.get(listOfMat.size()-1));

                    listOfHandpose.add(handPose);

                    if(handPose!=null) {
                        Log.i(TAG, "handpose after " + (listOfMat.size() - 1) + " (" + (System.currentTimeMillis() - elapsed_time) + "ms)");
                        int[] landmarks2draw = new int[]{WRIST, THUMB_TIP, THUMB_IP,THUMB_CMC, INDEX_TIP, INDEX_PIP,INDEX_MCP, MIDDLE_MCP, MIDDLE_PIP,MIDDLE_TIP, RING_MCP, RING_PIP, RING_TIP, PINKY_MCP, PINKY_PIP,PINKY_TIP};
                        Scalar[] colors = new Scalar[]{new Scalar(255,255,255), new Scalar(150,150,0), new Scalar(150,150,0), new Scalar(150,150,0), new Scalar(255,0,0), new Scalar(255,0,0), new Scalar(255,0,0), new Scalar(0,255,0), new Scalar(0,255,0),  new Scalar(0,255,0), new Scalar(0,0,255), new Scalar(0,0,255), new Scalar(0,0,255), new Scalar(150,0,150), new Scalar(150,0,150), new Scalar(150,0,150)};
                        for(int f=0; f<landmarks2draw.length;f++) {

                            Imgproc.circle(listOfMat.get(listOfMat.size() - 1),
                                    new Point(handPose.landmarks.get(landmarks2draw[f]).x() * listOfMat.get(listOfMat.size() - 1).cols(), handPose.landmarks.get(landmarks2draw[f]).y() * listOfMat.get(listOfMat.size() - 1).rows()),
                                    2, colors[f], 3);

                        } //next landmark
                        Imgproc.putText(listOfMat.get(listOfMat.size() - 1), String.valueOf(handPose.handeness.get(0).categoryName()),
                                new Point(70, 70),
                                2, 1, new Scalar(255,255,255));
                    } //end if handpose null

                    Imgproc.putText(listOfMat.get(listOfMat.size() - 1), String.valueOf(listOfMat.size()-1) ,
                                new Point(30, 30),
                                2, 1, new Scalar(255,0,0));

                    videoWriter.write(listOfMat.get(listOfMat.size()-1));
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    };

    public void init(int width, int height){
        myDateObj = LocalDateTime.now();
        formattedDate = myDateObj.format(myFormatObj);
        debugFileName = "/storage/emulated/0/Download/trackingdebug/" + formattedDate + "_trackingDebug.avi" ;
        fourcc = VideoWriter.fourcc('M','J','P','G');
        Log.i(TAG, "videowriter creation " + debugFileName);
        videoWriter = new VideoWriter(debugFileName, fourcc,
                3, new Size(width, height));
        Log.i(TAG, "Ready to save video " +width+"x"+height);

        listOfMat.clear();
        listOfHandpose.clear();

    }
    public void recImg(Mat img){

        synchronized (listOfMat){

            listOfMat.add(img);
            Log.i(TAG, "addedimg " + (listOfMat.size()-1));
            // queue handpose

//            poseEstHandler.post(poseEstimation);
            Thread t = new Thread(poseEstimation);
            t.start();


        }
        //Pose estimation

    }


    public String analyzeSeq(){

        // start analyzing from this pose
        int STARTING_IDX = 1;
        float minX=999f, minY=999f, maxX=-1f, maxY=-1f;
        float frontValue = 0;
        float knuck = 0, palm = 0;
        String hand ="";
        List<HandPose> previousHandposes = new ArrayList<>(listOfHandpose);

        boolean changedHorizDirection, openAndClosedFingers;
        float previousPos=0, currentPos=0, previousDirection=0, currentDirection=0;

        for (int i=STARTING_IDX; i<listOfHandpose.size(); i++){

            //sanity check
            if(listOfHandpose.get(i)==null)
                continue;

            previousDirection = currentDirection;
            currentDirection = listOfHandpose.get(i).landmarks.get(MIDDLE_TIP).x() + listOfHandpose.get(i).landmarks.get(RING_TIP).x() -previousDirection;


            if(listOfHandpose.get(i).landmarks.get(MIDDLE_TIP).x() + listOfHandpose.get(i).landmarks.get(RING_TIP).x() <minX){
                minX = listOfHandpose.get(i).landmarks.get(MIDDLE_TIP).x() + listOfHandpose.get(i).landmarks.get(RING_TIP).x();
            }
            if(listOfHandpose.get(i).landmarks.get(MIDDLE_TIP).x() + listOfHandpose.get(i).landmarks.get(RING_TIP).x()>maxX){
                maxX = listOfHandpose.get(i).landmarks.get(MIDDLE_TIP).x() + listOfHandpose.get(i).landmarks.get(RING_TIP).x();
            }
            if(listOfHandpose.get(i).landmarks.get(MIDDLE_TIP).y() + listOfHandpose.get(i).landmarks.get(RING_TIP).y()<minY){
                minY = listOfHandpose.get(i).landmarks.get(MIDDLE_TIP).y() + listOfHandpose.get(i).landmarks.get(RING_TIP).y();
            }
            if(listOfHandpose.get(i).landmarks.get(MIDDLE_TIP).y() + listOfHandpose.get(i).landmarks.get(RING_TIP).y()>maxY){
                maxY =listOfHandpose.get(i).landmarks.get(MIDDLE_TIP).y() + listOfHandpose.get(i).landmarks.get(RING_TIP).y();
            }

//            if(listOfHandpose.get(i).isFront())
//                isFront += 1;
//            else
//                isFront-=1;

            Log.i("gestanalyze", "frontvalue="+listOfHandpose.get(i).getFrontValue()
                    + "minX="+minX
                    + "maxX="+maxX
                    + "minY="+minY
                    + "maxY="+maxY);
            frontValue += listOfHandpose.get(i).getFrontValue();
            knuck = listOfHandpose.get(i).debugknucle;
            palm = listOfHandpose.get(i).debugpalm;
            hand = listOfHandpose.get(i).debughandesness;
        }

        Log.i("gestanalyze", "min max " + minX + ";"+ maxX + ";"+ minY + ";"+ maxY + "; " + ((frontValue<=0)? "BACK" : "FRONT")  + "("+frontValue+")" + " knucle=" + knuck + " palm=" + palm +  "  " + hand);

        if (frontValue>0.1){
            if(maxX-minX>=0.40){
//                if(maxY-minY<=1.0)
                Log.i("gestanalyze", "     =======> COUCOU");
                return "COUCOU";
            }
        }else if (frontValue<=-0.2)
        {
            if(maxY-minY>=0.7){
                Log.i("gestanalyze", "     =======> Come Here");
                return "COME HERE";
            }
        }
        // if hand is open front and landmarks are moving a lot horizontally
        Log.i("gestanalyze", "   :( nothing recognized");
        return "";
    }

    public void saveVideo(){
        try {
            videoWriter.release();
        } catch (Exception e) {
        }
    }

}
