package com.bfr.opencvapp.utils;

import android.util.Log;

import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;

import java.util.ArrayList;
import java.util.List;

/**
 class returned by pose estimation
 containing
 - 63 landmarks *3 coords [x, y, z]; where x, y in PIXEL from the upper left corener of the input image, and z respective to the wrist
 - probability of hand presence
 - handedness: <0.5=left hand, >0.5 right hand

 */
public class HandPose{

    String TAG = "Handpose";

    // whether it is the left or right hand
    public List<Category> handeness = new ArrayList<>();
    // hand landmarks https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    public List<NormalizedLandmark> landmarks = null;
    // landmarks id of phalanxes see https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    int[][] PHALANX_ID = new int[][]{{4,2, 1}, {8,6, 5}, {12,10, 9}, {16,14, 13}, {20, 18, 17}};

    private boolean front = false;

    public float debugknucle, debugpalm;
    public String debughandesness="";

    /**
     * Checks whether the palm of the hand is facing the camera or not
     * @return true if the palm mis facing the camera, false otherwise
     */
    public boolean isFront()
    {
        // vector of the knucle line, from the base of the index to the base of the pinkie
        //int[] knucleLine = new int[]{ (int)( (landmarks.get(17).x() - landmarks.get(5).x())  * 1024), (int)((landmarks.get(17).y() - landmarks.get(5).y())*768 ), (int)((landmarks.get(17).z() - landmarks.get(5).z())*100 ) };
        float[] knucleLine = new float[]{ ( (landmarks.get(17).x() - landmarks.get(5).x()) ), ((landmarks.get(17).y() - landmarks.get(5).y()) ) };
        // vector of the palm, from the wrist to the base of the index
        float[] palmLine = new float[]{ ( (landmarks.get(0).x() - landmarks.get(5).x())), ( (landmarks.get(0).y() - landmarks.get(5).y()) )};

        Log.d(TAG, "Knuckle =" + knucleLine[0] +","+ knucleLine[1] + "\nPalm = " + palmLine[0] +","+ palmLine[1] );
        // Now check the direction of the edge of the plam and the edge of the knucle, depending on the hand
        debugknucle = knucleLine[0];
        debugpalm = palmLine[1];
        debughandesness = handeness.get(0).categoryName();
        // if left hand
        if (handeness.get(0).categoryName().toUpperCase().contains("LEFT")){

            if(knucleLine[0]*palmLine[1]<0){
                Log.i("gestanalyze", "LEFT FRONT");
                this.front = true;
            }
            else{
                Log.i("gestanalyze", "LEFT BACK");
                this.front = false;
            }
        }
        //else Right hand
        else{
            if(knucleLine[0]*palmLine[1]>0){

                Log.i("gestanalyze", "RIGHT FRONT");
                this.front = true;
            }
            else
            {

                Log.i("gestanalyze", "RIGHT BACK");
                this.front = false;
            }
        }
        return this.front;
    }

    /** How to know a finger is opened : compute the hypotenuse  of the tip and 2nd phalanx
     * if the dist [tip of the finger to the wrist] < the dist [2nd phalanx to the wrist]  => the finger is open
     * https://github.com/opencv/opencv_zoo/blob/main/models/handpose_estimation_mediapipe/demo.py#L209
     * for a point (x1, y1) the dist is simply  = sqrt(x1^2 + y1^2)
     * for instance, the first finger tip has the id 8 , and the 2nd phalanx id 6
     * https://github.com/opencv/opencv_zoo/blob/main/models/handpose_estimation_mediapipe/demo.py#L205
     */
    public boolean isOpen(HandPoseEstimator.FINGER finger)
    {

        // For the index
        if (finger == HandPoseEstimator.FINGER.INDEX)
        {
            int TIP = PHALANX_ID[finger.ordinal()][0];
            int SECOND_PHALANX = PHALANX_ID[finger.ordinal()][1];
            int KNUCKLE = PHALANX_ID[finger.ordinal()][2];

            double distTip = Math.sqrt( (landmarks.get(TIP).x() -  landmarks.get(0).x())*(landmarks.get(TIP).x() -  landmarks.get(0).x())
                    + (landmarks.get(TIP).y() -  landmarks.get(0).y())*(landmarks.get(TIP).y() -  landmarks.get(0).y()) );

            double distPhalanx = Math.sqrt( (landmarks.get(SECOND_PHALANX).x() -  landmarks.get(0).x())*(landmarks.get(SECOND_PHALANX).x() -  landmarks.get(0).x())
                    + (landmarks.get(SECOND_PHALANX).y() -  landmarks.get(0).y())*(landmarks.get(SECOND_PHALANX).y() -  landmarks.get(0).y()) );

            double distPhalanxBase = Math.sqrt( (landmarks.get(SECOND_PHALANX).x() -  landmarks.get(KNUCKLE).x())*(landmarks.get(SECOND_PHALANX).x() -  landmarks.get(KNUCKLE).x())
                    + (landmarks.get(SECOND_PHALANX).y() -  landmarks.get(KNUCKLE).y())*(landmarks.get(SECOND_PHALANX).y() -  landmarks.get(KNUCKLE).y()) );

            double distPhalanxes = Math.sqrt( (landmarks.get(TIP).x() -  landmarks.get(SECOND_PHALANX).x())*(landmarks.get(TIP).x() -  landmarks.get(SECOND_PHALANX).x())
                    + (landmarks.get(TIP).y() -  landmarks.get(SECOND_PHALANX).y())*(landmarks.get(TIP).y() -  landmarks.get(SECOND_PHALANX).y()) );

            if (distTip <= distPhalanx //tip of the finger closer to the wriste
                    || distPhalanxes<distPhalanxBase // idem when the knucle is in front of the camera
            )
                return  false;
            else
                return true;
        }
        else if(finger== HandPoseEstimator.FINGER.THUMB)// THUMB is an exception :
        // the open state of the thumb is obtained by comparing the dist of the tip to the base of the index finger the dist of the first two knuckles
        {

            int TIP = 4;
            int INDEX_BASE = 5;
            int MIDDLE_BASE = 9;

            double distTip = Math.sqrt( (landmarks.get(TIP).x() -  landmarks.get(INDEX_BASE).x())*(landmarks.get(TIP).x() -  landmarks.get(INDEX_BASE).x())
                    + (landmarks.get(TIP).y() -  landmarks.get(INDEX_BASE).y())*(landmarks.get(TIP).y() -  landmarks.get(INDEX_BASE).y()) );
            double distKnuckle = Math.sqrt( (landmarks.get(MIDDLE_BASE).x() -  landmarks.get(INDEX_BASE).x())*(landmarks.get(MIDDLE_BASE).x() -  landmarks.get(INDEX_BASE).x())
                    + (landmarks.get(MIDDLE_BASE).y() -  landmarks.get(INDEX_BASE).y())*(landmarks.get(MIDDLE_BASE).y() -  landmarks.get(INDEX_BASE).y()) );

            // if tip of the thumb is close to the base of the middle finger
            if (distTip <= 2*distKnuckle)
            {
                String.format("%1$,.2f", distTip);
//                    Log.d("ccoucou", "distTip=" + String.format("%1$,.4f", distTip) + "distPhalanx=" + String.format("%1$,.4f", distKnuckle)  + "=> CLOSE");
                return  false;
            }
            else
            {
//                    Log.d("ccoucou", "distTip=" + String.format("%1$,.4f", distTip) + "distPhalanx=" + String.format("%1$,.4f", distKnuckle)  + "=> OPEN");
                return true;
            }

        }
        else // for the other fingers
        {
            int TIP = PHALANX_ID[finger.ordinal()][0];
            int SECOND_PHALANX = PHALANX_ID[finger.ordinal()][1];
            int KNUCKLE = PHALANX_ID[finger.ordinal()][2];

            double distTip = Math.sqrt( (landmarks.get(TIP).x() -  landmarks.get(0).x())*(landmarks.get(TIP).x() -  landmarks.get(0).x())
                    + (landmarks.get(TIP).y() -  landmarks.get(0).y())*(landmarks.get(TIP).y() -  landmarks.get(0).y()) );

            double distPhalanx = Math.sqrt( (landmarks.get(SECOND_PHALANX).x() -  landmarks.get(0).x())*(landmarks.get(SECOND_PHALANX).x() -  landmarks.get(0).x())
                    + (landmarks.get(SECOND_PHALANX).y() -  landmarks.get(0).y())*(landmarks.get(SECOND_PHALANX).y() -  landmarks.get(0).y()) );

            if (distTip <= distPhalanx //tip of the finger closer to the wriste
            )
                return  false;
            else
                return true;
        }

    } //end isOpen


    /**
     * returns the orientation of the finger as an angle in degrees [0-359], in the camera optical plane. 0 is horizontal, in anti-clockwise direction (so 90° si upward and -90° if downward)
     * @param finger the finger to compute the orientation
     * @return the angle in degree
     */
    public int fingerOrientation(HandPoseEstimator.FINGER finger)
    {

        int TIP = PHALANX_ID[finger.ordinal()][0];
        int SECOND_PHALANX = PHALANX_ID[finger.ordinal()][1];

         double angleRad= Math.atan2( (landmarks.get(TIP).y() -  landmarks.get(SECOND_PHALANX).y()) ,  (landmarks.get(TIP).x() -  landmarks.get(SECOND_PHALANX).x()) );

        // image is oriented with y towards bottom -> invert sign
        return -(int)Math.toDegrees(angleRad);

    } //end finger orientation


} //end handpose class