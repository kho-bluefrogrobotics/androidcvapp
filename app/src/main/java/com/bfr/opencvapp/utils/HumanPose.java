package com.bfr.opencvapp.utils;

import static com.bfr.opencvapp.utils.HumanPoseLandmarks.LEFT_HIP;
import static com.bfr.opencvapp.utils.HumanPoseLandmarks.LEFT_SHOULDER;
import static com.bfr.opencvapp.utils.HumanPoseLandmarks.LEFT_WRIST;
import static com.bfr.opencvapp.utils.HumanPoseLandmarks.RIGHT_HIP;
import static com.bfr.opencvapp.utils.HumanPoseLandmarks.RIGHT_SHOULDER;
import static com.bfr.opencvapp.utils.HumanPoseLandmarks.RIGHT_WRIST;

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;

import java.util.List;

/**
 class returned by pose estimation
 containing
 - 63 landmarks *3 coords [x, y, z]; where x, y in PIXEL from the upper left corener of the input image, and z respective to the wrist
 - probability of hand presence
 - handedness: <0.5=left hand, >0.5 right hand

 */
public class HumanPose{

    public List<NormalizedLandmark> landmarks = null;


//todelete
//    public int x(int landmark)
//    {
//        return (int)(landmarks.get(landmark).x()*1024);
//    }
//    public int y(int landmark)
//    {
//        return (int)(landmarks.get(landmark).y()*768);
//    }
//
    final float WRIST_VISIBILITY_THRES = 0.7f;

    /**
     * return the index of the Wrist landmark when the user is making a sign with the hand
     * (the user is considered signing if the wrist is closer to the shoulder than the hip)
     * @return LEFT_WRIST=15 if the user is making a hand sign with the left hand, or RIGHT_WRIST=16 if the user is signing with the right hand
     * returns -1 if the user is not signing
     */
    public int isSigning()
    {
        if( landmarks.get(LEFT_WRIST).visibility().get() > WRIST_VISIBILITY_THRES && (
             // if wrist is closer to the shoulder thant the hip
                Math.abs(landmarks.get(LEFT_SHOULDER).y()-landmarks.get(LEFT_WRIST).y() ) <= Math.abs(landmarks.get(LEFT_HIP).y()-landmarks.get(LEFT_WRIST).y())
        ))
        {
            return LEFT_WRIST;
        }
        else if(landmarks.get(RIGHT_WRIST).visibility().get() > WRIST_VISIBILITY_THRES && (
                    // if wrist is closer to the shoulder thant the hip
                Math.abs(landmarks.get(RIGHT_SHOULDER).y()-landmarks.get(RIGHT_WRIST).y() ) <= Math.abs(landmarks.get(RIGHT_HIP).y()-landmarks.get(RIGHT_WRIST).y())
        ))

        {
            return RIGHT_WRIST;
        }
        else
            return -1;
    } //end isSigning


    /**
     * checks if the user is singing with the specified hand
     * @param whichHand the index of the wrist to check, must be LEFT_WRIST or RIGHT_WRIST
     * @return true or false if the user is signing with specified hand
     */
    public boolean isSigning(int whichHand)
    {
        if(whichHand==LEFT_WRIST){
            return ( landmarks.get(LEFT_WRIST).visibility().get() > WRIST_VISIBILITY_THRES && (
                    // if wrist is closer to the shoulder thant the hip
                    Math.abs(landmarks.get(LEFT_SHOULDER).y()-landmarks.get(LEFT_WRIST).y() ) <= Math.abs(landmarks.get(LEFT_HIP).y()-landmarks.get(LEFT_WRIST).y())
            ));
        }
        else{
            return landmarks.get(RIGHT_WRIST).visibility().get() > WRIST_VISIBILITY_THRES && (
                    // if wrist is closer to the shoulder thant the hip
                    Math.abs(landmarks.get(RIGHT_SHOULDER).y()-landmarks.get(RIGHT_WRIST).y() ) <= Math.abs(landmarks.get(RIGHT_HIP).y()-landmarks.get(RIGHT_WRIST).y())
            );
        } //endif LEFT or RIGHT wrist
    }//end isSigning

} //end human pose class