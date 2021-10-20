package com.bfr.opencvapp.grafcet;


import android.os.IBinder;
import android.os.RemoteException;
import android.util.Log;

import com.bfr.usbservice.IUsbCommadRsp;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import org.opencv.core.Rect;


public class TrackingYesGrafcet extends bfr_Grafcet{

    public TrackingYesGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }



    private TrackingYesGrafcet grafcet=this;

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    //A CHANGER POUR LE YES
    final static int INTERVAL_MIN = 250;
    final static int INTERVAL_MAX = 350;
    private int mIntervalleHist = INTERVAL_MIN;
    private float speed = 10F;

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean bypass = false;

    public static Rect tracked=new Rect();

    // Last position where a face was detected
    public int lastValidPos = 0;
    // Lost faces
    public boolean lostFaces = false;

    //motor response
    private boolean isMoving = false;
    private float mMiddleRect;

    private IUsbCommadRsp iUsbCommadRsp = new IUsbCommadRsp.Stub(){

        @Override
        public void onSuccess(String success) throws RemoteException {
            Log.i("GRAFCET YES", "success --------------- : " + success);
        }

        @Override
        public void onFailed(String error) throws RemoteException {
            Log.i("GRAFCET YES", "error --------------- : " + error);

        }
    };

    // Define the sequence/grafcet to be executed
   /* This provides a template for a grafcet.
   The sequence is as follows:
   - check the checkbox
   - Move the yes from top to bottom
   - Move the no from bottom to top
   - If the check box is unchecked then stop
   - if not, repeat
    */
    // runable for grafcet
    private Runnable mysequence = new Runnable()
    {
        @Override
        public void run()
        {
            mMiddleRect = tracked.y /*+ (tracked.height/2)*/;
            Log.i("GRAFCET YES", "current Y: " + tracked.y + "  ");
            //Log.i("GRAFCET", "current step: " + step_num + "  ");

            // if step changed
            if( !(step_num == previous_step)) {
                // display current step
                Log.i("GRAFCET YES", "current step: " + step_num + "  ");
                // update
                previous_step = step_num;
            } // end if step = same


            // which grafcet step?
            switch (step_num) {
                case 0: // Wait for checkbox
                   /* try {
                        grafcet.mBuddySDK.getUsbInterface().buddyStopYesMove(iUsbCommadRsp);
                        grafcet.mBuddySDK.getUsbInterface().enableYesMove(0, iUsbCommadRsp);
                    } catch (RemoteException e) {
                        e.printStackTrace();
                    }*/
                    //wait until check box
                    if (go) {
                        // go to next step
                        step_num = 1;
                    }
                    break;

                case 1: // Enable yes
                    try {
                        grafcet.mBuddySDK.getUsbInterface().enableYesMove(1, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String success) throws RemoteException {
                                Log.i("enable yes SUCESS:", success);
                            }

                            @Override
                            public void onFailed(String error) throws RemoteException {
                                Log.e("enable yes FAILED", error);
                            }
                        });
                    } //end try enable
                    catch (RemoteException e) {
                        e.printStackTrace();
                    } // end catch

                    // go to next step
                    step_num = 2;
                    break;

                case 2: //wait for status yes enable
                    // if yes not disabled
                    if (!grafcet.mBuddyData.yesStatus.toUpperCase().contains("DISABLE")) {
                        // go to next step
                        step_num = 3;
                    }
                    break;

                case 3: // Move top or bottom or exit

                    // if checkbox unchecked
                    if (!go)
                    {
                        // go to next step
                        step_num = 10;
                    }
                    else if (lostFaces)
                    { // go to next step
                        step_num = 50;
                    }
                    if (mMiddleRect < INTERVAL_MIN )
                    {
                        Log.i("GRAFCET YES", "current tracked <200: ");

                        // go to next step
                        step_num = 20;
                    }
                    else if (mMiddleRect > INTERVAL_MAX)
                    {
                        Log.i("GRAFCET YES", "current tracked >400: ");

                        // go to next step
                        step_num = 30;
                    }
                    break;

                case 20: // Move to the top
                    Log.i("GRAFCET YES", "current tracked <200  step : " + step_num);

                    // start moving
                    try {
                        if(!isMoving)
                            grafcet.mBuddySDK.getUsbInterface().buddySayYesStraight(-speed, iUsbCommadRsp);
                        isMoving = true;

                    } catch (RemoteException e) {
                        e.printStackTrace();
                    }

                    // go to next step
                    step_num = 21;
                    break;

                case 21: // Wait for target to be in range

                    // if tracked target in range
                    if (mMiddleRect > INTERVAL_MIN && mMiddleRect < INTERVAL_MAX)
                    {
                        isMoving = false;
                        try {
                            grafcet.mBuddySDK.getUsbInterface().buddyStopYesMove(iUsbCommadRsp);

                            //grafcet.mBuddySDK.getUsbInterface().emergencyStopMotors(iUsbCommadRsp);
                        } catch (RemoteException e) {
                            e.printStackTrace();
                        }
                        // go to next step
                        step_num = 3;
                    }
                    else if(lostFaces) // if no face found
                        // go back to last position where a face was found
                    {
                        isMoving = false;
                        try {
                            grafcet.mBuddySDK.getUsbInterface().buddyStopYesMove(iUsbCommadRsp);
                        } catch (RemoteException e) {
                            e.printStackTrace();
                        }
                        // go to next step
                        step_num = 50;
                    }
                    break;

                case 30: // Move to the bottom
                    Log.i("GRAFCET YES", "current tracked >400  step : " + step_num);

                    // start moving
                    try {
                        if(!isMoving)
                            grafcet.mBuddySDK.getUsbInterface().buddySayYesStraight(speed, iUsbCommadRsp);
                        isMoving = true;
                    } catch (RemoteException e) {
                        e.printStackTrace();
                    }
                    // go to next step
                    step_num = 31;
                    break;

                case 31: // Wait for target to be in range

                    // if tracked target in range
                    if (mMiddleRect < INTERVAL_MAX && mMiddleRect > INTERVAL_MIN)
                    {
                        isMoving = false;
                        try {
                            grafcet.mBuddySDK.getUsbInterface().buddyStopYesMove(iUsbCommadRsp);
                        } catch (RemoteException e) {
                            e.printStackTrace();
                        }
                        // go to next step
                        step_num = 3;
                    }
                    else if(lostFaces) // if no face found
                    // go back to last position where a face was found
                    {
                        isMoving = false;
                        try {
                            grafcet.mBuddySDK.getUsbInterface().buddyStopYesMove(iUsbCommadRsp);
                        } catch (RemoteException e) {
                            e.printStackTrace();
                        }
                        // go to next step
                        step_num = 50;
                    }
                    break;


                case 10: // Disable motor

                    // disable motor
                    try {
                        grafcet.mBuddySDK.getUsbInterface().enableYesMove(0, iUsbCommadRsp);
                    } catch (RemoteException e) {
                        e.printStackTrace();
                    }
                    // go to next step
                    step_num = 11;
                    break;

                case 11: // Wait for motor to be disable

                    // if tracked target in range
                    if (grafcet.mBuddyData.yesStatus.toUpperCase().contains("DISABLE"))
                    {   // go to next step
                        step_num = 0;
                    }
                    break;

                case 50 : // move to last valid position
                    try {
                        grafcet.mBuddySDK.getUsbInterface().buddySayYes(20F, 0, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String success) throws RemoteException {            }

                            @Override
                            public void onFailed(String error) throws RemoteException {                  }
                        });
                    } catch (RemoteException e) {
                        e.printStackTrace();
                    }
                    // go to next step
                    step_num = 51;
                    break;

                case 51: // wait for end of movement
                    if (grafcet.mBuddyData.yesPos == lastValidPos) {
                        //reset
                        lostFaces = false;
                        // go to next step
                    step_num = 3;
                    }
                    // if finally found face
                    if (lostFaces == false) {
                        // go to next step
                        step_num = 3;
                    }
                    break;


                default :
                    // go to next step
                    step_num = 0;
                    break;
            } //End switch

        } // end run
    }; // end new runnable

}
