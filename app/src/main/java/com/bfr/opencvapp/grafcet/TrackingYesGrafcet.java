package com.bfr.opencvapp.grafcet;


import android.os.IBinder;
import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
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
//            Log.i("GRAFCET YES", "current Y: " + tracked.y + "  ");
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




                default :
                    // go to next step
                    step_num = 0;
                    break;
            } //End switch

        } // end run
    }; // end new runnable

}
