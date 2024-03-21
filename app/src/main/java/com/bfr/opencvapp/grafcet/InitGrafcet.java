package com.bfr.opencvapp.grafcet;


//import static com.bfr.opencvapp.MainActivity.alignCheckbox;

import static com.bfr.opencvapp.MainActivity.personTracker;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.ui.shared.FacialEvent;
import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import org.opencv.core.Point;

public class InitGrafcet extends bfr_Grafcet {

    public InitGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }


    private InitGrafcet grafcet=this;

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;
    final static int INTERVAL_MIN = 350;
    final static int INTERVAL_MAX = 450;
    private int mIntervalleHist = INTERVAL_MIN;
    private float speed = 10F;

    public static boolean rotationRequest = false;

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    private double timeSinceLastBlink = 0;
    private double randomBlinkInterval = 4000;

    public static int RESIZE_RATIO =20;
    public static double xCenter =0.0;
    private double xorig=0.0;
    private double deltaPixel=0.0;

    private float angleToRotate=0.0f;


    private IUsbCommadRsp wheelsRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) throws RemoteException { ackWheels = success; }

        @Override
        public void onFailed(String error) throws RemoteException { ackWheels = error; }
    };

    private IUsbCommadRsp yesRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) throws RemoteException { ackYes = success; }
        @Override
        public void onFailed(String error) throws RemoteException { ackYes = error; }
    };

    private IUsbCommadRsp noRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) throws RemoteException { ackNo = success; }
        @Override
        public void onFailed(String error) throws RemoteException { ackNo = error; }
    };
    String ackYes="";
    String ackNo="";
    String ackWheels="";

    // Define the sequence/grafcet to be executed
   /* This provides a template for a grafcet.
   The sequence is as follows:
   - check the checkbox
   - Move the No from Left to right
   - Move the no from right to left
   - If the check box is unchecked then stop
   - if not, repeat
    */
    // runable for grafcet
    private Runnable mysequence = new Runnable()
    {
        @Override
        public void run()
        {

            try {

                // if step changed
                if (!(step_num == previous_step)) {
                    // display current step
                    Log.i(name, "current step: " + step_num + "  ");
                    // update
                    previous_step = step_num;

                    // start counting time in current step
                    time_in_curr_step = System.currentTimeMillis();
                    //reset bypass
                    timeout = false;
                } // end if step = same
                else
                {
                    // if time > 2s
                    if ((System.currentTimeMillis()-time_in_curr_step > 5000) && step_num >0)
                    {
                        // activate bypass
                        timeout = true;
                    }
                }


                // blink ever 4s
                if (System.currentTimeMillis()-timeSinceLastBlink >randomBlinkInterval)
                {
                    // reset
                    timeSinceLastBlink = System.currentTimeMillis();
                    // compute next blink in random timelapse
                    randomBlinkInterval = (int) (Math.random()*6000)+3000;

                    Log.d(name, "Next Blink in " + randomBlinkInterval +"s");
                    //blink
                    BuddySDK.UI.playFacialEvent(FacialEvent.BLINK_EYES);
                }

                // which grafcet step?
                switch (step_num) {
                    case 0: // Wait for checkbox
                        //wait until check box
                        if (go) {
                            // go to next step
                            step_num = 5;
                        }
                        break;

                    case 5: //enable all motors

                        //reset
                        ackYes="";
                        ackNo="";
                        ackWheels="";

                        BuddySDK.USB.enableWheels(true, wheelsRsp);
                        BuddySDK.USB.enableNoMove(true, noRsp);
                        BuddySDK.USB.enableYesMove(true, yesRsp);

                        step_num = 10;
                        break;

                    case 10: //
                        if (ackWheels.toUpperCase().contains("OK")
                        && ackYes.toUpperCase().contains("OK")
                        && ackNo.toUpperCase().contains("OK"))
                        {
                            step_num = 15;
                        }
                        break;

                    case 15 : //wait for  wheels
                        if (!BuddySDK.Actuators.getLeftWheelStatus().toUpperCase().contains("DISABLE"))
                            step_num = 17;
                        break;

                    case 17 : //wait for  Yes
                        if (!BuddySDK.Actuators.getYesStatus().toUpperCase().contains("DISABLE"))
                            step_num = 18;
                        break;
                    case 18 : //wait for  Yes
                        if (!BuddySDK.Actuators.getNoStatus().toUpperCase().contains("DISABLE")) {
                            step_num = 20;
                        go = false;
                        }
                        break;

                    default:
                        // go to next step
                        step_num = 0;
                        break;
                } //End switch


            }//end try
            catch (Exception e) {
               Log.e(name, Log.getStackTraceString(e));
            }

        } // end run
    }; // end new runnable


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

}
