package com.bfr.opencvapp.grafcet;


import static com.bfr.opencvapp.MainActivity.personTrackerVIT;

import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.BboxCentroid;
import com.bfr.opencvapp.utils.bfr_Grafcet;


/**
 * Align the Yes to keep the target in the center of the camera
 */
public class TrackingYesGrafcet extends bfr_Grafcet{

    public TrackingYesGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    private int previous_step = -1;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    // centroid of tracked target
    BboxCentroid target = new BboxCentroid();



    public static float yesOffset =0.0f;
    float YES_OFFSET_THRES = 5.0f;
    float yesAngle =0.0f;
    float yesSpeed = 30.0f;
    float BASE_SPEED = 30.0f;
    float accFactor = 1.0f;

    String yesAck = "";
    private IUsbCommadRsp yesRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) {
            Log.i(name, "MotorYES : " + success);
            yesAck = success;
        }

        @Override
        public void onFailed(String error) {
            Log.i(name, "MotorYES : " + error);
            yesAck = error;
        }
    };

    // runable for grafcet
    private Runnable mysequence = new Runnable()
    {
        @Override
        public void run()
        {

            // if step changed
            if( !(step_num == previous_step)) {
                // display current step
                Log.i(name, "current step: " + step_num + "  ");
                // update
                previous_step = step_num;
                // start counting time in current step
                time_in_curr_step = System.currentTimeMillis();
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


            // which grafcet step?
            switch (step_num) {
                case 0: // Wait for checkbox
                    //wait until check box
                    if (go) {
                        // go to next step
                        step_num = 10;
                    }
                    break;


                case 10: // get target position

                    if (personTrackerVIT.tracked.objectClass==0) // if tracking a human silouhette
                        target.y = Math.max(0,(int) (personTrackerVIT.tracked.box.y+ personTrackerVIT.tracked.box.height/4));
                    else // tracking a face
                        target.y = Math.max(0,(int) (personTrackerVIT.tracked.box.y+ personTrackerVIT.tracked.box.height));

                    // compute angle for the wideAngle camera
                    // resolution of 1024x768, with a 120° aperture
                    // => 1pixel ~= 120 / sqrt(1024^2+768^2) = 0.09375
                    yesOffset = (target.y-(768/2))*0.09375f;

                    // if target off centered
                    if(Math.abs(yesOffset)>YES_OFFSET_THRES)
                        step_num = 20;
                    break;

                case 20: // move head
                    //reset
                    yesAck = "";

                    yesAngle = Math.max(-5, BuddySDK.Actuators.getYesPosition()- yesOffset);

                    Log.d(name, "rotating to " + yesAngle + " (offset=" + yesOffset +") with Yes position = " + BuddySDK.Actuators.getYesPosition() + " at " + yesSpeed);

                    BuddySDK.USB.buddySayYes(BASE_SPEED, yesAngle, yesRsp);
                    step_num = 25;
                    break;

                case 25: // wait for OK
                    if (yesAck.toUpperCase().contains("OK") || timeout)
                    {
                        step_num = 27;
                    }
                    break;


                case 27 : // waiting for end of mvt
                    if (yesAck.toUpperCase().contains("FINISHED") || timeout)
                    {
                        step_num = 10;
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
