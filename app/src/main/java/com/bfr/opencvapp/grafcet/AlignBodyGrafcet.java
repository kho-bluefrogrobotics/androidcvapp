package com.bfr.opencvapp.grafcet;


import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.buddysdk.services.companion.TaskCallback;
import com.bfr.opencvapp.utils.bfr_Grafcet;

/***
 * This grafcet manages the body movements when in Watch-Me mode
 * it mainly align the body to the head, and make the robot rotate when the head cannot follow the target (NO at max position)
 */
public class AlignBodyGrafcet extends bfr_Grafcet {

    public AlignBodyGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    public static boolean rotationRequest = false;

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;


    String ackNo="";
    String ackWheels="";

    private IUsbCommadRsp wheelsRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) {
            Log.i(name, "success --------------- : " + success);
            ackWheels = success;
        }

        @Override
        public void onFailed(String error) {
            Log.i(name, "error --------------- : " + error);
            ackWheels = error;
        }
    };

    private IUsbCommadRsp noRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) {
            Log.i(name, "success --------------- : " + success);
            ackNo = success;
        }

        @Override
        public void onFailed(String error) {
            Log.i(name, "error --------------- : " + error);
            ackNo = error;
        }
    };

    private TaskCallback wheelsTskCbk =  new TaskCallback() {
        @Override
        public void onStarted() {
            ackWheels = "OK";
        }

        @Override
        public void onSuccess(String s) {
            ackWheels = "FINISHED";
        }

        @Override
        public void onCancel() {
            ackWheels = "FINISHED";
        }

        @Override
        public void onError(String s) {
            ackWheels = "ERROR";
        }
    };

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

                // which grafcet step?
                switch (step_num) {
                    case 0: // Wait for checkbox
                        //wait until check box
                        if (go) {
                            // go to next step
                            step_num = 10;
                        }
                        break;

                    case 5: //init wheels
                        BuddySDK.USB.enableWheels(1, 1, wheelsRsp);
                        step_num = 7;
                        break;

                    case 7: //wait for enabled
                        if (!BuddySDK.Actuators.getLeftWheelStatus().toUpperCase().contains("DISABLE")
                                && !BuddySDK.Actuators.getRightWheelStatus().toUpperCase().contains("DISABLE")) {
                            step_num = 10;
                        }
                        break;

                    case 10: // sync with trackingNo grafcet
                        // target is static -> align body with head
                        if (TrackingNoGrafcet.waitingForAlign)
                            step_num = 15;

                        // NO at maximum position -> request from No grafcet to rotate body
                        if(rotationRequest)
                            step_num = 50;

                        break;

                    case 15: // rotate body to align
                        ackWheels = "";
                        Log.d(name, "Rotating to " + BuddySDK.Actuators.getNoPosition());
                        // rotate
                        BuddySDK.USB.rotateNoPrecision(40.0f, -BuddySDK.Actuators.getNoPosition(), 0, wheelsTskCbk);

                        // compensate with head (NO)
                        ackNo = "";
                        BuddySDK.USB.buddySayNo(25.0f, 0.0f, noRsp);

                        step_num = 17;
                        break;

                    case 17: //wait for OK
                        if (ackWheels.toUpperCase().contains("OK") || timeout ) {
                            step_num = 20;
                        }
                        break;


                    case 20: // wait for end of mvt
                        if (ackWheels.toUpperCase().contains("FINISHED") || timeout ) {
                            if (ackNo.toUpperCase().contains("FINISHED")|| timeout )
                            {
                                //reset handshake
                                TrackingNoGrafcet.waitingForAlign = false;
                                // go to sync step
                                step_num = 10;
                            }
                        }
                        break;


                    case 50: //No at limit > request to rotate body

                        //reset
                        ackWheels = "";
                        BuddySDK.USB.rotateNoPrecision(50.0f, -TrackingNoGrafcet.noOffset, 0, wheelsTskCbk );

                        step_num = 53;
                        break;

                    case 53 ://wait for OK
                        if(ackWheels.toUpperCase().contains("OK") || timeout)
                            step_num = 55;
                        break;

                    case 55: // wait for mvt finished
                        if(ackWheels.toUpperCase().contains("FINISHED") || timeout) {
                            step_num = 10;
                            rotationRequest = false;
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


}
