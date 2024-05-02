package com.bfr.opencvapp.grafcet;


import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.utils.bfr_Grafcet;

public class InitGrafcet extends bfr_Grafcet {

    public InitGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;
    }

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    // Response callback from USB
    String ackYes="";
    String ackNo="";
    String ackWheels="";

    private IUsbCommadRsp wheelsRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) { ackWheels = success; }

        @Override
        public void onFailed(String error) { ackWheels = error; }
    };

    private IUsbCommadRsp yesRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) { ackYes = success; }
        @Override
        public void onFailed(String error) { ackYes = error; }
    };

    private IUsbCommadRsp noRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) { ackNo = success; }
        @Override
        public void onFailed(String error) { ackNo = error; }
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

                        }
                        break;

                    case 20 : // reset head position

                        //reset
                        ackYes="";
                        ackNo="";

                        BuddySDK.USB.buddySayYes(50.0f, 30.0f, yesRsp);

                        BuddySDK.USB.buddySayNo(50.0f, 0.0f, noRsp);
                        go = false;
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
