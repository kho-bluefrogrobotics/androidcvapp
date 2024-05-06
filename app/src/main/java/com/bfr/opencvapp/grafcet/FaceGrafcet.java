package com.bfr.opencvapp.grafcet;


import static com.bfr.opencvapp.MainActivity.personTrackerVIT;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.ui.shared.FacialEvent;
import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.utils.bfr_Grafcet;

/***
 * This grafcet manages the Face :
 * it makes it blink every xxx seconds (random)
 * it makes the eyes follow the target
 */
public class FaceGrafcet extends bfr_Grafcet {

    public FaceGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }


    private FaceGrafcet grafcet=this;

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    public static boolean obstacleFaceEvtReq = false;

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    private double timeSinceLastBlink = 0;
    private double randomBlinkInterval = 4000;

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
                    // Log.d(name, "current step: " + step_num + "  ");
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


                // blink every xxx random amount of time
                if (System.currentTimeMillis()-timeSinceLastBlink >randomBlinkInterval)
                {
                    // reset
                    timeSinceLastBlink = System.currentTimeMillis();
                    // compute next blink in random timelapse (min 3s)
                    randomBlinkInterval = (int) (Math.random()*6000)+3000;

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

                    case 5: //make the eyes follow the target

                        // scaling the tracked box between 0;1
                        // scaling a value v =[min1;max1] to a range [min2; max2]
                        // new_value= ( v - min1)  * [ ( max2-min2)/(max1-min1) ] + min2

                        // empirically, we observe the tracked box horizontal position of its center is between 200;830
                        float centerPosX = (float)(personTrackerVIT.tracked.box.x  + personTrackerVIT.tracked.box.width/2);
                        // changing the range
                        float scaleX = (1.0f-0.0f) / (830 - 200.0f);
                        // !!!the tracking is mirrored tracking.x = 0 -> position value must be 1
                        float xpos = (( 200.0f- centerPosX)*scaleX + 1.0f) ;
                        // the final value for the eyes mus be between 0;1300
                        xpos = xpos *1300;

                        // same thing for Y
                        float centerPosY = (float)(personTrackerVIT.tracked.box.y ); //pointing to the top of the bbox
                        float scaleY = (0.7f-0.3f) / (350 - 150.0f);
                        // Y is not inverted
                        float ypos = (( centerPosY - 150.0f)*scaleY + 0.3f) ;
                        ypos = ypos * 900;

                        // move eyes
                        BuddySDK.UI.lookAtXY(xpos, ypos , true);

                        step_num = 10;
                        break;

                    case 10: // obstacle behind, make the eyes look up

                        // if obstacle behind
                        if(obstacleFaceEvtReq) {
                            BuddySDK.UI.lookAtXY(-100, -100, true);
                            step_num = 30;
                        }
                        else // continue with eyes following target
                            step_num = 5;
                        break;


                    case 30 : //play facial event for obstacle behind
                        // facial event
                        Thread.sleep(500);

                        BuddySDK.UI.playFacialRelativeEvent();
                        // Blink leds in red
                        BuddySDK.USB.blinkAllLed("#ff1100", 500, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {}

                            @Override
                            public void onFailed(String s) throws RemoteException {}
                        });
                        //wait
                        Thread.sleep(1000);
                        // reset leds to blue
                        BuddySDK.USB.updateAllLed("#61E3EB",  new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {}

                            @Override
                            public void onFailed(String s) throws RemoteException {}
                        });

                        //reset
                        obstacleFaceEvtReq = false;
                        step_num = 5;
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
