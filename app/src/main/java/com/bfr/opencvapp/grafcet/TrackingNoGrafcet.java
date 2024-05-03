package com.bfr.opencvapp.grafcet;


import static com.bfr.opencvapp.MainActivity.personTrackerVIT;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.BboxCentroid;
import com.bfr.opencvapp.utils.bfr_Grafcet;

/***
 * This grafcet aligns the NO with the target
 */
public class TrackingNoGrafcet extends bfr_Grafcet{

    public TrackingNoGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;
    }

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    final int STABILIZATION_TIME = 1000;
    public static boolean waitingForAlign = false;

    private int previous_step = -1;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    BboxCentroid target = new BboxCentroid();


    public static float noOffset=0.0f;
    float NO_OFFSET_THRES = 0.7f;
    float previousOffset=0.0f;
    float noAngle=0.0f;
    float noSpeed = 30.0f;
    float BASE_SPEED = 30.0f;
    float accFactor = 1.0f;

    String ackNo = "";

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

    // runable for grafcet
    private Runnable mysequence = new Runnable()
    {
        @Override
        public void run()
        {

            try{

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
                        target.getCentroid(personTrackerVIT.tracked.box.x,
                                personTrackerVIT.tracked.box.y,
                                personTrackerVIT.tracked.box.height,
                                personTrackerVIT.tracked.box.width
                        );

                        // compute angle for the wideAngle camera
                        // resolution of 1024x768, with a 120° aperture
                        // => 1pixel ~= 120 / sqrt(1024^2+768^2) = 0.09375
                        noOffset = (target.x-(1024/2))*0.09375f;

                        // if target off center
                        if(Math.abs(noOffset)>NO_OFFSET_THRES)
                            step_num = 20;
                        break;

                    case 20: // move head to look at target

                        //reset
                        ackNo = "";
                        previousOffset = noOffset;

                        if (noOffset>0)
                            noAngle = 150.0f; // turn head max to the right
                        else
                            noAngle = -150.0f; // turn max the other way

                        //move head
                        BuddySDK.USB.buddySayNo(BASE_SPEED, noAngle, noRsp);

                        step_num = 28;
                        break;

                    case 25: // wait for OK
                        if (ackNo.contains("OK"))
                        {
                            step_num = 28;
                        }
                        break;


                    case 28 : // wait for target in range
                        target.getCentroid(personTrackerVIT.tracked.box.x,
                                personTrackerVIT.tracked.box.y,
                                personTrackerVIT.tracked.box.height,
                                personTrackerVIT.tracked.box.width
                        );

                        // compute angle for the wideAngle camera
                        // resolution of 1024x768, with a 120° aperture
                        // => 1pixel ~= 120 / sqrt(1024^2+768^2) = 0.09375
                        noOffset = (target.x-(1024/2))*0.09375f;

                        // if target in range
                        if (Math.abs(noOffset)<5.0f)
                        {
                            Log.d(name, "offset = " + noOffset + " -> STOP");
                            // stop head rotation
                            BuddySDK.USB.buddyStopNoMove(noRsp);
                            // go to stabilization step
                            step_num = 60;
                        }
                        else // target not in range
                        {
                            // if No not moving
                            if (ackNo.toUpperCase().contains("FINISHED"))
                            {
                                //if No at maximum position
                                if (Math.abs(BuddySDK.Actuators.getNoPosition()) >=59) //empiric value which defines max No position
                                {
                                    // make body move
                                    AlignBodyGrafcet.rotationRequest = true;
                                }
                                else // No not moving for unknown reason, while target still not in range
                                {
                                    Log.d(name, "No not moving + No position=" + Math.abs(BuddySDK.Actuators.getNoPosition()) );
                                    //reset
                                    step_num = 59;
                                }

                            }
                            else // still moving > adjusting speed
                            {
                                // if current offset>= previous offset => target is moving
                                if (Math.abs(noOffset)-Math.abs(previousOffset)>1)
                                {
                                    Log.d(name, "offset is moving: "
                                            + Math.abs(noOffset) + "-"+ Math.abs(previousOffset)
                                            +"=" +(Math.abs(noOffset)-Math.abs(previousOffset)));

                                    // rotate head faster
                                    accFactor = Math.abs(noOffset)-Math.abs(previousOffset);
                                    step_num = 30;
                                }
                            } //end if still moving


                        } //end if target not in range

                        break;


                    case 30: // moving target adjust speed
                        //reset
                        ackNo = "";
                        previousOffset = noOffset;

                        if (noOffset>0)
                            noAngle = 150.0f; // turn head max to the right
                        else
                            noAngle = -150.0f; // turn max the other way

                        // adjust speed
                        noSpeed = accFactor*BASE_SPEED;
                        // hard limit
                        if (noSpeed>60.0f)
                            noSpeed=60.0f;

                        Log.d(name, "rotating to " + noAngle + " (offset=" + noOffset +") at " + noSpeed);
                        //move head
                        BuddySDK.USB.buddySayNo(BASE_SPEED, noAngle, noRsp);

                        step_num = 28;
                        break;

                    case 59: // No not moving and Target off range-> check if need to align body before restarting

                        // if head is turned
                        if (Math.abs(BuddySDK.Actuators.getNoPosition())>5)
                        {
                            TrackingNoGrafcet.waitingForAlign = true;
                            step_num = 65;
                        }
                        else // No axis  aligned with body
                        {
                            step_num = 10;
                        }
                        break;

                    case 60 : // wait around 1s to see if target stable

                        target.getCentroid(personTrackerVIT.tracked.box.x,
                                personTrackerVIT.tracked.box.y,
                                personTrackerVIT.tracked.box.height,
                                personTrackerVIT.tracked.box.width
                        );

                        // compute angle for the wideAngle camera
                        // resolution of 1024x768, with a 120° aperture
                        // => 1pixel ~= 120 / sqrt(1024^2+768^2) = 0.09375
                        noOffset = (target.x-(1024/2))*0.09375f;

                        // if target moving
                        if(Math.abs(noOffset)>5.0f)
                        {
                            //move head
                            step_num = 20;
                        }
                        else{
                            // wait during a stabilization time
                            // if time spent waiting > stabilization time
                            if(System.currentTimeMillis()-time_in_curr_step > STABILIZATION_TIME)
                            {
                                // go to sync step with aligning body
                                waitingForAlign = true;
                                step_num = 65;
                            }
                        }
                        break;

                    case 65: // wait end of aligning body
                        if (!waitingForAlign)
                            step_num = 10;

                        // if target is moving
                        //=> cancel body rotation

                        target.getCentroid(personTrackerVIT.tracked.box.x,
                                personTrackerVIT.tracked.box.y,
                                personTrackerVIT.tracked.box.height,
                                personTrackerVIT.tracked.box.width
                        );

                        // compute angle for the wideAngle camera
                        // resolution of 1024x768, with a 120° aperture
                        // => 1pixel ~= 120 / sqrt(1024^2+768^2) = 0.09375
                        noOffset = (target.x-(1024/2))*0.09375f;

                        if(Math.abs(noOffset)>=10)
                        {
                            Log.w(name, " interruption because offset= " + noOffset);
                            // stop wheels
                            BuddySDK.USB.emergencyStopMotors(noRsp);

                            AlignBodyGrafcet.rotationRequest = false;
                            AlignBodyGrafcet.step_num = 10;

                            step_num = 20;
                        }
                        break;


                    default :
                        // go to next step
                        step_num = 0;
                        break;
                } //End switch


            } catch (Exception e) {
                e.printStackTrace();
            }

        } // end run
    }; // end new runnable

}
