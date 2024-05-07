package com.bfr.opencvapp.grafcet;

import static com.bfr.opencvapp.MainActivity.personTrackerVIT;
import static com.bfr.opencvapp.MainActivity.speedLinearGrafcet;

import android.content.Context;
import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.buddysdk.services.companion.TaskCallback;
import com.bfr.opencvapp.R;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import java.util.Random;

/***
 * This grafcet manages the Search person after the traccking is lost
 */
public class SearchPersonGrafcet extends bfr_Grafcet {

    Context context;

    public SearchPersonGrafcet(String mname) {
        this(mname, null);
    }

    public SearchPersonGrafcet(String mname, Context context) {
        super(mname);
        this.grafcet_runnable = mysequence;
        this.context = context;
    }


    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    private float noAngle=0.0f;

    int FONT_TOF_THRES = 999;
    int LATERAL_TOF_THRES = 999;

    public boolean obstacleL = false;
    public boolean obstacleR = false;
    public boolean obstacleM = false;

    String ackYes="";
    String ackNo="";
    String ackWheels="";
    // random Buddy vocal
    String[] arrayOfStrings;
    String randomString;

    private IUsbCommadRsp wheelsRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success){
            ackWheels = success;
            Log.d(name, "Motor wheels ack="+ackWheels);
        }

        @Override
        public void onFailed(String error) { ackWheels = error; }
    };

    private IUsbCommadRsp yesRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) { ackYes = success; Log.d(name, "Motor YES ack="+ackYes);}
        @Override
        public void onFailed(String error)  { ackYes = error;Log.d(name, "Motor YES ack="+ackYes); }
    };

    private IUsbCommadRsp noRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success){ ackNo = success; Log.d(name, "Motor NO ack="+ackNo);}
        @Override
        public void onFailed(String error) { ackNo = error; Log.d(name, "Motor NO ack="+ackNo);}
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


                /*** Compute obstacle detection */

                if ((BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() >15 && BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() < LATERAL_TOF_THRES) )
                    obstacleL = true;
                else
                    obstacleL = false;


                if( (BuddySDK.Sensors.TofSensors().FrontRight().getDistance() >15 && BuddySDK.Sensors.TofSensors().FrontRight().getDistance() < LATERAL_TOF_THRES) )
                    obstacleR = true;
                else
                    obstacleR = false;

                if( (BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() >15 && BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() < FONT_TOF_THRES) )
                    obstacleM = true;
                else
                    obstacleM = false;



                // which grafcet step?
                switch (step_num) {
                    case 0: // Wait for checkbox
                        //wait until check box
                        if (go) {
                            // go to next step
                            step_num = 5;
                        }
                        break;

                    case 5: // Say that buddy lost the target
                        arrayOfStrings = context.getResources().getStringArray(R.array.user_lost);
                        randomString = arrayOfStrings[new Random().nextInt(arrayOfStrings.length)];
                        BuddySDK.Speech.startSpeaking(randomString);
                        BuddySDK.USB.setBuddySpeed(0.0f, 0.0f, 0.2f,   wheelsRsp);
                        step_num = 20;
                        break;


                    case 7: // wait a bit to give a chance to the tracking
                        if(personTrackerVIT.isTracking)
                            step_num=900;
                        if (System.currentTimeMillis()-time_in_curr_step > 500)
                            step_num = 10;
                        break;

                    case 10 :// wait end of speech
                        if(BuddySDK.Speech.isReadyToSpeak())
                            step_num = 20;
                        break;

                    case 20: //Head at zero
                        //if tracking again then skip
                        if(personTrackerVIT.isTracking)
                        {
                            step_num = 0;
                            go = false;
                            break;
                        }

                        //reset
                        ackYes="";
                        ackNo="";

                        // reset head position
                        BuddySDK.USB.buddySayNo(40, noAngle, noRsp);
                        BuddySDK.USB.buddySayYes(40, 10, yesRsp);

                        if(obstacleL || obstacleR || obstacleM)
                            BuddySDK.USB.emergencyStopMotors(wheelsRsp);

                        step_num = 25;
                        break;


                    case 25 : //wait for wheels

                        if(obstacleL || obstacleR || obstacleM)
                            BuddySDK.USB.emergencyStopMotors(wheelsRsp);

                        if (ackWheels.toUpperCase().contains("FINISHED")
                                || BuddySDK.Actuators.getLeftWheelSpeed()<5
                                || timeout)
                            step_num = 27;
                        break;

                    case 27 : //wait for Yes

                        if(obstacleL || obstacleR || obstacleM)
                            BuddySDK.USB.emergencyStopMotors(wheelsRsp);

                        if (ackYes.toUpperCase().contains("RESPONDING")) // managing the msg "board not responding (timeout)"
                            step_num = 20;

                        if (ackYes.toUpperCase().contains("FINISHED")|| timeout)
                            step_num = 28;
                        break;

                    case 28 : //wait for No

                        if(obstacleL || obstacleR || obstacleM)
                            BuddySDK.USB.emergencyStopMotors(wheelsRsp);

                        if (ackNo.toUpperCase().contains("RESPONDING")) // managing the msg "board not responding (timeout)"
                            step_num = 20;

                        if (ackNo.toUpperCase().contains("FINISHED") ||
                        ( Math.abs(BuddySDK.Actuators.getNoPosition()) < 2  )
                                || timeout) {
                            step_num = 29;
                        }
                        break;

                    case 29:// calc no Angle

                        //if tracking OK skip
                        if(personTrackerVIT.isTracking) {
                            step_num = 100;
                            break;
                        }
                        if(BuddySDK.Actuators.getNoPosition()>=0)
                            noAngle = -40.0f;
                        else
                            noAngle = 40.0f;
                        step_num = 30;
                        break;

                    case 30: // turn head No
                        ackNo = "";
                        BuddySDK.USB.buddySayNo(20.0f, noAngle, noRsp);
                        step_num=32;
                        break;

                    case 32: // wait for OK

                        //if tracking OK skip
                        if(personTrackerVIT.isTracking) {
                            step_num = 100;
                            break;
                        }

                        if (ackNo.toUpperCase().contains("RESPONDING")) // managing the msg "board not responding (timeout)"
                            step_num = 30;

                        if(ackNo.toUpperCase().contains("OK") || timeout)
                            step_num = 33;

                        break;

                    case 33 : //wait for end of mvt

                        //if tracking OK skip
                        if(personTrackerVIT.isTracking) {
                            step_num = 100;
                            break;
                        }

                        if(ackNo.toUpperCase().contains("FINISHED") || timeout)
                            step_num = 50;
                        break;


                    case 40: // Align body and Head

                        //if tracking OK skip
                        if(personTrackerVIT.isTracking) {
                            step_num = 100;
                            break;
                        }

                        ackWheels = "";
                        ackNo = "";
                        BuddySDK.USB.rotateNoPrecision(70.0f, -BuddySDK.Actuators.getNoPosition(), 0, wheelsTskCbk);

                        // compensate with head (NO)
                        BuddySDK.USB.buddySayNo(35.0f, 0.0f, noRsp);

                        step_num = 45;
                        break;

                    case 45: //wait for OK
                        if (ackWheels.toUpperCase().contains("OK") || timeout ) {

                            step_num = 47;
                        }
                        break;


                    case 47: // wait for end of mvt

                        if (ackWheels.toUpperCase().contains("FINISHED") || timeout ) {
                            if (ackNo.toUpperCase().contains("FINISHED")|| timeout )
                            {
                                Thread.sleep(500);
                                if(personTrackerVIT.isTracking) // tracking sucessful
                                    step_num=100;
                                else // tracking not succesfull
                                {
                                    // randomly make a U-Turn (1 out of 3times)
                                    if( (new Random().nextInt(4) == 3) )
                                        step_num = 50;
                                    else // make the head turn alone
                                    {
                                        noAngle = (int)Math.floor(Math.random() * (60 +60 + 1) -60);
                                        step_num=30;
                                    }

                                }
                            }
                        }
                        break;

                    case 50 : // make the body move
                        int[] buddyAngles={-90, -40, 90, 50, 180};
                        int index = new Random().nextInt(5);
                        //reset
                        ackWheels = "";
                        BuddySDK.USB.rotateBuddy(100.0f, (float)buddyAngles[index], wheelsRsp);
                        step_num = 53;
                        break;

                    case 53 : //wait for FINISHED
                        if(ackWheels.toUpperCase().contains("FINISHED") || timeout)
                        {
                            step_num = 29;
                        }
                        break;

                    case 100:// wait for tracking OK

                        if(personTrackerVIT.isTracking)
                        {
                            //Stop movement
                            BuddySDK.USB.buddyStopNoMove(noRsp);
                            BuddySDK.USB.emergencyStopMotors(wheelsRsp);

                            arrayOfStrings = context.getResources().getStringArray(R.array.search_person_found);
                            randomString = arrayOfStrings[new Random().nextInt(arrayOfStrings.length)];
                            BuddySDK.Speech.startSpeaking(randomString);
                            step_num = 900;
                        }
                        break;

                    case 105: // wait end of speech
                        if(BuddySDK.Speech.isReadyToSpeak())
                        {
                          step_num = 900;
                        }
                        break;

                    case 900://end of grafcet
                        step_num = 0;
                        go=false;
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
