package com.bfr.opencvapp.grafcet;



import static com.bfr.opencvapp.MainActivity.personTrackerVIT;
import static com.bfr.opencvapp.MainActivity.speedAngularGrafcet;
import static com.bfr.opencvapp.MainActivity.speedLinearGrafcet;

import android.content.Context;
import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.buddysdk.services.companion.TaskCallback;
import com.bfr.opencvapp.BboxCentroid;
import com.bfr.opencvapp.R;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import java.util.Random;


public class AlignBodyAndComeHereGrafcet extends bfr_Grafcet {

    Context context;

    public AlignBodyAndComeHereGrafcet(String mname, Context context) {
        super(mname);
        this.grafcet_runnable = mysequence;

        FRONT_TOF_LIM_LOWSPEED = 500;
        FRONT_TOF_LIM_HIGHSPEED = 650;
        LATERAL_TOF_LIM_LOWSPEED = 400;
        LATERAL_TOF_LIM_HIGHSPEED = 450;

        this.context = context;

    }


    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;


    // grafcet management
    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    // random Buddy vocal
    String[] arrayOfStrings;
    String randomString;

    int FRONT_TOF_LIM_LOWSPEED = 600;
    int FRONT_TOF_LIM_HIGHSPEED = 600;
    int LATERAL_TOF_LIM_LOWSPEED = 600;
    int LATERAL_TOF_LIM_HIGHSPEED = 600;

    int frontTofThres = 999;
    int lateralTofThres = 999;

    BboxCentroid target = new BboxCentroid();

    public static float yesOffset =0.0f;
    float YES_OFFSET_THRES = 5.0f;
    float yesAngle =0.0f;
    float yesSpeed = 30.0f;
    public static float noOffset=0.0f;
    float NO_OFFSET_THRES = 0.7f;
    float noAngle=0.0f;
    float noSpeed = 30.0f;
    float BASE_SPEED = 30.0f;
    float accFactor = 1.0f;

    public int MAX_UPPER_LIMIT = 200;

    String ackNo="";
    String ackYes="";
    String ackWheels="";

    private IUsbCommadRsp yesRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) {
            Log.i(name, "YES --------------- : " + success);
            ackWheels = success;
        }

        @Override
        public void onFailed(String error) {
            Log.i(name, "YES error --------------- : " + error);
            ackWheels = error;
        }
    };

    private IUsbCommadRsp noRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) {
            Log.i(name, "NO --------------- : " + success);
            ackNo = success;
        }

        @Override
        public void onFailed(String error) {
            Log.i(name, "NO error --------------- : " + error);
            ackNo = error;
        }
    };

    private IUsbCommadRsp wheelsRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) {
            Log.i(name, "Wheels --------------- : " + success);
            ackWheels = success;
        }

        @Override
        public void onFailed(String error) {
            Log.i(name, "Wheels error --------------- : " + error);
            ackWheels = error;
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

                    case 10: // get target position

                        target.getCentroid(personTrackerVIT.tracked.box.x,
                                personTrackerVIT.tracked.box.y,
                                personTrackerVIT.tracked.box.height,
                                personTrackerVIT.tracked.box.width
                        );
                        if (personTrackerVIT.tracked.objectClass==0) // if tracking a human silouhette
                            target.y = Math.max(0,(int) (personTrackerVIT.tracked.box.y+ personTrackerVIT.tracked.box.height/4));
                        else // tracking a face
                            target.y = Math.max(0,(int) (personTrackerVIT.tracked.box.y+ personTrackerVIT.tracked.box.height));

                        // compute angle for the wideAngle camera
                        // resolution of 1024x768, with a 120° aperture
                        // => 1pixel ~= 120 / sqrt(1024^2+768^2) = 0.09375
                        yesOffset = (target.y-(768/2))*0.09375f;
                        noOffset = (target.x-(1024/2))*0.09375f;

                        step_num = 15;
                        break;

                    case 15: // rotate head to align with target
                        //reset
                        ackYes = "";
                        ackNo = "";

                        yesAngle = Math.max(-13, BuddySDK.Actuators.getYesPosition()- yesOffset);
                        noAngle = BuddySDK.Actuators.getNoPosition()+noOffset;

                        BuddySDK.USB.buddySayYes(BASE_SPEED, yesAngle, yesRsp);
                        BuddySDK.USB.buddySayNo(BASE_SPEED, noAngle, noRsp);

                        step_num = 20;
                        break;

                    case 20: // wait for No OK
                        if(ackNo.toUpperCase().contains("OK") || timeout)
                            step_num = 22;
                        if(ackNo.toUpperCase().contains("TIMEOUT"))
                            step_num = 15;
                        break;

                    case 22: // wait for Yes OK
                        if(ackYes.toUpperCase().contains("OK") || timeout)
                            step_num = 25;
                        if(ackYes.toUpperCase().contains("TIMEOUT"))
                            step_num = 15;
                        break;

                    case 25 : // wait for end of mvt NO
                        if(ackNo.toUpperCase().contains("FINISHED") || timeout)
                            step_num = 27;
                        break;

                    case 27 : // wait for end of mvt YES
                        if(ackYes.toUpperCase().contains("FINISHED") || timeout)
                            step_num = 30;
                        break;

                    case 30 : // rotate body to align
                        ackWheels = "";
                        Log.d(name, "Rotating to " + BuddySDK.Actuators.getNoPosition());
                        // rotate
                        BuddySDK.USB.rotateNoPrecision(40.0f, -BuddySDK.Actuators.getNoPosition(), 0, wheelsTskCbk);

                        // compensate with head (NO)
                        ackNo = "";
                        BuddySDK.USB.buddySayNo(25.0f, 0.0f, noRsp);

                        step_num = 35;
                    break;


                    case 35: //wait for OK
                        if (ackWheels.toUpperCase().contains("OK") || timeout ) {
                            step_num = 40;
                        }
                        break;


                    case 40: // wait for end of mvt
                        if (ackWheels.toUpperCase().contains("FINISHED") || timeout ) {
                            if (ackNo.toUpperCase().contains("FINISHED")|| timeout )
                            {
                                step_num = 50;
                            }
                        }
                        break;

                    case 50 : // enable Yes tracking
                        TrackingYesGrafcet.step_num = 0;
                        TrackingYesGrafcet.go = true;

                        speedAngularGrafcet.go = true;
                        speedAngularGrafcet.step_num = 0;

                        speedLinearGrafcet.go = true;
                        speedLinearGrafcet.step_num = 0;

                        AlignBodyAndFollowGrafcet.go = true;
                        AlignBodyAndFollowGrafcet.step_num = 0;

                        FaceGrafcet.go = true;

                        step_num = 55;
                        break;

                    case 55 : //wait for Yes to track
                        Thread.sleep(500);
                        step_num = 60;
                        break;

                    case 60: //wait arrived at destination
                        if(SpeedLinearGrafcet.step_num == 250)
                        {
                            AlignBodyAndFollowGrafcet.go = false;
                            AlignBodyAndFollowGrafcet.step_num = 0;
                            step_num = 70;
                        }
                        break;
                        
                    case 70 : //stopped -> vocal acknowledge

                        arrayOfStrings = context.getResources().getStringArray(R.array.come_here_arrived);
                        randomString = arrayOfStrings[new Random().nextInt(arrayOfStrings.length)];
                        BuddySDK.Speech.startSpeaking(randomString);

                        step_num = 75;
                        break;

                    case 75://wait for end of speech
                        if(BuddySDK.Speech.isReadyToSpeak())
                            step_num = 100;
                        break;

                    case 100: //end of grafcet
                        step_num = 0;
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
