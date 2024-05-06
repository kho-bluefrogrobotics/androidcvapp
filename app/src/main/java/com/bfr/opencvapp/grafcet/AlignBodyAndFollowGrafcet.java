package com.bfr.opencvapp.grafcet;

import static com.bfr.opencvapp.MainActivity.personTrackerVIT;
import static com.bfr.opencvapp.MainActivity.speedAngularGrafcet;
import static com.bfr.opencvapp.MainActivity.speedLinearGrafcet;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.BboxCentroid;
import com.bfr.opencvapp.utils.bfr_Grafcet;


/***
 * This grafcet manages the body movements when in Follow-Me mode
 * it continuously makes the robot go at a specified linear and angular speed
 * the linear speed and angular speed is computed in two separate other grafcets
 */

public class AlignBodyAndFollowGrafcet extends bfr_Grafcet {

    public AlignBodyAndFollowGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

        FRONT_TOF_LIM_LOWSPEED = 500;
        FRONT_TOF_LIM_HIGHSPEED = 650;
        LATERAL_TOF_LIM_LOWSPEED = 400;
        LATERAL_TOF_LIM_HIGHSPEED = 450;

    }

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;


    // grafcet management
    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    float rotspeed=1.0f;
    float linearspeed = 0.0f;
    float accel =  0.5f;

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

    String ackNo="";
    String ackYes="";
    String ackWheels="";

    private IUsbCommadRsp yesRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) {
            Log.i(name, "YES --------------- : " + success);
            ackYes = success;
        }

        @Override
        public void onFailed(String error) {
            Log.i(name, "YES error --------------- : " + error);
            ackYes = error;
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
                            step_num = 3;
                        }
                        break;


                    case 3: // get target position

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

                        step_num = 5;
                        break;

                    case 5: // rotate head and body to align with target
                        //reset
                        ackYes = "";
                        ackNo = "";
                        ackWheels = "";

                        yesAngle = Math.max(-13, BuddySDK.Actuators.getYesPosition()- yesOffset);
                        noAngle = BuddySDK.Actuators.getNoPosition()+noOffset;

                        BuddySDK.USB.buddySayYes(BASE_SPEED, yesAngle, yesRsp);
                        BuddySDK.USB.buddySayNo(BASE_SPEED, 0, noRsp);
                        BuddySDK.USB.rotateBuddy(40.0f, -noAngle, wheelsRsp);

                        step_num = 7;
                        break;

                    case 7: // wait for end of NO mvt
                        if(ackNo.toUpperCase().contains("TIMEOUT"))
                            step_num = 5;
                        if(ackNo.toUpperCase().contains("FINISHED") || timeout)
                            step_num = 10;
                        break;

                    case 10: // wait for wheels en d of mvt
                        if(ackWheels.toUpperCase().contains("FINISHED") || timeout)
                            step_num = 15;
                        break;

                    case 15: // Move body

                        // Obstacle distance for stopping
                        if (speedLinearGrafcet.linearSpeed >=0.3)
                        {
                            frontTofThres = FRONT_TOF_LIM_HIGHSPEED;
                            lateralTofThres = LATERAL_TOF_LIM_HIGHSPEED;
                        }
                        else
                        {
                            frontTofThres = FRONT_TOF_LIM_LOWSPEED;
                            lateralTofThres = LATERAL_TOF_LIM_LOWSPEED;
                        }

                        // if obstacle detected and going moving forward
                        if(  (
                                (BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() >15 && BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() < lateralTofThres)
                                        || (BuddySDK.Sensors.TofSensors().FrontRight().getDistance() >15 && BuddySDK.Sensors.TofSensors().FrontRight().getDistance() < lateralTofThres)
                                        || (BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() >15 && BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() < frontTofThres)
                        ) && speedLinearGrafcet.linearSpeed >0 )
                        {
                            //stop
                            BuddySDK.USB.moveBuddy(5.0f, 0.0f, -0.01f, 90.0f, new IUsbCommadRsp.Stub() {
                                @Override
                                public void onSuccess(String s) throws RemoteException {}
                                @Override
                                public void onFailed(String s) throws RemoteException {}
                            });
                            step_num = 90;
                        }
                        else { //else, No obstacle

                            // going forward at the speed computed in other grafcets
                            linearspeed = speedLinearGrafcet.linearSpeed;
                            rotspeed = speedAngularGrafcet.angularSpeed;
                            accel = Math.max(speedLinearGrafcet.accel, speedAngularGrafcet.accel);

                            BuddySDK.USB.setBuddySpeed(linearspeed, rotspeed, accel, new IUsbCommadRsp.Stub() {
                                @Override
                                public void onSuccess(String s) throws RemoteException {
                                    ackWheels = s;
                                }

                                @Override
                                public void onFailed(String s) throws RemoteException {
                                    ackWheels = s;
                                }
                            });

                            // stay in this step in an infinite loop (unless obstacle detected)
                        }

                        break;


                    case 90 : // obstacle - emergency stop

                        Log.d(name, "Stopping" );

                        step_num = 95;
                        break;

                    case 95: //wait for end obstacle or going backwards
                        if( (!speedLinearGrafcet.obstacleL && !speedLinearGrafcet.obstacleR && !speedLinearGrafcet.obstacleM)
                                || speedLinearGrafcet.linearSpeed <0)
                            step_num = 0;

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
