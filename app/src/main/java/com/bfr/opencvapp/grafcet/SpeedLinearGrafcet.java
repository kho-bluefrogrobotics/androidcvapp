package com.bfr.opencvapp.grafcet;

import static com.bfr.opencvapp.MainActivity.personTrackerVIT;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.MainActivity;
import com.bfr.opencvapp.utils.bfr_Grafcet;


/***
 * In FollowMe mode, computes the linear speed needed to keep the robot close to the target
 * it uses the torso height of the target to estimate the distance
 * (the torso height is obtained with a pose estimation of the tracked human)
 */
public class SpeedLinearGrafcet extends bfr_Grafcet {

    public SpeedLinearGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;
        Log.i("coucou", "Lenear speed contructor "+  MainActivity.FRONT_TOF_LIM_LOWSPEED);

        FRONT_TOF_LIM_LOWSPEED = 500;
        FRONT_TOF_LIM_HIGHSPEED = 650;
        LATERAL_TOF_LIM_LOWSPEED = 400;
        LATERAL_TOF_LIM_HIGHSPEED = 450;
    }


    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    public float linearSpeed = 0.0f;
    public float COME_HERE_SPEED = 0.2f;

    public float accel = 0.5f;

    int FRONT_TOF_LIM_LOWSPEED = 600;
    int FRONT_TOF_LIM_HIGHSPEED = 600;
    int LATERAL_TOF_LIM_LOWSPEED = 500;
    int LATERAL_TOF_LIM_HIGHSPEED = 450;

    int FONT_TOF_THRES = 999;
    int LATERAL_TOF_THRES = 999;

    public boolean obstacleL = false;
    public boolean obstacleR = false;
    public boolean obstacleM = false;
    public boolean obstacleBehind = false;
    boolean bboxTooBig = false;

    public int MAX_UPPER_LIMIT = 200;
    public int FINAL_UPPER_LIMIT = 120;

    boolean LLedOn, RLedOn, MLedOn;

    // runable for grafcet
    private Runnable mysequence = new Runnable()
    {
        @Override
        public void run()
        {

            try {
                /*** Compute obstacle detection */
                if (linearSpeed >=0.3)
                {
                    FONT_TOF_THRES = FRONT_TOF_LIM_HIGHSPEED;
                    LATERAL_TOF_THRES = LATERAL_TOF_LIM_HIGHSPEED;
                }
                else
                {
                    FONT_TOF_THRES = FRONT_TOF_LIM_LOWSPEED;
                    LATERAL_TOF_THRES = LATERAL_TOF_LIM_LOWSPEED;
                }


                if ((BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() >15 && BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() < LATERAL_TOF_THRES) )
                {
                    obstacleL = true;
                }

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


                if( (BuddySDK.Sensors.TofSensors().Back().getDistance() >15 && BuddySDK.Sensors.TofSensors().Back().getDistance() < 450) )
                    obstacleBehind = true;
                else
                    obstacleBehind = false;

                //to debug Led on
                if(obstacleL && !LLedOn)
                {
                    BuddySDK.USB.updateLedColor(0, "#ff1100", new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {}
                        @Override
                        public void onFailed(String s) throws RemoteException {}
                    });
                    LLedOn = true;
                } //end if obstalce and led off
                if(!obstacleL && LLedOn)
                {
                    BuddySDK.USB.updateLedColor(0, "#61E3EB", new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {}
                        @Override
                        public void onFailed(String s) throws RemoteException {}
                    });
                    LLedOn = false;
                } //end if obstalce and led off
                /***/
                if(obstacleR && !RLedOn)
                {
                    BuddySDK.USB.updateLedColor(1, "#ff1100", new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {}
                        @Override
                        public void onFailed(String s) throws RemoteException {}
                    });
                    RLedOn = true;
                } //end if obstalce and led off
                if(!obstacleR && RLedOn)
                {
                    BuddySDK.USB.updateLedColor(1, "#61E3EB", new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {}
                        @Override
                        public void onFailed(String s) throws RemoteException {}
                    });
                    RLedOn = false;
                } //end if obstalce and led off
                /***/
                if(obstacleM && !MLedOn)
                {
                    BuddySDK.USB.updateLedColor(2, "#ff1100", new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {}
                        @Override
                        public void onFailed(String s) throws RemoteException {}
                    });
                    MLedOn = true;
                } //end if obstalce and led off
                if(!obstacleM && MLedOn)
                {
                    BuddySDK.USB.updateLedColor(2, "#61E3EB", new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {}
                        @Override
                        public void onFailed(String s) throws RemoteException {}
                    });
                    MLedOn = false;
                } //end if obstalce and led off


                // if top of the tracked bbox touches the upper limit of the image
                // (means that the target is really close)
                if(personTrackerVIT !=null)
                    if ((personTrackerVIT.tracked.box.y) <= MAX_UPPER_LIMIT)
                        bboxTooBig = true;
                    else
                        bboxTooBig = false;


                // if step changed
                if (!(step_num == previous_step)) {
                    // display current step
                    Log.i(name, "current step: " + step_num + "  speed="+linearSpeed + " accel="+accel);
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
                    if ((System.currentTimeMillis()-time_in_curr_step > 10000) && step_num >0)
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

                    case 5: // wait for tracking OK

                        if(personTrackerVIT.isTracking)
                            step_num=15;
                        break;


                    case 15: // compute linear speed to keep target close

                        // if top of the bounding box is touching the upper edge of the image
                        if (personTrackerVIT.tracked.box.y <= FINAL_UPPER_LIMIT) {
                            // go backwards
                            linearSpeed = -0.15f;
                            step_num = 190;
                            break;
                        }

                        // define linear speed <-> obstacle or size of the torso
                        if(obstacleL || obstacleR || obstacleM
                            ||     bboxTooBig)
                        {
                            Log.d(name, "OBSTACLE L/M/R= " +
                                BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontRight().getDistance() + "\n step -> 250");
                            accel = 1.0f;
                            linearSpeed = 0.0f;
                            step_num = 250;
                        }
                        else{ //box OK and no pbstacle
                            if (personTrackerVIT.torsoHeight<=350 && personTrackerVIT.torsoHeight>300)
                            {
                                Log.d(name, "Torso height = " + personTrackerVIT.torsoHeight + " -> step = 100");
                                accel = 0.5f;
                                if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                    linearSpeed = 0.15f; //recommended 0.15f
                                else // in Come Here mode
                                    linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                                step_num = 110;
                            }
                            else if (personTrackerVIT.torsoHeight<=300 && personTrackerVIT.torsoHeight>250) {
                                Log.d(name, "Torso height = " + personTrackerVIT.torsoHeight + " -> step = 110");
                                accel = 0.6f;
                                if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                    linearSpeed = 0.3f;
                                else // in Come Here mode
                                    linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                                step_num = 120;
                            }
                            else if (personTrackerVIT.torsoHeight<=250 && personTrackerVIT.torsoHeight>210)
                            {
                                Log.d(name, "Torso height = " + personTrackerVIT.torsoHeight + " -> step = 120");
                                accel = 1.0f;
                                if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                    linearSpeed = 0.4f;
                                else // in Come Here mode
                                    linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                                step_num = 130;
                            }
                            else if (personTrackerVIT.torsoHeight<=210 )
                            {
                                Log.d(name, "Torso height = " + personTrackerVIT.torsoHeight + " -> step = 150");
                                accel = 1.0f;
                                if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                    linearSpeed = 0.56f;
                                else // in Come Here mode
                                    linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                                step_num = 140;
                            }
                            else
                            {
                                Log.d(name, "Torso height = " + personTrackerVIT.torsoHeight + " -> step = 140");
                                accel = 0.3f;
                                linearSpeed = 0.0f;
                                step_num = 100;
                            }

                        } //end if no obstacle

                        break;

                    case 100://target [350;inf]

                        // target is getting far
                        if (personTrackerVIT.torsoHeight<350 )
                        {
                            if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                linearSpeed = 0.15f;
                            else // in Come Here mode
                                linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                            accel = 1.0f;
                            step_num = 110;
                        }

                        // interrupt if obstacle
                        if(obstacleL || obstacleR || obstacleM
                                ||     bboxTooBig)
                        {
                            Log.d(name, "OBSTACLE L/M/R= " +
                                BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontRight().getDistance() + "\n step -> 250");
                            accel = 1.0f;
                            linearSpeed = 0.0f;
                            step_num = 250;
                        }
                        break;

                    case 110 : // target [300;350]

                        //target is getting closer
                        if(personTrackerVIT.torsoHeight>350)
                        {
                            linearSpeed = 0.0f;
                            accel = 0.3f;
                            step_num=100;
                        }

                        //target is getting far
                        if(personTrackerVIT.torsoHeight<300)
                        {
                            if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                linearSpeed = 0.3f;
                            else // in Come Here mode
                                linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                            accel = 1.0f;
                            step_num=120;
                        }

                        // interrupt if obstacle
                        if(obstacleL || obstacleR || obstacleM
                                ||     bboxTooBig)
                        {
                            Log.d(name, "OBSTACLE L/M/R= " +
                                BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontRight().getDistance() + "\n step -> 250");
                            accel = 0.3f;
                            linearSpeed = 0.0f;
                            step_num = 250;
                        }
                        break;

                    case 120:// target [250;300]

                        //target is getting closer
                        if(personTrackerVIT.torsoHeight>300)
                        {
                            if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                linearSpeed = 0.15f;
                            else // in Come Here mode
                                linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                            accel = 0.3f;
                            step_num=110;
                        }
                        //target is getting far
                        if(personTrackerVIT.torsoHeight<250)
                        {
                            if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                linearSpeed = 0.4f;
                            else // in Come Here mode
                                linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                            accel = 1.0f;
                            step_num=130;
                        }

                        // interrupt if obstacle
                        if(obstacleL || obstacleR || obstacleM
                                ||     bboxTooBig)
                        {
                            Log.d(name, "OBSTACLE L/M/R= " +
                                BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontRight().getDistance() + "\n step -> 250");
                            accel = 1.0f;
                            linearSpeed = 0.0f;
                            step_num = 250;
                        }
                        break;

                    case 130://target [210;250]
                        //target is getting closer
                        if(personTrackerVIT.torsoHeight>250)
                        {
                            if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                linearSpeed = 0.3f;
                            else // in Come Here mode
                                linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                            accel = 0.3f;
                            step_num=120;
                        }
                        //target is getting far
                        if(personTrackerVIT.torsoHeight<210)
                        {
                            if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                linearSpeed = 0.56f;
                            else // in Come Here mode
                                linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                            accel = 1.0f;
                            step_num=140;
                        }

                        // interrupt if obstacle
                        if(obstacleL || obstacleR || obstacleM)
                        {
                            Log.d(name, "OBSTACLE L/M/R= " +
                                BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontRight().getDistance() + "\n step -> 250");
                            accel = 1.0f;
                            linearSpeed = 0.0f;
                            step_num = 250;
                        }
                        break;
                    case 140:// target [0;210]

                        //target is getting closer
                        if(personTrackerVIT.torsoHeight>210)
                        {
                            if (MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                                linearSpeed = 0.4f;
                            else // in Come Here mode
                                linearSpeed = COME_HERE_SPEED; //recommended 0.15f
                            accel = 0.3f;
                            step_num=130;
                        }

                        // interrupt if obstacle
                        if(obstacleL || obstacleR || obstacleM)
                        {
                            Log.d(name, "OBSTACLE L/M/R= " +
                                BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontRight().getDistance() + "\n step -> 250");
                            accel = 1.0f;
                            linearSpeed = 0.0f;
                            step_num = 250;
                        }
                        break;



                    case 190:// going back
                        if(personTrackerVIT.tracked.box.y> FINAL_UPPER_LIMIT)
                            step_num = 15;

                        if(obstacleBehind)
                        {
                            linearSpeed = 0.0f;
                            // request to FaceGrafcet to play a facial event to signal an obstacle behind
                            FaceGrafcet.obstacleFaceEvtReq = true;
                            step_num = 194;
                        }

                        break;


                    case 194 : // play facial elvent to signal obstacle behind
                        // wait for handshake
                        if (!FaceGrafcet.obstacleFaceEvtReq)
                            step_num = 195;
                        break;

                    case 195:// stop going back because of obstacle behind
                        if(personTrackerVIT.tracked.box.y> FINAL_UPPER_LIMIT) // if person is leaving
                        {
                            step_num = 15;
                        }
                        else // person still close
                        {
                            if(!obstacleBehind)
                            {
                                linearSpeed = -0.15f;
                                step_num = 190;
                            }
                        }

                        break;

                    case 250:// Obstacle-->Stopped

                        // if tracked bbox touches the upper limit of the image
                        if (personTrackerVIT.tracked.box.y <= FINAL_UPPER_LIMIT) {
                            // go backwards
                            linearSpeed = -0.15f;
                            step_num = 190;
                            break;
                        }

                        // if no more obstacle
                        if(!obstacleL && !obstacleR && !obstacleM && !bboxTooBig)
                            step_num = 15;
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
