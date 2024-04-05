package com.bfr.opencvapp.grafcet;


//import static com.bfr.opencvapp.MainActivity.alignCheckbox;

import static com.bfr.opencvapp.MainActivity.personTracker;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import org.opencv.core.Point;

public class SpeedLinearGrafcet extends bfr_Grafcet {

    public SpeedLinearGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }


    private SpeedLinearGrafcet grafcet=this;

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

    public static int RESIZE_RATIO =20;
    public static double xCenter =0.0;
    private double xorig=0.0;
    private double deltaPixel=0.0;

    private float angleToRotate=0.0f;


    private IUsbCommadRsp iUsbCommadRsp = new IUsbCommadRsp.Stub(){

        @Override
        public void onSuccess(String success) throws RemoteException {
            Log.i("GRAFCET NO", "success --------------- : " + success);
        }

        @Override
        public void onFailed(String error) throws RemoteException {
            Log.i("GRAFCET NO", "error --------------- : " + error);

        }
    };

    String ackYes="";
    String ackNo="";
    String ackWheels="";
    public float linearSpeed = 0.0f;

    final float BASE_SPEED=0.7f;

    boolean obstacleL = false;
    boolean obstacleR = false;
    boolean obstacleM = false;
    boolean bboxTooBig = false;

    boolean LLedOn, RLedOn, MLedOn;

    public float initialTorsoHeight =0.0f;

    // Define the sequence/grafcet to be executed
   /* This provides a template for a grafcet.
   The sequence is as follows:
   - check the checkbox
   - Move the No from Left to right
   - Move the no from right to left
   - If the check box is unchecked then stop<
   - if not, repeat
    */
    // runable for grafcet
    private Runnable mysequence = new Runnable()
    {
        @Override
        public void run()
        {

            try {


                /*** Compute obstacle detection */
                if ((BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() >5 && BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() < 400) )
                    obstacleL = true;
                else
                    obstacleL = false;


                if( (BuddySDK.Sensors.TofSensors().FrontRight().getDistance() >5 && BuddySDK.Sensors.TofSensors().FrontRight().getDistance() < 400) )
                    obstacleR = true;
                else
                    obstacleR = false;

                if( (BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() >5 && BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() < 500) )
                    obstacleM = true;
                else
                    obstacleM = false;

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



               if(personTracker.tracked.objectClass==0) // if tracking a human silouhette
               {
                   if ((personTracker.tracked.box.height*personTracker.tracked.box.width) >40000)
                       bboxTooBig =true;
                   else
                       bboxTooBig = false;
               }
               else // tracking a face
               {
                   if ((personTracker.tracked.box.height*personTracker.tracked.box.width) >21000)
                       bboxTooBig =true;
                   else
                       bboxTooBig = false;
               }


//                Log.e(name, "STOP!!!!!! area=" +  (personTracker.tracked.box.height*personTracker.tracked.box.width)
//                        +"   sensors= " + BuddySDK.Sensors.USSensors().LeftUS().getDistance() + " , " + BuddySDK.Sensors.USSensors().RightUS().getDistance()
//                        +"   ampl= " + BuddySDK.Sensors.USSensors().LeftUS().getAmplitude()+ " , " + BuddySDK.Sensors.USSensors().RightUS().getAmplitude()
//
//                );


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

                        if(personTracker.trackingSuccess)
                            step_num=10;

                    case 7: // record initial torso height
                        initialTorsoHeight = Math.max(200, personTracker.getTorsoHeight());
                        step_num =10;
                        break;

                    case 10: // check target offaxis alignment

                        step_num = 15;
                        break;


                    case 15: // rotate body to align

                        ackWheels = "";

//                        linearSpeed =Math.max(0.0f, (float) (-0.004 * personTracker.torsoHeight  +1) );


                        if(obstacleL || obstacleR || obstacleM)
//                            ||     bboxTooBig)
                        {   Log.i(name, "OBSTACLE L/M/R= " +
                                BuddySDK.Sensors.TofSensors().FrontLeft().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance() + ","
                                + BuddySDK.Sensors.TofSensors().FrontRight().getDistance() + "\n step -> 250");
                            linearSpeed = 0.0f;
                            step_num = 250;
                        }
                        else{ //box OK and no pbstacle
                            if (personTracker.torsoHeight<=350 && personTracker.torsoHeight>300)
                            {
                                Log.i(name, "Torso height = " + personTracker.torsoHeight + " -> step = 100");
                                linearSpeed = 0.15f;
                                step_num = 100;
                            }
                            else if (personTracker.torsoHeight<=300 && personTracker.torsoHeight>250) {
                                Log.i(name, "Torso height = " + personTracker.torsoHeight + " -> step = 110");
                                linearSpeed = 0.3f;
                                step_num = 110;
                            }
                            else if (personTracker.torsoHeight<=250 && personTracker.torsoHeight>160)
                            {
                                Log.i(name, "Torso height = " + personTracker.torsoHeight + " -> step = 120");
                                linearSpeed = 0.4f;
                                step_num = 120;
                            }
                            else if (personTracker.torsoHeight<=160 )
                            {
                                Log.i(name, "Torso height = " + personTracker.torsoHeight + " -> step = 130");
                                linearSpeed = 0.56f;
                                step_num = 130;
                            }
                            else
                            {
                                Log.i(name, "Torso height = " + personTracker.torsoHeight + " -> step = 140");
                                linearSpeed = 0.0f;
                                step_num = 140;
                            }



//                            if (personTracker.torsoHeight> 0.85*initialTorsoHeight) {
//                                Log.i("coucou", "Torso>Initial: " +personTracker.torsoHeight  + "/"+ initialTorsoHeight);
//                                linearSpeed = 0.0f;
//                            }
//                            else if (personTracker.torsoHeight<=0.85*initialTorsoHeight && personTracker.torsoHeight>0.75*initialTorsoHeight) {
//                                Log.i("coucou", "Torso>0.85*Initial: " +personTracker.torsoHeight  + "/"+ initialTorsoHeight);
//                                linearSpeed = 0.15f;
//                            }
//                            else if (personTracker.torsoHeight<=0.75*initialTorsoHeight && personTracker.torsoHeight>0.6*initialTorsoHeight) {
//                                Log.i("coucou", "Torso>0.75*Initial: " +personTracker.torsoHeight  + "/"+ initialTorsoHeight);
//                                linearSpeed = 0.3f;
//                            }
//                            else if (personTracker.torsoHeight<=0.6*initialTorsoHeight && personTracker.torsoHeight>0.4*initialTorsoHeight) {
//                                Log.i("coucou", "Torso>0.6*Initial: " +personTracker.torsoHeight  + "/"+ initialTorsoHeight);
//                                linearSpeed = 0.45f;
//                            }
//                            else if (personTracker.torsoHeight<=0.4*initialTorsoHeight ) {
//                                Log.i("coucou", "Torso<0.6*Initial: " +personTracker.torsoHeight  + "/"+ initialTorsoHeight);
//                                linearSpeed = 0.56f;
//                            }
//                            else {
//                                Log.i("coucou", "Torso - Initial ELSE: " +personTracker.torsoHeight  + "/"+ initialTorsoHeight);
//                                linearSpeed = 0.0f;
//                            }

                        }

                        break;

                    case 100 : //
                        if(personTracker.torsoHeight>350 || personTracker.torsoHeight<300)
                            step_num=15;
                        break;

                    case 110://
                        if (personTracker.torsoHeight>300 || personTracker.torsoHeight<250)
                            step_num=15;
                        break;

                    case 120://
                        if (personTracker.torsoHeight>250 || personTracker.torsoHeight<160)
                            step_num=15;
                        break;
                    case 130://
                        if (personTracker.torsoHeight>160 )
                            step_num=15;
                        break;
                    case 140://

                        if (personTracker.torsoHeight<350 )
                            step_num=15;
                        break;

                    case 250://
//                        linearSpeed = 0.0f;
                        if(!obstacleL && !obstacleR && !obstacleM)
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
