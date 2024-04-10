package com.bfr.opencvapp.grafcet;


//import static com.bfr.opencvapp.MainActivity.alignCheckbox;

import static com.bfr.opencvapp.MainActivity.personTracker;

import android.content.Context;
import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.ui.shared.FacialEvent;
import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.buddysdk.services.ModuleUSB;
import com.bfr.buddysdk.services.companion.TaskCallback;
import com.bfr.opencvapp.R;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import org.opencv.core.Point;

import java.util.Random;

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

    private SearchPersonGrafcet grafcet=this;

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

    private double timeSinceLastBlink = 0;
    private double randomBlinkInterval = 4000;

    public static int RESIZE_RATIO =20;
    public static double xCenter =0.0;
    private double xorig=0.0;
    private double deltaPixel=0.0;

    private float noAngle=0.0f;
    private float wheelsAngle=0.0f;


    private IUsbCommadRsp wheelsRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) throws RemoteException { ackWheels = success;
        Log.d(name, "Motor wheels ack="+ackWheels);
            }

        @Override
        public void onFailed(String error) throws RemoteException { ackWheels = error; }
    };

    private IUsbCommadRsp yesRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) throws RemoteException { ackYes = success; Log.d(name, "Motor YES ack="+ackYes);}
        @Override
        public void onFailed(String error) throws RemoteException { ackYes = error; }
    };

    private IUsbCommadRsp noRsp = new IUsbCommadRsp.Stub(){
        @Override
        public void onSuccess(String success) throws RemoteException { ackNo = success; Log.d(name, "Motor NO ack="+ackNo);}
        @Override
        public void onFailed(String error) throws RemoteException { ackNo = error; }
    };
    String ackYes="";
    String ackNo="";
    String ackWheels="";
    String[] arrayOfStrings;
    String randomString;

    // Define the sequence/grafcet to be executed
   /* This provides a template for a grafcet.
   The sequence is as follows:
   - check the checkbox
   - Move the No from Left to right
   - Move the no from right to left
   - If the check box is unchecked then stop
   - if not, repeat
    */
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

                    case 5://

                        arrayOfStrings = context.getResources().getStringArray(R.array.user_lost);
                        randomString = arrayOfStrings[new Random().nextInt(arrayOfStrings.length)];
                        BuddySDK.Speech.startSpeaking(randomString);

                        step_num = 7;
                        break;


                    case 7: // wait a bit to give a chance to the tracking
                        if(personTracker.trackingSuccess)
                            step_num=900;
                        if (System.currentTimeMillis()-time_in_curr_step > 1000)
                            step_num = 10;
                        break;

                    case 10 :// wait end of speech
                        if(BuddySDK.Speech.isReadyToSpeak())
                            step_num = 20;
                        break;

                    case 20: //Head at zero

                        //if tracking again then skip
                        if(personTracker.trackingSuccess)
                        {
                            step_num = 0;
                            go = false;
                            break;
                        }

                        //reset
                        ackYes="";
                        ackNo="";
                        ackWheels="";

//                        if (personTracker.tracked.box.x<500)
//                            noAngle = -30;
//                        else if(personTracker.tracked.box.x>550)
//                            noAngle = 30;
//                        else
//                            noAngle = 0.0f;

//                        if (personTracker.tracked.box.x<500)
//                            wheelsAngle = 90;
//                        else if(personTracker.tracked.box.x>550)
//                            wheelsAngle = -90;
//                        else
//                            wheelsAngle = 0.0f;

                        //Stop wheels
//                        BuddySDK.USB.emergencyStopMotors(wheelsRsp);
                        BuddySDK.USB.moveBuddy(0.3f, 0.0f, 0.1f, 0.05f,  wheelsRsp);


                        BuddySDK.USB.buddySayNo(40, noAngle, noRsp);
                        BuddySDK.USB.buddySayYes(40, 30, yesRsp);

                        step_num = 25;
                        break;

//                    case 10: // wait for OK
//                        if (ackWheels.toUpperCase().contains("OK")
//                        && ackYes.toUpperCase().contains("OK")
//                        && ackNo.toUpperCase().contains("OK") || timeout)
//                        {
//                            step_num = 18;
//                        }
//                        break;

                    case 25 : //wait for  wheels
                        if (ackWheels.toUpperCase().contains("FINISHED")
                                || BuddySDK.Actuators.getLeftWheelSpeed()<5
                                || timeout)
                            step_num = 27;
                        break;

                    case 27 : //wait for  Yes
                        if (ackYes.toUpperCase().contains("FINISHED") ||
                                ( Math.abs(BuddySDK.Actuators.getYesPosition()) < 22 && Math.abs(BuddySDK.Actuators.getYesPosition()) > 19 )
                                || timeout)
                            step_num = 28;
                        break;
                    case 28 : //wait for  Yes
                        if (ackNo.toUpperCase().contains("FINISHED") ||
                        ( Math.abs(BuddySDK.Actuators.getNoPosition()) < 2  )

                                || timeout) {
                            step_num = 29;
                        }
                        break;

                    case 29:// calc no Angle
                        //                        if(BuddySDK.Actuators.getNoPosition()>=0)
                        if (personTracker.tracked.box.x + 0.5*personTracker.tracked.box.width<500)
                            noAngle = -50.0f;
                        else
                            noAngle = 50.0f;
                        step_num = 30;
                        break;

                    case 30: // turn head No

                        ackNo = "";
                        BuddySDK.USB.buddySayNo(80.0f, noAngle, noRsp);
                        step_num=32;
                        break;

                    case 32: // wait for OK
                        if(ackNo.toUpperCase().contains("OK") || timeout)
                            step_num = 33;
                        break;
                    case 33 : //wait for end
                        if(ackNo.toUpperCase().contains("FINISHED") || timeout)
                            step_num = 40;
                        break;

//                    case 35: // check if tracking OK
//                        Thread.sleep(800);
//                        if(personTracker.trackingSuccess)
//                        {
//                            arrayOfStrings = context.getResources().getStringArray(R.array.search_person_found);
//                            randomString = arrayOfStrings[new Random().nextInt(arrayOfStrings.length)];
//                            BuddySDK.Speech.startSpeaking(randomString);
//                            step_num = 105;
//                        }
//                        else
//                        {
//                            step_num=40;
//                        }
//                        break;

                    case 40: // rotate body to align
                        ackWheels = "";
                        ackNo = "";
                        BuddySDK.USB.rotateNoPrecision(70.0f, -BuddySDK.Actuators.getNoPosition(), 0, new TaskCallback() {
                            @Override
                            public void onStarted() {
                                ackWheels="OK";
                            }

                            @Override
                            public void onSuccess(String s) {
                                ackWheels="FINISHED";
                            }

                            @Override
                            public void onCancel() {

                            }

                            @Override
                            public void onError(String s) {
                                ackWheels="ERROR";
                            }
                        });
////

//                        BuddySDK.USB.rotateBuddy(40.0f, -BuddySDK.Actuators.getNoPosition(), wheelsRsp);
                        Log.i(name, "Current No pos= " +BuddySDK.Actuators.getNoPosition() + "  ");
//                        BuddySDK.USB.rotateBuddy(70.0f, -BuddySDK.Actuators.getNoPosition(), wheelsRsp);
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
                                if(personTracker.trackingSuccess)
                                    step_num=100;
                                else
                                {
                                    noAngle = (int)Math.floor(Math.random() * (60 +60 + 1) -60);
                                    step_num=30;
                                }
                            }
                        }
                        break;

                    case 100:// wait for tracking OK

                        if(personTracker.trackingSuccess)
                        {
                            arrayOfStrings = context.getResources().getStringArray(R.array.search_person_found);
                            randomString = arrayOfStrings[new Random().nextInt(arrayOfStrings.length)];
                            BuddySDK.Speech.startSpeaking(randomString);
                            step_num = 105;
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
