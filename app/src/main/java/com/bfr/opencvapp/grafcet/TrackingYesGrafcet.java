package com.bfr.opencvapp.grafcet;


import static com.bfr.opencvapp.MainActivity.personTracker;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import org.opencv.core.Point;
import org.opencv.core.Rect;


public class TrackingYesGrafcet extends bfr_Grafcet{

    public TrackingYesGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }



    private TrackingYesGrafcet grafcet=this;

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    //A CHANGER POUR LE YES
    final static int INTERVAL_MIN = 250;
    final static int INTERVAL_MAX = 350;
    private int mIntervalleHist = INTERVAL_MIN;
    private float speed = 10F;

    private int previous_step = -1;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    public static Rect tracked=new Rect();

    // Last position where a face was detected
    public int lastValidPos = 0;
    // Lost faces
    public boolean lostFaces = false;

    //motor response
    private boolean isMoving = false;
    private float mMiddleRect;


    Point target;
    int targetX, targetY;

    String motorAck = "";

    public static float yesOffset =0.0f;
    float previousOffset=0.0f;
    float noAngle=0.0f;
    float noSpeed = 30.0f;
    float BASE_SPEED = 30.0f;
    float accFactor = 1.0f;

    private IUsbCommadRsp iUsbCommadRsp = new IUsbCommadRsp.Stub(){

        @Override
        public void onSuccess(String success) throws RemoteException {
            Log.i("GRAFCET YES", "success --------------- : " + success);
        }

        @Override
        public void onFailed(String error) throws RemoteException {
            Log.i("GRAFCET YES", "error --------------- : " + error);

        }
    };

    // Define the sequence/grafcet to be executed
   /* This provides a template for a grafcet.
   The sequence is as follows:
   - check the checkbox
   - Move the yes from top to bottom
   - Move the no from bottom to top
   - If the check box is unchecked then stop
   - if not, repeat
    */
    // runable for grafcet
    private Runnable mysequence = new Runnable()
    {
        @Override
        public void run()
        {
            mMiddleRect = tracked.y /*+ (tracked.height/2)*/;
//            Log.i("GRAFCET YES", "current Y: " + tracked.y + "  ");
            //Log.i("GRAFCET", "current step: " + step_num + "  ");

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
                   /* try {
                        grafcet.mBuddySDK.getUsbInterface().buddyStopYesMove(iUsbCommadRsp);
                        grafcet.mBuddySDK.getUsbInterface().enableYesMove(0, iUsbCommadRsp);
                    } catch (RemoteException e) {
                        e.printStackTrace();
                    }*/
                    //wait until check box
                    if (go) {

                        // go to next step
                        step_num = 5;
                    }
                    break;

                case 5: // enable wheels
                    BuddySDK.USB.enableYesMove(1, new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {

                        }

                        @Override
                        public void onFailed(String s) throws RemoteException {

                        }
                    });
                    step_num = 6;
                    break;

                case 6 : // Wait for enable
                    if( !BuddySDK.Actuators.getYesStatus().toUpperCase().contains("DISABLE"))
                    {
                        step_num = 10;
                    }
                    break;

                case 10: // get target position

                    target = getCentroid(personTracker.tracked.box.x,
                            personTracker.tracked.box.y,
                            personTracker.tracked.box.height,
                            personTracker.tracked.box.width
                            );
                    targetX = (int) target.x;
                    targetY = Math.max(0,(int) (target.y+ personTracker.tracked.box.height/4));
//                    Log.d(name, "Target at " + targetX + "," + targetY);
                    // compute angle
                    yesOffset = (targetY-(768/2))*0.09375f;
                    Log.d(name, "TargetY = "+ targetY + " Offset= " + yesOffset);

                    if(Math.abs(yesOffset)>7.0f)
                        step_num = 20;
                    break;

                case 20: // move head

                    //reset
                    motorAck = "";
                    previousOffset = yesOffset;

                    noAngle = BuddySDK.Actuators.getYesPosition()- yesOffset;

//                    if (noOffset>0)
//                        noAngle = -150.0f;
//                    else
//                        noAngle = 150.0f;

                    Log.d(name, "rotating to " + noAngle + " (offset=" + yesOffset +") with Yes position = " + BuddySDK.Actuators.getYesPosition() + " at " + noSpeed);
                    // speed
//                    noSpeed = Math.max(noOffset*1.3f, 30.0f);

//                    BuddySDK.USB.buddySayNo(Math.abs(noOffset*3), noAngle, new IUsbCommadRsp.Stub() {
                    BuddySDK.USB.buddySayYes(BASE_SPEED, noAngle, new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {
                            motorAck = s;
                        }

                        @Override
                        public void onFailed(String s) throws RemoteException {

                        }
                    });
                    step_num = 25;
                    break;

                case 25: // wait for OK
                    if (motorAck.toUpperCase().contains("OK") || timeout)
                    {
                        step_num = 27;
                    }
                    break;






                case 27 : // waiting for end of mvt
                    if (motorAck.toUpperCase().contains("FINISHED") || timeout)
                    {
                        step_num = 10;
                    }
                    break;













                case 28 : // wait for target in range
                    target = getCentroid(personTracker.tracked.box.x,
                            personTracker.tracked.box.y,
                            personTracker.tracked.box.height,
                            personTracker.tracked.box.width
                    );
                    targetX = (int) target.x;
                    targetY = (int) target.y;
//                    Log.d(name, "Target at " + targetX + "," + targetY);
                    // compute angle
                    yesOffset = (targetY-(768/2))*0.09375f;

                    // if target in range
                    if (Math.abs(yesOffset)<5)
                    {
                        Log.d(name, "offset = " + yesOffset + " -> STOP");
                        BuddySDK.USB.buddyStopYesMove(new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {

                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {

                            }
                        });
                        step_num = 10;
                    }
                    else
                    {
                        if (Math.abs(yesOffset)-Math.abs(previousOffset)>1)
                        {
                            Log.d(name, "offset is moving: "
                                    + Math.abs(yesOffset) + "-"+ Math.abs(previousOffset)
                                    +"=" +(Math.abs(yesOffset)-Math.abs(previousOffset)));
                            accFactor = Math.abs(yesOffset)-Math.abs(previousOffset);
                            step_num = 30;
                        }
                    }

                    break;



                case 30000 : // wait
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    step_num = 10;
                    break;

                default :
                    // go to next step
                    step_num = 0;
                    break;
            } //End switch

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
