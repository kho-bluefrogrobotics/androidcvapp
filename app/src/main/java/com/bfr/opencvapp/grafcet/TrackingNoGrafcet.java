package com.bfr.opencvapp.grafcet;


import static com.bfr.opencvapp.MainActivity.personTracker;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.PersonTracker;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import org.opencv.core.Point;
import org.opencv.core.Rect;


public class TrackingNoGrafcet extends bfr_Grafcet{

    public TrackingNoGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }



    private TrackingNoGrafcet grafcet=this;

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
    private boolean bypass = false;

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

    float noOffset=0.0f;
    float noAngle=0.0f;

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
            } // end if step = same


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
                    BuddySDK.USB.enableNoMove(true, new IUsbCommadRsp.Stub() {
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
                    if( !BuddySDK.Actuators.getNoStatus().toUpperCase().contains("DISABLE"))
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
                    targetY = (int) target.y;
                    Log.d(name, "Target at " + targetX + "," + targetY);
                    // compute angle
                    noOffset = (targetX-(1024/2))*0.09375f;
                    Log.d(name, "Rotation " + noAngle);

                    if(Math.abs(noOffset)>5.0f)
                        step_num = 20;
                    break;

                case 20: // move head

                    //reset
                    motorAck = "";
                    noAngle = BuddySDK.Actuators.getNoPosition()+noOffset;
                    Log.d(name, "rotating to " + noAngle);

                    BuddySDK.USB.buddySayNo(Math.abs(noOffset*3), noAngle, new IUsbCommadRsp.Stub() {
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
                    if (motorAck.contains("OK"))
                    {
                        step_num = 28;
                    }
                    break;

                case 28 : // wait for end of mvt
                    if(motorAck.contains("FINISHED"))
                    {
                        step_num = 30000;
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
                case 30: // // move head

                    //reset
                    motorAck = "";
                    BuddySDK.USB.buddySayNo(30.0f, -45f, new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {
                            motorAck = s;
                        }

                        @Override
                        public void onFailed(String s) throws RemoteException {

                        }
                    });
                    step_num = 35;
                    break;

                case 35: // wait for OK
                    if (motorAck.contains("OK"))
                    {
                        step_num = 38;
                    }
                    break;

                case 38 : // wait for end of mvt
                    if(motorAck.contains("FINISHED"))
                    {
                        step_num = 10;
                    }
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
