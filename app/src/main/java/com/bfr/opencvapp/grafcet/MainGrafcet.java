package com.bfr.opencvapp.grafcet;


//import static com.bfr.opencvapp.MainActivity.alignCheckbox;

import static com.bfr.opencvapp.MainActivity.alignBodyFollowGrafcet;
import static com.bfr.opencvapp.MainActivity.initGrafcet;
import static com.bfr.opencvapp.MainActivity.personTracker;
import static com.bfr.opencvapp.MainActivity.searchPersonGrafcet;
import static com.bfr.opencvapp.MainActivity.speedAngularGrafcet;
import static com.bfr.opencvapp.MainActivity.speedLinearGrafcet;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.ui.shared.FacialEvent;
import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import org.opencv.core.Point;
import org.opencv.face.Face;

public class MainGrafcet extends bfr_Grafcet {

    public MainGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }


    private MainGrafcet grafcet=this;

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

                    case 5: //start Init

                        initGrafcet.start();
                        initGrafcet.go = true;
                        initGrafcet.step_num=0;
                        step_num = 7;
                        break;

                    case 7: // wait for end of init
//
                        if(!initGrafcet.go)
                        {
                            initGrafcet.stop();
                            initGrafcet.go = false;
                            step_num = 8;
                        }
                        break;




                    case 8://starting body alignment
                        alignBodyFollowGrafcet.start(100);
                        alignBodyFollowGrafcet.go = true;
                        alignBodyFollowGrafcet.step_num = 0;


                        speedAngularGrafcet.start(20);
                        speedAngularGrafcet.go = true;
                        speedAngularGrafcet.step_num = 0;

                        speedLinearGrafcet.start(20);
                        speedLinearGrafcet.go = true;
                        speedLinearGrafcet.step_num = 0;

                        TrackingNoGrafcet.go = true;
                        TrackingYesGrafcet.go = true;

                        step_num = 9;
                        break;


                    case 9 : // wait for end of grafcet
                        if(!alignBodyFollowGrafcet.go)
                            step_num = 7;

                        if (personTracker.frameCount==0)
                        {
                            alignBodyFollowGrafcet.go = false;
                            alignBodyFollowGrafcet.step_num = 0;

                            searchPersonGrafcet.start();
                            searchPersonGrafcet.go=true;
                            searchPersonGrafcet.step_num=0;

                            TrackingNoGrafcet.go = false;
                            TrackingYesGrafcet.go = false;
                            TrackingNoGrafcet.step_num=0;
                            TrackingYesGrafcet.step_num=0;

                            step_num = 70;
                        }

                        break;


                    case 10: //start Tracking
                        TrackingNoGrafcet.go = true;
                        TrackingYesGrafcet.go = true;
                        AlignGrafcet.go = true;
                        FaceGrafcet.go = true;

                        step_num = 15;
                        break;

                    case 15:
                        if(!TrackingNoGrafcet.go)
                            step_num = 20;

                        if (personTracker.frameCount==0)
                        {
                            TrackingNoGrafcet.go = false;
                            TrackingYesGrafcet.go = false;
                            AlignGrafcet.go = false;
                            TrackingNoGrafcet.step_num=0;
                            TrackingYesGrafcet.step_num=0;
                            AlignGrafcet.step_num=0;

                            searchPersonGrafcet.start();
                            searchPersonGrafcet.go=true;
                            searchPersonGrafcet.step_num=0;
                            step_num = 70;
                        }
                        break;


                    case 70: //wait for end of searchperson
                        if(!searchPersonGrafcet.go)
                        {
                            searchPersonGrafcet.stop();
                            searchPersonGrafcet.go = false;
//                            step_num = 10;
                            step_num = 8;
                        }

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
