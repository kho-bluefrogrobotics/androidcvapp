package com.bfr.opencvapp.grafcet;


//import static com.bfr.opencvapp.MainActivity.alignCheckbox;

import static com.bfr.opencvapp.MainActivity.personTracker;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.buddysdk.services.companion.TaskCallback;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import org.opencv.core.Point;

public class FaceGrafcet extends bfr_Grafcet {

    public FaceGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }


    private FaceGrafcet grafcet=this;

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
                        if (true) {
                            // go to next step
                            step_num = 5;
                        }
                        break;

                    case 5: //get position

                        // scaling the tracked box between 0;1
                        // scaling a value v =[min1;max1] to a range [min2; max2]
                        // new_value= ( v - min1)  * [ ( max2-min2)/(max1-min1) ] + min2

                        // empirically, we observe the tracked box horizontal position of its center is between 200;830
                        float centerPosX = (float)(personTracker.tracked.box.x  + personTracker.tracked.box.width/2);
                        // changing the range
                        float scaleX = (1.0f-0.0f) / (830 - 200.0f);
                        // !!!the tracking is mirrored tracking.x = 0 -> position value must be 1
                        float xpos = (( 200.0f- centerPosX)*scaleX + 1.0f) ;
                        // the final value for the eyes mus be between 0;1300
                        xpos = xpos *1300;

                        // same thing for Y
                        float centerPosY = (float)(personTracker.tracked.box.y ); //pointing to the top of the bbox
                        float scaleY = (0.7f-0.3f) / (350 - 150.0f);
                        // Y is not inverted
                        float ypos = (( centerPosY - 150.0f)*scaleY + 0.3f) ;
                        ypos = ypos * 900;

                        BuddySDK.UI.lookAtXY(xpos,
                               ypos , true);
//                        BuddySDK.UI.lookAtXY(1200, 1000, true);
                        Log.i(name, "coords " +
                                personTracker.tracked.box.x + "," + personTracker.tracked.box.y + " -- " +
                               xpos + ","+ ypos);
                        step_num = 5;
                        break;

                    case 10:
//                        Thread.sleep(100);
                        step_num = 5;
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
