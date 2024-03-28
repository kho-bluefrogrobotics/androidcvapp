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

    boolean obstacle = false;

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
                if (
                        (BuddySDK.Sensors.USSensors().LeftUS().getDistance() >5 && BuddySDK.Sensors.USSensors().LeftUS().getDistance() < 350)
                        ||  (BuddySDK.Sensors.USSensors().RightUS().getDistance() >5 && BuddySDK.Sensors.USSensors().RightUS().getDistance() < 350)
                )
                    obstacle = true;
                else
                    obstacle = false;

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
                            step_num = 10;
                        }
                        break;


                    case 10: // check target offaxis alignment


                        step_num = 15;
                        break;


                    case 15: // rotate body to align


                        ackWheels = "";


                        if (personTracker.torsoHeight<=200 && personTracker.torsoHeight>100)
                            linearSpeed = 0.2f;
                        else if (personTracker.torsoHeight<=100)
                            linearSpeed = 0.3f;
                        else
                            linearSpeed = 0.0f;

                        if(obstacle ||
                            (personTracker.tracked.box.height*personTracker.tracked.box.width) >21000)
                        {
                            linearSpeed = 0.0f;
                            Log.e(name, "STOP!!!!!! area=" +  (personTracker.tracked.box.height*personTracker.tracked.box.width)
                            +"   sensors= " + BuddySDK.Sensors.USSensors().LeftUS().getDistance() + " , " + BuddySDK.Sensors.USSensors().RightUS().getDistance());

                        }


//                        step_num = 17;

                        break;



                    case 30: // get closer to target
                        if (personTracker.tracked.box.height<200)
                        {
                            step_num = 35;
                            Log.i("coucou", "target size=" +personTracker.tracked.box.height );
                        }
                        else
                        {
                            Log.i("coucou", "target size=" +personTracker.tracked.box.height );
                            step_num = 10;
                        }

                        break;

                    case 35: // go forward
                        ackWheels = "";
                        BuddySDK.USB.moveBuddy(0.1f, 1.0f, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {
                                ackWheels = s;
                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {
                                ackWheels = s;
                            }
                        });
                        step_num = 37;
                        break;


                    case 37: //wait for OK
                        if(ackWheels.toUpperCase().contains("OK") ||
                                ackWheels.toUpperCase().contains("CANCELED") || timeout )
                            step_num = 38;
                        break;

                    case 38://wait for end
                        if(ackWheels.toUpperCase().contains("FINISHED") ||
                                ackWheels.toUpperCase().contains("CANCELED") || timeout )
                        {
                            Log.i("coucou", "going to step 10 because ack=" +ackWheels + " and timeout=" + timeout );
                            BuddySDK.USB.setBuddySpeed(0.0f, 0.0f, 0, new IUsbCommadRsp.Stub() {
                                @Override
                                public void onSuccess(String s) throws RemoteException { }

                                @Override
                                public void onFailed(String s) throws RemoteException {}
                            });
                            step_num = 10;
                        }

                        int distThres = 500;
                        int tofMiddle =BuddySDK.Sensors.TofSensors().FrontMiddle().getDistance();
                        int tofLeft =BuddySDK.Sensors.TofSensors().FrontLeft().getDistance();
                        int tofRight =BuddySDK.Sensors.TofSensors().FrontRight().getDistance();

                        boolean obstacle = false;

                        if( (tofLeft >0 && tofLeft < distThres)
                        || (tofMiddle >0 && tofMiddle < distThres)
                        || (tofRight >0 && tofRight < distThres))
                            obstacle = true;



                        Log.i("coucou", "***************Sensors value=" + tofMiddle + ", " + tofLeft + ", "  + tofRight);
                        if(obstacle)
                        {
                            Log.i("coucou", "STOP!!!  sensors=" + tofMiddle + ", " + tofLeft + ", "  + tofRight);

                            BuddySDK.USB.setBuddySpeed(0.0f, 0.0f, 0, new IUsbCommadRsp.Stub() {
                                @Override
                                public void onSuccess(String s) throws RemoteException { }

                                @Override
                                public void onFailed(String s) throws RemoteException {}
                            });
                            step_num = 10;
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
