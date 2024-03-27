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

public class AlignBodyFollowGrafcet extends bfr_Grafcet {

    public AlignBodyFollowGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }


    private AlignBodyFollowGrafcet grafcet=this;

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
    float rotspeed=1.0f;
    float linearspeed = 0.0f;

    final float BASE_SPEED=0.7f;
    float targetangle = 0.0f;

    Point target;
    int targetX, targetY;
    public static float noOffset=0.0f;
    long timerotating=0;

    float rotationSpeed = 15.0F;

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

                // Compute target position
                target = getCentroid(personTracker.tracked.box.x,
                        personTracker.tracked.box.y,
                        personTracker.tracked.box.height,
                        personTracker.tracked.box.width
                );
                targetX = (int) target.x;
                targetY = (int) target.y;
//                    Log.d(name, "Target at " + targetX + "," + targetY);
                // compute angle
                noOffset = (targetX-(1024/2))*0.09375f;


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

                        if(Math.abs(noOffset)>5.0f)
                            step_num = 15;
                        break;


                    case 12: // timer for stabilization
                        Thread.sleep(800);
                        step_num = 15;
                        break;

                    case 15: // rotate body to align


                        ackWheels = "";
                        timerotating = System.currentTimeMillis();
                        if (noOffset>= 15.0f )
                            rotspeed=-BASE_SPEED;
                        else if(noOffset>= 5 && noOffset < 15.0f)
                            rotspeed=-0.3f;
                        else if(noOffset<=-15.0f)
                            rotspeed=BASE_SPEED;
                        else
                            rotspeed = 0.3f;
                        Log.i(name, "**** Nooffset =" + noOffset + " rotspeed="+rotspeed );

                        targetangle = noOffset;


                        if (personTracker.torsoHeight<=200)
                            linearspeed = 0.2f;
                        else
                            linearspeed = 0.0f;

                        BuddySDK.USB.setBuddySpeed(linearspeed, rotspeed, 9999.0f, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {
                                ackWheels = s;
                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {
                                ackWheels = s;
                            }
                        });

//                        step_num = 17;
                        step_num = 20;
                        break;

                    case 17: //wait for OK
                        if (ackWheels.toUpperCase().contains("OK") ||
                                ackWheels.toUpperCase().contains("CANCELED") ||
                                timeout ) {
                            timerotating = System.currentTimeMillis();
                            step_num = 20;
                        }
                        break;


                    case 20: // wait for target in range

//                        if(Math.abs(noOffset)<=10.0f)
                        float angleInrads = (float) Math.toRadians(Math.abs(targetangle));
//                        Log.i(name, "########## Elapsed time= " + (int)((System.currentTimeMillis()-timerotating)/1000)
//                                +"\n Nooffset =" + angleInrads + " rotspeed="+rotspeed );


                        if (personTracker.torsoHeight<=200)
                            linearspeed = 0.2f;
                        else
                            linearspeed = 0.0f;



                        if( (int)((System.currentTimeMillis()-timerotating)) >= Math.abs(angleInrads/rotspeed)*1000 ) {
                            Log.e(name, "currtime= " + timerotating + " vs " + System.currentTimeMillis()
                                    + "\n for offset = " + angleInrads + " rotSpeed=" + rotationSpeed);
                            rotspeed = 0.0f;
                        } //end if time elapsed OK


                            BuddySDK.USB.setBuddySpeed(linearspeed, rotspeed, 99999.0f, new IUsbCommadRsp.Stub() {
                                @Override
                                public void onSuccess(String s) throws RemoteException {
                                    ackWheels = s;
                                    Log.w(name, "answer from motors: " + ackWheels);
                                }

                                @Override
                                public void onFailed(String s) throws RemoteException {
                                    ackWheels = s;
                                    Log.w(name, "answer from motors: " + ackWheels);
                                }
                            });

                           step_num = 15;



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
