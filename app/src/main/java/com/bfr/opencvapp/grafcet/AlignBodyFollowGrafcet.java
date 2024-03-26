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
                            step_num = 10;
                        }
                        break;


                    case 10: // check target offaxis alignment

                        if(Math.abs(noOffset)>10.0f)
                            step_num = 12;
                        break;


                    case 12: // timer for stabilization
                        Thread.sleep(800);
                        step_num = 15;
                        break;

                    case 15: // rotate body to align



                        ackWheels = "";
                        timerotating = System.currentTimeMillis();

//
//                        float radSpeed = (float) Math.toRadians(rotationSpeed);
//                        if (noOffset>0)
//                            radSpeed= radSpeed*-1;
//
//                        float dist= 0.1f* Math.abs(noOffset)/rotationSpeed;
//
//                        Log.i(name, "Rotating to " + noOffset + " at rotationspeed=" + rotationSpeed + " with dist="+dist);
//
//                        BuddySDK.USB.moveBuddy(0.01f, rotationSpeed, dist, 100.0f, new IUsbCommadRsp.Stub() {
//                            @Override
//                            public void onSuccess(String s) throws RemoteException {
//                                ackWheels = s;
//                                Log.w(name, "answer from motors: " + ackWheels);
//                            }
//
//                            @Override
//                            public void onFailed(String s) throws RemoteException {
//                                ackWheels = s;
//                                Log.w(name, "answer from motors: " + ackWheels);
//                            }
//                        });




//                        BuddySDK.USB.rotateBuddy(60.0f, -noOffset,
//                                0,
//                                1,
//                                new IUsbCommadRsp.Stub() {
//                            @Override
//                            public void onSuccess(String s) throws RemoteException {
//                                ackWheels = s;
//                            }
//
//                            @Override
//                            public void onFailed(String s) throws RemoteException {
//                                ackWheels = s;
//                            }
//                        });







                        BuddySDK.USB.rotateNoPrecision(50.0f, -noOffset, 0, new TaskCallback() {
                            @Override
                            public void onStarted() {
                                ackWheels = "OK";
                                Log.i("coucou", "task onstarted ");
                            }

                            @Override
                            public void onSuccess(String s) {
                                ackWheels = "FINISHED";
                                Log.i("coucou", "task onSuccess " + s);
                            }

                            @Override
                            public void onCancel() {
                                Log.i("coucou", "task onCancel ");
                            }

                            @Override
                            public void onError(String s) {
                                ackWheels = "error :" + s;
                                Log.i("coucou", "task onError " + s);
                            }
                        });

                        step_num = 20;
                        break;

                    case 17: //wait for OK
                        if (ackWheels.toUpperCase().contains("OK") ||
                                ackWheels.toUpperCase().contains("CANCELED") ||
                                timeout ) {
                            step_num = 20;
                        }
                        break;


                    case 20: // wait for end of mvt

                        if(ackWheels.toUpperCase().contains("FINISHED") ||
                                ackWheels.toUpperCase().contains("CANCELED") || timeout )
//                                if( (System.currentTimeMillis()- timerotating)/1000 >  Math.abs((noOffset/rotationSpeed)  )  )
                        {
//
//                            Log.w(name, "currtime= " + timerotating + " vs " +System.currentTimeMillis()
//                            + "\n for offset = " + noOffset  + " rotSpeed=" + rotationSpeed);
//
//                            BuddySDK.USB.setBuddySpeed(0.0f, 0.0f, 0.0f, new IUsbCommadRsp.Stub() {
//                                @Override
//                                public void onSuccess(String s) throws RemoteException {
//                                    ackWheels = s;
//                                    Log.w(name, "answer from motors: " + ackWheels);
//                                }
//
//                                @Override
//                                public void onFailed(String s) throws RemoteException {
//                                    ackWheels = s;
//                                    Log.w(name, "answer from motors: " + ackWheels);
//                                }
//                            });

                            if(Math.abs(noOffset)>10.0f)
                                step_num = 15;
                            else
                                step_num = 10;
                            
                        }

                        break;

                    case 21 : //wait

                        Thread.sleep(500);
                        step_num=10;
                        break;
                    case 30: // blink
                    BuddySDK.USB.updateAllLed("#eb3434", new IUsbCommadRsp.Stub() {
                        @Override
                        public void onSuccess(String s) throws RemoteException {

                        }

                        @Override
                        public void onFailed(String s) throws RemoteException {

                        }
                    });
                    step_num =33;
                    break;

                    case 33 : //wait
                        Thread.sleep(1000);
                        step_num = 35;
                        break;

                    case 35 : //reset LED
                        BuddySDK.USB.updateAllLed("#34dfeb", new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {

                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {

                            }
                        });
                        step_num = 10;
                        break;



                    case 50: //No at limit > request to rotate body

                        //reset
                        ackWheels = "";

                        BuddySDK.USB.rotateNoPrecision(50.0f, -TrackingNoGrafcet.noOffset, 0, new TaskCallback() {
                            @Override
                            public void onStarted() {
                                ackWheels = "OK";
                            }

                            @Override
                            public void onSuccess(String s) {
                                ackWheels = "FINISHED";
                            }

                            @Override
                            public void onCancel() {
                                ackWheels = "FINISHED";
                            }

                            @Override
                            public void onError(String s) {
                                ackWheels = "ERROR";
                            }
                        });


//                        BuddySDK.USB.updateAllLed("#9E74FF", new IUsbCommadRsp.Stub() {
//                            @Override
//                            public void onSuccess(String s) throws RemoteException {                            }
//
//                            @Override
//                            public void onFailed(String s) throws RemoteException {}
//                        });

                        step_num = 53;
                        break;

                    case 53 ://wait for OK
                        if(ackWheels.toUpperCase().contains("OK") || timeout)
                            step_num = 55;
                        break;

                    case 55: // wait for mvt finished
                        if(ackWheels.toUpperCase().contains("FINISHED") || timeout) {
                            step_num = 10;
                            rotationRequest = false;



//                            BuddySDK.USB.updateAllLed("#34dfeb", new IUsbCommadRsp.Stub() {
//                                @Override
//                                public void onSuccess(String s) throws RemoteException {                                }
//                                @Override
//                                public void onFailed(String s) throws RemoteException {                                }
//                            });

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
