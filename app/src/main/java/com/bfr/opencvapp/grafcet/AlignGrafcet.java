package com.bfr.opencvapp.grafcet;


//import static com.bfr.opencvapp.MainActivity.alignCheckbox;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.buddysdk.services.companion.TaskCallback;
import com.bfr.opencvapp.utils.bfr_Grafcet;

public class AlignGrafcet extends bfr_Grafcet {

    public AlignGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;

    }


    private AlignGrafcet grafcet=this;

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
                        if (go) {
                            // go to next step
                            step_num = 5;
                        }
                        break;

                    case 5: //init wheels
                        BuddySDK.USB.enableWheels(1, 1, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {

                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {

                            }
                        });
                        step_num = 7;
                        break;

                    case 7: //wait for enabled
                        if (!BuddySDK.Actuators.getLeftWheelStatus().toUpperCase().contains("DISABLE")
                                && !BuddySDK.Actuators.getRightWheelStatus().toUpperCase().contains("DISABLE")) {
                            step_num = 10;
                        }
                        break;

                    case 10: // sync with trackingNo grafcet

                        if (TrackingNoGrafcet.waitingForAlign)
                                step_num = 15;

                        if(rotationRequest)
                            step_num = 50;
                        break;

                    case 15: // rotate body to align
                        ackWheels = "";
                        Log.i(name, "Rotating to " + BuddySDK.Actuators.getNoPosition());

                        BuddySDK.USB.rotateNoPrecision(40.0f, -BuddySDK.Actuators.getNoPosition(), 0, new TaskCallback() {
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

//                        BuddySDK.USB.rotateBuddy(40.0f, -BuddySDK.Actuators.getNoPosition(), new IUsbCommadRsp.Stub() {
//                            @Override
//                            public void onSuccess(String s) throws RemoteException {
//                                ackWheels=s;
//                            }
//
//                            @Override
//                            public void onFailed(String s) throws RemoteException {
//                                ackWheels=s;
//                            }
//                        });

                        // compensate with head (NO)
                        BuddySDK.USB.buddySayNo(25.0f, 0.0f, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {
                                ackNo = s;
                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {
                                ackNo = s;
                            }
                        });

                        step_num = 17;
                        break;

                    case 17: //wait for OK
                        if (ackWheels.toUpperCase().contains("OK") || timeout ) {
                            step_num = 20;
                        }
                        break;


                    case 20: // wait for end of mvt
                        if (ackWheels.toUpperCase().contains("FINISHED") || timeout ) {
                            if (ackNo.toUpperCase().contains("FINISHED")|| timeout )
                            {
                                //reset handshake
                                TrackingNoGrafcet.waitingForAlign = false;
                                // go to sync step
                                step_num = 10;
                            }

//                                step_num = 30;
                        }
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


}
