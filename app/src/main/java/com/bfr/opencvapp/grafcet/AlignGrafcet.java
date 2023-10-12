package com.bfr.opencvapp.grafcet;


import static com.bfr.opencvapp.MainActivity.alignCheckbox;

import android.os.RemoteException;
import android.util.Log;

import com.bfr.buddy.usb.shared.IUsbCommadRsp;
import com.bfr.buddysdk.BuddySDK;
import com.bfr.opencvapp.utils.bfr_Grafcet;

import org.opencv.core.Rect;

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

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean bypass = false;

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
                } // end if step = same


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

                    case 10: // enable Yes
                        BuddySDK.USB.enableYesMove(1, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {

                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {

                            }
                        });
                        step_num = 12;
                        break;

                    case 12: //wait for enabled
                        if (!BuddySDK.Actuators.getYesStatus().toUpperCase().contains("DISABLE")
                        ) {
                            step_num = 15;
                        }
                        break;

                    case 15: // enable Yes
                        BuddySDK.USB.enableNoMove(1, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {

                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {

                            }
                        });
                        step_num = 17;
                        break;

                    case 17: //wait for enabled
                        if (!BuddySDK.Actuators.getNoStatus().toUpperCase().contains("DISABLE")
                        ) {
                            step_num = 20;
                        }
                        break;


                    case 20: // reset head Yes position
                        ackYes = "";
                        BuddySDK.USB.buddySayYes(20.0f, -20.0f, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {
                                ackYes = s;
                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {
                                ackYes = s;
                            }
                        });
                        step_num = 22;
                        break;

                    case 22: // wait end of Yes
                        if (ackYes.toUpperCase().contains("FINISH")
                        ) {
                            step_num = 25;
                        }
                        break;

                    case 25: // reset head Yes position
                        ackNo = "";
                        BuddySDK.USB.buddySayNo(40.0f, 0.0f, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {
                                ackNo = s;
                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {
                                ackNo = s;
                            }
                        });
                        step_num = 27;
                        break;

                    case 27: // wait end of Yes
                        if (ackNo.toUpperCase().contains("FINISH")
                        ) {
                            step_num = 30;
                        }
                        break;

                    case 30: // compute angle
                        /**
                         256x256 with 120° => 0,33145630368119415206289579473665 °/pixel
                         */

                        //xCenter in a 1xRESIZERATIO image (compared to 256 (orig) )
                        xorig = xCenter * 256 / RESIZE_RATIO;
                        deltaPixel = xorig - (256 / 2);
                        angleToRotate = (float) (deltaPixel * 0.3314563036);
                        Log.w(name, "Angle to rotate=" + angleToRotate);
                        if(Math.abs(angleToRotate)<10)
                            step_num = 40;
                        else
                            step_num = 31;
                        break;

                    case 31: // rotate
                        ackWheels = "";
                        BuddySDK.USB.rotateBuddy(60.0f, -angleToRotate, new IUsbCommadRsp.Stub() {
                            @Override
                            public void onSuccess(String s) throws RemoteException {
                                ackWheels = s;
                            }

                            @Override
                            public void onFailed(String s) throws RemoteException {
                                ackWheels = s;
                            }
                        });
                        step_num = 35;
                        break;

                    case 35: // wait end of rotation
                        if (ackWheels.toUpperCase().contains("FINISH")
                        ) {
                            step_num = 40;
                        }
                        break;

                    case 40: // wait
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        if (alignCheckbox.isChecked())
                            step_num = 30;
                        else
                            step_num = 0;
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
