package com.bfr.opencvapp.grafcet;


import static com.bfr.opencvapp.MainActivity.alignBodyAndFollowGrafcet;
import static com.bfr.opencvapp.MainActivity.initGrafcet;
import static com.bfr.opencvapp.MainActivity.personTrackerVIT;
import static com.bfr.opencvapp.MainActivity.speedAngularGrafcet;
import static com.bfr.opencvapp.MainActivity.speedLinearGrafcet;

import android.util.Log;

import com.bfr.opencvapp.MainActivity;
import com.bfr.opencvapp.utils.bfr_Grafcet;

public class MainGrafcet extends bfr_Grafcet {

    public MainGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;
    }

    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;


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
                    case 0: // Wait for begining

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
                        if(!initGrafcet.go)
                        {
                            initGrafcet.stop();
                            initGrafcet.go = false;

                            step_num = 9;
                        }
                        break;

                    case 9: // Wait for tracking OK
                        if(personTrackerVIT.isTracking)
                        {
                            step_num = 10;
                        }

                        if(timeout) // if no tracking (timeout)
                            step_num = 20; // activate search person

                        break;


                    case 10:// redirect to WatchMe Follow or ComeHere

                        if(MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.COMEHERE)
                            step_num = 60;
                        else if(MainActivity.followmeMode == MainActivity.FOLLOWME_MODE.FOLLOWME)
                            step_num = 20;
                        else
                            step_num = 40;

                        break;



                    case 20: // Follow me mode
                        alignBodyAndFollowGrafcet.go = true;
                        alignBodyAndFollowGrafcet.step_num = 0;

                        speedAngularGrafcet.go = true;
                        speedAngularGrafcet.step_num = 0;

                        speedLinearGrafcet.go = true;
                        speedLinearGrafcet.step_num = 0;

                        TrackingYesGrafcet.go = true;
                        FaceGrafcet.go = true;

                        step_num = 90;
                        break;




                    case 40: //WatchMe mode
                        TrackingNoGrafcet.go = true;
                        TrackingYesGrafcet.go = true;
                        AlignBodyGrafcet.go = true;
                        FaceGrafcet.go = true;

                        step_num = 90;
                        break;



                    case 60: //ComeHere mode
                        TrackingNoGrafcet.go = true;
                        TrackingYesGrafcet.go = true;
                        AlignBodyAndComeHereGrafcet.go = true;
                        FaceGrafcet.go = true;

                        step_num = 90;
                        break;


                    case 90 : // if tracking lost
                        if (personTrackerVIT.frameCount==0)
                        {
                            step_num = 100;
                        }
                        break;

                    case 100:// activate search person

                        TrackingNoGrafcet.go = false;
                        TrackingYesGrafcet.go = false;
                        AlignBodyGrafcet.go = false;
                        TrackingNoGrafcet.step_num=0;
                        TrackingYesGrafcet.step_num=0;
                        AlignBodyGrafcet.step_num=0;

                        SpeedLinearGrafcet.go=false;
                        SpeedLinearGrafcet.step_num=0;
                        SpeedAngularGrafcet.go=false;
                        SpeedAngularGrafcet.step_num = 0;
                        AlignBodyAndFollowGrafcet.go=false;
                        AlignBodyAndFollowGrafcet.step_num=0;

                        SearchPersonGrafcet.go=true;
                        SearchPersonGrafcet.step_num=0;

                        step_num = 105;
                        break;

                    case 105: //wait for end of searchperson
                        if(!SearchPersonGrafcet.go)
                        {
                            SearchPersonGrafcet.step_num=0;
                            SearchPersonGrafcet.go = false;
                            step_num = 9;
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
