package com.bfr.opencvapp.grafcet;


import static com.bfr.opencvapp.MainActivity.personTrackerVIT;

import android.util.Log;
import com.bfr.opencvapp.utils.bfr_Grafcet;

/***
 * In FollowMe mode, computes the angular speed needed to align the body with the target
 */
public class SpeedAngularGrafcet extends bfr_Grafcet {

    public SpeedAngularGrafcet(String mname) {
        super(mname);
        this.grafcet_runnable = mysequence;
    }


    // Static variable (to manage the grafcet from outside)
    public static int step_num =0;
    public static boolean go = false;

    private int previous_step = 0;
    private double time_in_curr_step = 0;
    private boolean timeout = false;

    public float angularSpeed =1.0f;
    public float accel =0.5f;

    final float BASE_SPEED=0.9f;
    final float BASE_LOW_SPEED=0.15f;
    float targetangle = 0.0f;

    Point target;
    int targetX, targetY;
    public float noOffset=0.0f;


    // runable for grafcet
    private Runnable mysequence = new Runnable()
    {
        @Override
        public void run()
        {

            try {

                /*** Compute target position */
                target = getCentroid(personTrackerVIT.tracked.box.x,
                        personTrackerVIT.tracked.box.y,
                        personTrackerVIT.tracked.box.height,
                        personTrackerVIT.tracked.box.width
                );
                targetX = (int) target.x;
                targetY = (int) target.y;

                // compute angle for the wideAngle camera
                // resolution of 1024x768, with a 120° aperture
                // => 1pixel ~= 120 / sqrt(1024^2+768^2) = 0.09375
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

                        if(Math.abs(noOffset)>10.0f)
                            step_num = 15;
                        break;


                    case 15: // compute angular speed to align with target

                        // if target is within 30 pixels margins
                        if (personTrackerVIT.tracked.box.x>30 && (personTrackerVIT.tracked.box.x+ personTrackerVIT.tracked.box.width)<(1024-30))
                        {
                            // Big angle => higher speed
                            if(noOffset>= 15.0f) {
                                accel = 1.1f;
                                angularSpeed = -BASE_SPEED;
                            }
                            // small angle => lower speed
                            else if(noOffset> 5 && noOffset < 15.0f)
                            {
                                angularSpeed =-BASE_LOW_SPEED;
                            }
                            else if(noOffset<=-15.0f)
                            {
                                accel = 1.1f;
                                angularSpeed =BASE_SPEED;
                            }
                            else if (noOffset< -5 && noOffset > -15.0f)
                            {
                                angularSpeed = BASE_LOW_SPEED;
                            }
                            else // target in range
                            {
                                angularSpeed = 0.0f;
                            }
                        }
                        else // bbox on the image edge => High speed
                        {
                            if (personTrackerVIT.tracked.box.x>30)
                                angularSpeed =-1.0f;
                            else if((personTrackerVIT.tracked.box.x+ personTrackerVIT.tracked.box.width)<(1024-30))
                                angularSpeed =1.0f;
                        } // end if box touches the image edges


                        targetangle = noOffset;

                        // stay in this step in an infinite loop
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

    //Point in the image with coords in pixel
    class Point{
        int x = 0;
        int y = 0;
    }
}
