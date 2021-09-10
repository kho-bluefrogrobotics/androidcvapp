package com.bfr.opencvapp;

import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class bfr_Grafcet
{
    private final String TAG = "BFR_GRAFCET";

    // Default Params for scheduler
    // initial delay before start
    private final long INITIALDELAY = 0;
    // time period of execution
    private final long TIME_PERIOD = 20;

    // grafcet step number
    public int step_num = 0 ;
    // Scheduler for grafcet
    private ScheduledExecutorService myscheduler ;

    // Name
    public String name = "";

    // state
    public boolean started = false;

    // Runnable containing the grafcet sequence
    private Runnable grafcet_runnable;

    // *********************  Constructor overloads
    bfr_Grafcet(Runnable runnable, String mname)
    {
        name = mname;
        grafcet_runnable = runnable;
    }

    // start grafcet with default parameters
    public void start()
    {
        //if not started yet
        if (myscheduler == null || myscheduler.isShutdown())
        {
            //Log.i(TAG, "Grafcet " + name + " starting \n" + Log.getStackTraceString(new Exception()));
            Log.i(TAG, "\n********* GRAFCET **********\nGrafcet " + name + " starting \n" + Log.getStackTraceString(new Exception()));
            //init thread
            myscheduler = Executors.newScheduledThreadPool(1);
            // start scheduled task
            myscheduler.scheduleWithFixedDelay(grafcet_runnable, INITIALDELAY, TIME_PERIOD, TimeUnit.MILLISECONDS);
        }
        else // grafcet already started
        {
            //Log.i(TAG, "Grafcet "+ name+ " already started. Skipping \n" + Log.getStackTraceString(new Exception()));
            Log.i(TAG, "\n********* GRAFCET **********\nGrafcet "+ name+ " already started. Skipping \n" + Log.getStackTraceString(new Exception()));
        }

        // update state
        started = true;
    }

    // start grafcet and specify params (time period, ...)
    public void start(long mtime_period)
    {
        //if not started yet
        if (myscheduler.isShutdown())
        {
            Log.i(TAG, "\n********* GRAFCET **********\nGrafcet " + name + " starting \n");
            // log the stack strace in verbose mode
            Log.v(TAG, "Grafcet " + name + " called at " + Log.getStackTraceString(new Exception()));
            //init thread
            myscheduler = Executors.newScheduledThreadPool(1);
            // start scheduled task
            //myscheduler.scheduleWithFixedDelay(grafcet_runnable, INITIALDELAY, mtime_period, TimeUnit.MILLISECONDS);
            myscheduler.scheduleWithFixedDelay(grafcet_runnable, INITIALDELAY, mtime_period, TimeUnit.MILLISECONDS);
        }
        else // grafcet already started
        {
            Log.i(TAG, "\n********* GRAFCET **********\nGrafcet "+ name+ " already started. Skipping \n" );
            // log the stack strace in verbose mode
            Log.v(TAG, "Grafcet " + name + " called at " + Log.getStackTraceString(new Exception()));
        }

        // update state
        started = true;
    }


    public void start(long mtime_period, long m_init_delay )
    {
        //if not started yet
        if (myscheduler.isShutdown())
        {
            Log.i(TAG, "\n********* GRAFCET **********\nGrafcet " + name + " starting \n");
            // log the stack strace in verbose mode
            Log.v(TAG, "Grafcet " + name + " called at " + Log.getStackTraceString(new Exception()));
            //init thread
            myscheduler = Executors.newScheduledThreadPool(1);
            // start scheduled task
            myscheduler.scheduleWithFixedDelay(grafcet_runnable, m_init_delay, mtime_period, TimeUnit.MILLISECONDS);
        }
        else // grafcet already started
        {
            Log.i(TAG, "\n********* GRAFCET **********\nGrafcet "+ name+ " already started. Skipping \n");
            // log the stack strace in verbose mode
            Log.v(TAG, "Grafcet " + name + " called at " + Log.getStackTraceString(new Exception()));
        }

        // update state
        started = true;

    }

    // Stop the grafcet
    public void stop()
    {
        //if not started yet
        if (myscheduler.isShutdown())
        {
            Log.i(TAG, "\n********* GRAFCET **********\nGrafcet " + name + " not started yet \n");
            // log the stack strace in verbose mode
            Log.v(TAG, "Grafcet " + name + " Required to stop at " + Log.getStackTraceString(new Exception()));
        }
        else // grafcet already started
        {
            Log.i(TAG, "\n********* GRAFCET **********\nShutting down grafcet " + name + "\n");
            // log the stack strace in verbose mode
            Log.v(TAG, "Grafcet " + name + " required to stop at " + Log.getStackTraceString(new Exception()));
            myscheduler.shutdown();
        }

        // update state
        started = false;
    }
} // end of class bfr_grafcet
