package com.bfr.opencvapp.utils;

/**
 * Pose of a mobile robot
 *      - x (m)
 *      - y (m)
 *      - theta (rad)
 */
public class Pose {
    public double x, y, theta;

    public Pose(){
        x = 0;
        y = 0;
        theta = 0;
    }

    public Pose(Pose pose){
        this.x = pose.x;
        this.y = pose.y;
        this.theta = pose.theta;
    }

    public Pose(double x, double y, double theta){
        this.set(x, y, theta);
    }

    @Override
    public String toString() {
        return "Pose{" +
                "x=" + x +
                ", y=" + y +
                ", theta=" + Math.toDegrees(theta) +
                "°}";
    }

    public void set(double x, double y, double theta){
        this.x = x;
        this.y = y;
        this.theta = theta;
    }

    public void reformatTheta(){
        theta = Math.atan2(Math.sin(theta), Math.cos(theta));
    }

    public double distanceTo(Pose target){
        return Math.sqrt(Math.pow(target.x - x, 2) + Math.pow(target.y - y, 2));
    }
}
