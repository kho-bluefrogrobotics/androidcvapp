package com.bfr.opencvapp.utils;

/**
 Get the centroid of a bbox (from upper left corner coordinates and height/width) in pixel
 */
public class BboxCentroid {

    public int x = 0;
    public int y = 0;

    public BboxCentroid()
    {
        this.x = 0;
        this.y = 0;
    }

    public BboxCentroid(int x, int y, int height, int width)
    {
        this.x = x + (int)(width/2);
        this.y = y + (int)(height/2);
    }

    public void getCentroid(int x, int y, int height, int width)
    {
        this.x = x + (int)(width/2);
        this.y = y + (int)(height/2);
    }
}
