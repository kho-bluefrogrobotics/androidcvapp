package com.bfr.opencvapp.utils;

import org.opencv.core.Mat;

public class FacialIdentity implements java.io.Serializable{

    public String name = "";
    public Mat embedding;

    public FacialIdentity()
    {
        embedding = new Mat();
    }

    public FacialIdentity(String nameInput, Mat embeddingInput)
    {
        name = nameInput;
        embedding = embeddingInput;
    }
}
