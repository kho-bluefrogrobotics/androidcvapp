package com.bfr.opencvapp.utils;

import org.opencv.core.Mat;

public class FacialIdentity implements java.io.Serializable{

    public String name = "";
    public Mat embedding;
    public float recogScore = 0.0f;

    public FacialIdentity()
    {
        embedding = new Mat();
    }

    public FacialIdentity(String nameInput, Mat embeddingInput)
    {
        name = nameInput;
        embedding = embeddingInput;
    }

    public FacialIdentity(String nameInput, Mat embeddingInput, float scoreInput)
    {
        name = nameInput;
        embedding = embeddingInput;
        recogScore = scoreInput;
    }
}
