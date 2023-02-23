package com.bfr.opencvapp.utils;

public class SerializableId implements java.io.Serializable{

    public String name = "";
    public float[] floats;

    public SerializableId (String nameInput, float[] inputFloats)
    {
        name = nameInput;
        floats = inputFloats;
    }

}
