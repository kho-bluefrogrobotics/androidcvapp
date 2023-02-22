package com.bfr.opencvapp.utils;

import static org.opencv.core.CvType.*;

import android.util.Log;

import org.opencv.core.Mat;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class IdentitiesDatabase {

    private String STORAGE_FILE = "/sdcard/identities.ser";

    private class serilizableFacialId implements java.io.Serializable{
        public String name = "";
        public byte[] bytes;

        public serilizableFacialId(String nameInput, byte[] inputBytes)
        {
            name = nameInput;
            bytes = inputBytes;
        }
    }


    // List of known faces
    public ArrayList<FacialIdentity> identities = new ArrayList<>();

    // serializable object
    private ArrayList<serilizableFacialId> serializableIdentities = new ArrayList<>();


    public IdentitiesDatabase(){
        identities.add(new FacialIdentity("UNKNOWN", Mat.zeros(1,128, CV_32F)));
    }

    public void loadFromStorage()
    {

    }

    public void saveToStorage()
    {
        //for each saved face
        for (int i=0; i<identities.size(); i++)
        {
            //convert mat to byte array
            int length = (int) (identities.get(i).embedding.total() * identities.get(i).embedding.elemSize());
            byte[] bytes = new byte[length];
            identities.get(i).embedding.get(0,0,bytes);
            // add to serializable object
            serializableIdentities.add(new serilizableFacialId(identities.get(i).name, bytes) );

            //Serialize
            try {
                FileOutputStream fileOut =
                        new FileOutputStream(STORAGE_FILE);
                ObjectOutputStream out = new ObjectOutputStream(fileOut);
                out.writeObject(serializableIdentities);
                out.close();
                fileOut.close();
                System.out.printf("Serialized data is saved in /tmp/employee.ser");
            } catch (Exception e) {
                Log.e("IdentityDatabase", "Error during saving identities: " + e);
            }
        }
    }

}
