package com.bfr.opencvapp.utils;

import static org.opencv.core.CvType.*;

import android.util.Log;

import org.opencv.core.Mat;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class IdentitiesDatabase implements java.io.Serializable{

    private String STORAGE_FILE = "/sdcard/identities.ser";



    // List of known faces
    public ArrayList<FacialIdentity> identities = new ArrayList<>();

    // serializable object
    private ArrayList<SerializableId> serializableIdentities = new ArrayList<>();


    public IdentitiesDatabase(){
        identities.add(new FacialIdentity("UNKNOWN", Mat.zeros(1,128, CV_32F)));
    }

    public void loadFromStorage()
    {
        try {
            FileInputStream fileIn = new FileInputStream(STORAGE_FILE);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            serializableIdentities = (ArrayList<SerializableId>) in.readObject();
            in.close();
            fileIn.close();
        } catch (Exception e) {
            Log.e("IdentityDatabase", "Error during loading identities: " + Log.getStackTraceString(e));
        }

        // for each element of serializable object
        for (int i=0; i< serializableIdentities.size(); i++)
        {
            // converting to array of identities
            identities.add(new FacialIdentity(
                    serializableIdentities.get(i).name,
                    Mat.zeros(1,128, CV_32F)));
            // filling embeddings
            for (int idx = 0; idx < identities.get(identities.size()-1).embedding.total(); idx++) {

                /* doesn't work
                identities.get(identities.size()-1).embedding.put(0,0,
                        serializableIdentities.get(i).floats[idx]);*/


                //store data
                double[] data = identities.get(identities.size() - 1).embedding.get(0, idx);

                data [0] =(float)serializableIdentities.get(i).floats[idx];

                identities.get(identities.size()-1).embedding.put(0, idx, data);

                Log.w("IdentityDatabase", "filling floats in array "
                        + "from " + serializableIdentities.get(i).floats[idx]
                        +" to " +  identities.get(identities.size() - 1).embedding.get(0, idx)[0]);

            }
        }

    }

    public void saveToStorage()
    {
        try {
        //for each saved face
        for (int i=0; i<identities.size(); i++) {
            //convert mat to byte array
            int length = (int) (identities.get(i).embedding.total());
            Log.w("IdentityDatabase", "Adding new floats for " + identities.get(i).name
            + "of size " + length);
//            byte[] bytes = new byte[length];
//            identities.get(i).embedding.get(0,0,bytes);

            // add to serializable object
            serializableIdentities.add(new SerializableId(identities.get(i).name,
                    new float[length]));

            for (int idx = 0; idx < identities.get(i).embedding.total(); idx++) {
                if (serializableIdentities.get(serializableIdentities.size() - 1)
                        .floats == null)
                    Log.w("IdentityDatabase", "serializableIdentities floats is null");
                if (identities.get(i).embedding == null)
                    Log.w("IdentityDatabase", "identities.get(i).embedding is null");
                else
                    Log.w("IdentityDatabase", "identities.get(i).embedding size " + identities.get(i).embedding.size()
                    +"= " + identities.get(i).embedding.elemSize()
                    +"-- " + identities.get(i).name);

                Log.w("IdentityDatabase", "starting filling floats array " +  serializableIdentities.get(serializableIdentities.size() - 1)
                        .floats.length);

                serializableIdentities.get(serializableIdentities.size() - 1)
                        .floats[idx] = (float) identities.get(i).embedding.get(0, idx)[0];
            }
        }// nextidentity

            //Serialize

                FileOutputStream fileOut =
                        new FileOutputStream(STORAGE_FILE);
                ObjectOutputStream out = new ObjectOutputStream(fileOut);
                out.writeObject(serializableIdentities);
                out.close();
                fileOut.close();
                System.out.printf("Serialized data is saved in /tmp/employee.ser");

            } catch (Exception e) {
                Log.e("IdentityDatabase", "Error during saving identities: " + Log.getStackTraceString(e));
            }

    }

}
