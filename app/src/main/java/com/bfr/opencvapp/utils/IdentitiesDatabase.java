package com.bfr.opencvapp.utils;

import static org.opencv.core.CvType.*;

import android.util.Log;

import org.opencv.core.Mat;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class IdentitiesDatabase implements java.io.Serializable{


    // List of known faces
    public ArrayList<FacialIdentity> identities = new ArrayList<>();

    // serializable object
    private ArrayList<SerializableId> serializableIdentities = new ArrayList<>();


    public IdentitiesDatabase(){
        //init
        identities.add(new FacialIdentity("UNKNOWN", Mat.zeros(1,128, CV_32F)));
    }


    public void loadFromStorage(String fileName) throws Exception {
        try {
            //reset
            serializableIdentities.clear();
            identities.clear();

            FileInputStream fileIn = new FileInputStream(fileName);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            serializableIdentities = (ArrayList<SerializableId>) in.readObject();
            in.close();
            fileIn.close();

            // for each element of serializable object
            for (int i=0; i< serializableIdentities.size(); i++)
            {
                // converting to array of identities
                identities.add(new FacialIdentity(
                        serializableIdentities.get(i).name,
                        Mat.zeros(1,128, CV_32F)));
                // filling embeddings
                for (int idx = 0; idx < identities.get(identities.size()-1).embedding.total(); idx++) {

                    //store data
                    double[] data = identities.get(identities.size() - 1).embedding.get(0, idx);
                    data[0] = serializableIdentities.get(i).floats[idx];
                    identities.get(identities.size()-1).embedding.put(0, idx, data);

                } //next embedding
            } //next identity

        }
        catch (Exception e) {
            Log.e("IdentityDatabase", "Error during loading identities: " + Log.getStackTraceString(e));
            // init
            identities.add(new FacialIdentity(
                    "UNKNOWN",
                    Mat.zeros(1,128, CV_32F)));

            throw new Exception("Error during loading identities: " + e);
        }

    }// end loadfromstorage


    public void saveToStorage(String fileName) throws Exception {
        try {
                //reset
                serializableIdentities.clear();
                //for each saved face
                for (int i=0; i<identities.size(); i++) {
                    //convert mat to byte array
                    int length = (int) (identities.get(i).embedding.total());

                    // add to serializable object
                    serializableIdentities.add(new SerializableId(identities.get(i).name,
                            new float[length]));

                    for (int idx = 0; idx < identities.get(i).embedding.total(); idx++) {

                        serializableIdentities.get(serializableIdentities.size() - 1)
                                .floats[idx] = (float) identities.get(i).embedding.get(0, idx)[0];
                    } //next embedding
                }// next identity

                //Serialize
                FileOutputStream fileOut =
                        new FileOutputStream(fileName);
                ObjectOutputStream out = new ObjectOutputStream(fileOut);
                out.writeObject(serializableIdentities);
                out.close();
                fileOut.close();
            }
        catch (Exception e)
        {
            Log.e("IdentityDatabase", "Error during saving identities: " + Log.getStackTraceString(e));
            throw new Exception("Error during saving identities: " + e);
        }

    }

}
