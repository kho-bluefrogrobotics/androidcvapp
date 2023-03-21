package com.bfr.opencvapp.utils;

import android.graphics.Bitmap;
import android.util.Log;
import android.widget.Toast;

import com.google.mlkit.vision.face.Face;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceRecognizerSF;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;

public class FaceRecognizer {

    private final String TAG = "FaceRecognizerSface";

    //
    private final float MARGIN_FACTOR = 0.05f;
    // default file for storing identities
    private final String STORAGE_FILE = "/sdcard/identities.bin";
    //where to find the models
    private final String DIR = "/sdcard/Android/data/com.bfr.opencvapp/files/nnmodels/";
    private final String MODEL_NAME = "/face_recognition_sface_2021dec.onnx";
//    private final String MODEL_NAME = "face_recognition_sface_2021dec-act_int8-wt_int8-quantized.onnx";
    // Face recognition
    private FaceRecognizerSF faceRecognizer;
    Mat faceEmbedding;
    Mat faceMat, rotatedFace;
    Rect faceROI;

    // MLKit face detector
    private MLKitFaceDetector mlKitFaceDetector = new MLKitFaceDetector();

    // image rotation
    Point center = new Point();
    double angle = 0;
    Mat mapMatrix;

    // List of known faces
    private IdentitiesDatabase idDatabase;
    //
    private ArrayList<FacialIdentity> kTopResults = new java.util.ArrayList<>();

    SimpleDateFormat sdf = new SimpleDateFormat("yyMMddHHmmss");
    String currentDateandTime = "";

    //to debug
    double elapsedTime=0.0;
    public boolean withPreprocess=true;

    public FaceRecognizer()
    {
        // init
        faceEmbedding=new Mat();
        idDatabase= new IdentitiesDatabase();
        try{
            idDatabase.loadFromStorage(STORAGE_FILE);
        }
        catch (Exception e){
            e.printStackTrace();
        }

        // Load face recog model
        faceRecognizer = FaceRecognizerSF.create(DIR + MODEL_NAME,
                "");

    } // end constructor


    // creates the comparator fto sort from the closest to the most different face
    class FacialIdComparator implements Comparator<FacialIdentity> {

        // override the compare() method
        @Override
        public int compare(FacialIdentity o1, FacialIdentity o2) {
            if (o1.recogScore == o2.recogScore)
                return 0;
            else if (o1.recogScore < o2.recogScore)
                return 1;
            else
                return -1;
        }
    }//end comparator

    /**
     * Pre-process to align and crop face
     * @param frame original frame
     * @param leftIn coord to crop the frame where the face is, in % of the original size
     * @param rightIn coord to crop the frame where the face is, in % of the original size
     * @param bottomIn coord to crop the frame where the face is, in % of the original size
     * @param topIn coord to crop the frame where the face is, in % of the original size
     * @return a FacialIdentity object containing the name of the face and the recognition score
     * returns null if the recognition fails
     */
    private Mat cropAndAlign(Mat frame, float leftIn, float rightIn, float topIn, float bottomIn)
    {
        int cols = frame.cols();
        int rows = frame.rows();

        int left   = (int)(leftIn * cols);
        int top    = (int)(topIn * rows);
        int right  = (int)(rightIn * cols);
        int bottom = (int)(bottomIn* rows);

        //If face touches the margin, skip -> we need a fully visible face for recognition
        if(left<10 || top <10 || right> frame.cols()-10 || bottom > frame.rows()-10)
            // take next detected face
            return null;
        elapsedTime = System.currentTimeMillis();
        //crop image around face
//        faceROI= new Rect(
//                // alternative to crop more: Math.max(left, (int)(left- MARGIN_FACTOR *(right-left)) ),
//                left,
//                Math.max(top, (int)(top+ 0.5*MARGIN_FACTOR *(bottom-top)) ),
//                //alternative to crop more: (int)(right-left)+(int)(MARGIN_FACTOR *(right-left)),
//                right-left,
//                (bottom-top) -(int)(MARGIN_FACTOR *(bottom-top)));
//                //alternative to crop less: (int)(bottom-top));

        faceROI= new Rect(
                // alternative to crop more: Math.max(left, (int)(left- MARGIN_FACTOR *(right-left)) ),
                left,
                top,
                //alternative to crop more: (int)(right-left)+(int)(MARGIN_FACTOR *(right-left)),
                right-left,
                (bottom-top) );


        faceMat = frame.submat(faceROI);

//        rotatedFace = faceMat.clone();

        if(withPreprocess) {
            /*** face orientation*/
            //convert to bitmap
            Bitmap bitmapImage = Bitmap.createBitmap(faceMat.cols(), faceMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(faceMat, bitmapImage);
            // detect-classify face
            Face detectedFace = mlKitFaceDetector.detectSingleFaceFromBitmap(bitmapImage);

            Log.i(TAG, "elapsed time face MLKit : " + (System.currentTimeMillis() - elapsedTime));
            elapsedTime = System.currentTimeMillis();

            if (detectedFace == null) {
                Log.i(TAG, "error processing image with MLKit");
                return null;
            }
            //adjust crop
            //crop image around face
            Log.i("coucou", "face rect " + detectedFace.getBoundingBox().left
            + " "+ detectedFace.getBoundingBox().right
            + " " + detectedFace.getBoundingBox().top
            + " " + detectedFace.getBoundingBox().bottom);

            right = left + detectedFace.getBoundingBox().right;
            left = left + detectedFace.getBoundingBox().left;
            bottom = top + detectedFace.getBoundingBox().bottom + (int)(MARGIN_FACTOR *(bottom-top));
            top = top + + detectedFace.getBoundingBox().top;

            Rect newfaceROI= new Rect(
                    // alternative to crop more: Math.max(left, (int)(left- MARGIN_FACTOR *(right-left)) ),
                    left,
                    top,
                    //alternative to crop more: (int)(right-left)+(int)(MARGIN_FACTOR *(right-left)),
                    right-left,
                    (bottom-top) );

            faceMat = frame.submat(newfaceROI);

            // image rotation
            center.x = faceMat.cols() / 2;
            center.y = faceMat.rows() / 2;
            angle = detectedFace.getHeadEulerAngleZ();
            mapMatrix = Imgproc.getRotationMatrix2D(center, -angle, 1.0);
            // rotate
            Imgproc.warpAffine(faceMat, faceMat, mapMatrix, new Size(faceMat.cols(), faceMat.rows()));
        }
        return faceMat;
    }

    /**
     * Recognize a face from a cropped image
     * @param frame original frame
     * @param leftIn coord to crop the frame where the face is, in % of the original size
     * @param rightIn coord to crop the frame where the face is, in % of the original size
     * @param bottomIn coord to crop the frame where the face is, in % of the original size
     * @param topIn coord to crop the frame where the face is, in % of the original size
     * @return a FacialIdentity object containing the name of the face and the recognition score
     * returns null if the recognition fails
     */
    public FacialIdentity RecognizeFace(Mat frame, float leftIn, float rightIn, float topIn, float bottomIn)
    {
        elapsedTime = System.currentTimeMillis();

        faceRecognizer.feature(
                cropAndAlign(frame, leftIn, rightIn, topIn, bottomIn),
                faceEmbedding);

        Log.i(TAG, "elapsed time calc embedding : " + (System.currentTimeMillis()-elapsedTime)
                +"\n " + faceEmbedding.size() +" type " + faceEmbedding.depth());
        elapsedTime = System.currentTimeMillis();

        ////////////////////////////// Matching

        // Look for closest
        double cosineScore, maxScore = 0.0;
        int identifiedIdx=0;
        kTopResults.clear();

        // for each known face
        for (int faceIdx=0; faceIdx< idDatabase.identities.size(); faceIdx++)
        {

            //compute similarity;
            cosineScore = faceRecognizer.match(faceEmbedding, idDatabase.identities.get(faceIdx).embedding,
                    FaceRecognizerSF.FR_COSINE);
//                        Log.i(TAG, "TODEBUG FaceCOmputing : " + idDatabase.identities.get(faceIdx).name + " - " + cosineScore
//                         + "   " + idDatabase.identities.get(faceIdx).embedding.get(0,0)[0]
//                         + " " + idDatabase.identities.get(faceIdx).embedding.get(0,1)[0]
//                         + " " + idDatabase.identities.get(faceIdx).embedding.get(0,2)[0]
//                         + " " + idDatabase.identities.get(faceIdx).embedding.get(0,3)[0]
//                         + " " + idDatabase.identities.get(faceIdx).embedding.get(0,4)[0]
//                        );

            // store k-top results
            kTopResults.add(idDatabase.identities.get(faceIdx));
            kTopResults.get(kTopResults.size()-1).recogScore = (float)cosineScore;

            // if better score
            if(cosineScore>maxScore) {
                maxScore = cosineScore;
                identifiedIdx = faceIdx;
            }
        } //next known face

        // sort results from closest to most different
        kTopResults.sort(new FacialIdComparator());

        // return recognized face
        return new FacialIdentity(idDatabase.identities.get(identifiedIdx).name.split("_")[0], new Mat(), (float) maxScore);

    }

    /**
     * Save the identity of a face found in a cropped image
     * @param frame original frame
     * @param leftIn coord to crop the frame where the face is, in % of the original size
     * @param rightIn coord to crop the frame where the face is, in % of the original size
     * @param bottomIn coord to crop the frame where the face is, in % of the original size
     * @param topIn coord to crop the frame where the face is, in % of the original size
     * @param name name to link to the face
     * @return null
     */
    public void saveFace(Mat frame, float leftIn, float rightIn, float topIn, float bottomIn, String name)
    {
        saveFace(frame, leftIn, rightIn, topIn, bottomIn, name, STORAGE_FILE);
    }

    /**
     * Save the identity of a face found in a cropped image
     * @param frame original frame
     * @param leftIn coord to crop the frame where the face is, in % of the original size
     * @param rightIn coord to crop the frame where the face is, in % of the original size
     * @param bottomIn coord to crop the frame where the face is, in % of the original size
     * @param topIn coord to crop the frame where the face is, in % of the original size
     * @param name name to link to the face
     * @param storingFile where to store all the identities
     * @return null
     */
    public void saveFace(Mat frame, float leftIn, float rightIn, float topIn, float bottomIn, String name, String storingFile)
    {
        faceRecognizer.feature(
                cropAndAlign(frame, leftIn, rightIn, topIn, bottomIn),
                faceEmbedding);

        Log.i(TAG, "elapsed time calc embedding : " + (System.currentTimeMillis()-elapsedTime)
                +"\n " + faceEmbedding.size() +" type " + faceEmbedding.depth());
        elapsedTime = System.currentTimeMillis();

        ////////////////////////////// Saving new face
        //adding date and time to manage homonyms
        currentDateandTime = sdf.format(new Date());
        idDatabase.identities.add(new FacialIdentity(name+"_"+currentDateandTime, faceEmbedding.clone()));

        //saving file
        Thread savingThread = new Thread(new Runnable() {
            @Override
            public void run() {
                idDatabase.saveToStorage(storingFile);
                //TODEBUG only
                Mat toSave = new Mat();
                Imgproc.cvtColor(faceMat, toSave, Imgproc.COLOR_RGB2BGR);
                Imgcodecs.imwrite("/sdcard/faceReco_"+ name+"_"+currentDateandTime+".jpg", toSave);
            }
        });
        savingThread.start();

    } //end save face

    /**
     * Load the saved identities
     * @return null
     */
    public void loadFaces()
    {
        loadFaces(STORAGE_FILE);
    }

    /**
     * Load a custom file which contains saved identities
     * @param fileName file
     * @return null
     */
    public void loadFaces(String fileName)
    {
        idDatabase.identities.clear();
        idDatabase.loadFromStorage(fileName);
    }

    /**
     * Get the list of top-k recognized faces
     * @param k number of first k candidates
     * @return an array of FacialIdentity objects containing the name and embedding
     */
    public ArrayList<FacialIdentity> getTopKResults(int k)
    {
        ArrayList<FacialIdentity> candidates = new ArrayList<>();
        try{
            Log.i(TAG, "kTopResult size to get: " + kTopResults.size());
            for (int i=0; i<k; i++)
            {
                candidates.add(kTopResults.get(i));
            }
            return candidates;
        }
        catch (Exception e)
        {
            Log.i(TAG, "Error getting K-top candidates " + e);
            return null;
        }
    }


    /**
     * Get the list of saved identities
     * @return an array of FacialIdentity objects containing the name and embedding
     */
    public ArrayList<FacialIdentity> getSavedIdentities()
    {
        try{
            return idDatabase.identities;
        }
          catch (Exception e)
        {
            Log.i(TAG, "Error getting saved identities " + e);
            return null;
        }
    }

    /**
     * remove i-th save identity and save.
     * WARNING: this operation is not reversible
     * @param idx number of first k candidates
     * @return nothing
     */
    public void removeSavedIdentity(int idx)
    {
        removeSavedIdentity(idx, STORAGE_FILE);
    } // end remove identity


    /**
     * remove i-th save identity and save to a custom file an the device.
     * WARNING: this operation is not reversible
     * @param idx number of first k candidates
     * @param storageFile custom file to save on device
     * @return nothing
     */
    public void removeSavedIdentity(int idx, String storageFile)
    {
        try{
            // always leave at least one identity in the list
            if(idDatabase.identities.size()>1)
            {
                idDatabase.identities.remove(idx);
                //saving file
                Thread savingThread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        idDatabase.saveToStorage(storageFile);
                    }
                });
                savingThread.start();
            }
            else
                Log.w(TAG, "Can't remove the identity : you should always leave at least one identity in the database");

        } catch (Exception e)
        {
            Log.i(TAG, "Error removing saved identity " + e);
        }

    } // end remove identity

}
