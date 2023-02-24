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
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceRecognizerSF;

import java.text.SimpleDateFormat;
import java.util.Date;

public class FaceRecognizer {

    private final String TAG = "FaceRecognizerSface";

    //
    private final float MARGIN_FACTOR = 0.1f;
    // default file for storing identities
    private String STORAGE_FILE = "/sdcard/identities.bin";
    //where to find the models
    private final String DIR = "/sdcard/Android/data/com.bfr.opencvapp/files/";
    // Face recognition
    private FaceRecognizerSF faceRecognizer;
    Mat faceEmbedding;
    Mat faceMat;
    Rect faceROI;

    // MLKit face detector
    MLKitFaceDetector mlKitFaceDetector = new MLKitFaceDetector();

    // image rotation
    Point center = new Point();
    double angle = 0;
    Mat mapMatrix;

    // List of known faces
    private IdentitiesDatabase idDatabase;

    SimpleDateFormat sdf = new SimpleDateFormat("yyMMddHHmmss");
    String currentDateandTime = "";

    //to debug
    double elapsedTime=0.0;

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
        faceRecognizer = FaceRecognizerSF.create(DIR + "/nnmodels/face_recognition_sface_2021dec.onnx",
//        faceRecognizer = FaceRecognizerSF.create(dir + "/nnmodels/face_recognition_sface_2021dec-act_int8-wt_int8-quantized.onnx",
                "");

    } // end constructor


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

        //crop image around face
        faceROI= new Rect( Math.max(left, (int)(left- MARGIN_FACTOR *(right-left)) ),
                Math.max(top, (int)(top- MARGIN_FACTOR *(bottom-top)) ),
                (int)(right-left)+(int)(2* MARGIN_FACTOR *(right-left)),
                (int)(bottom-top) + +(int)(MARGIN_FACTOR *(bottom-top)));
        faceMat = frame.submat(faceROI);

        ////////////////////////////// face orientation
        //convert to bitmap
        Bitmap bitmapImage = Bitmap.createBitmap(faceMat.cols(), faceMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(faceMat, bitmapImage);
        // detect-classify face
        Face detectedFace = mlKitFaceDetector.detectSingleFaceFromBitmap(bitmapImage);

        Log.i(TAG, "elapsed time face MLKit : " + (System.currentTimeMillis()-elapsedTime)  );
        elapsedTime = System.currentTimeMillis();

        if(detectedFace==null){
            Imgproc.putText(frame, "MLKIT Failure", new Point(100, 100),1, 2,
                    new Scalar(0, 255, 0), 2);
            return null;
        }
        // image rotation
        center.x = (int)faceMat.cols()/2;
        center.y = (int) faceMat.rows()/2;
        angle = detectedFace.getHeadEulerAngleZ();
        mapMatrix = Imgproc.getRotationMatrix2D(center, -angle, 1.0);
        // rotate
        Imgproc.warpAffine(faceMat, faceMat, mapMatrix, new Size(faceMat.cols(), faceMat.rows()));

        Log.i(TAG, "elapsed time rotation : " + (System.currentTimeMillis()-elapsedTime)  );
        elapsedTime = System.currentTimeMillis();

        faceRecognizer.feature(faceMat, faceEmbedding);

        Log.i(TAG, "elapsed time calc embedding : " + (System.currentTimeMillis()-elapsedTime)
                +"\n " + faceEmbedding.size() +" type " + faceEmbedding.depth());
        elapsedTime = System.currentTimeMillis();

        ////////////////////////////// Matching

        // Look for closest
        double cosineScore, maxScore = 0.0;
        int identifiedIdx=0;

        // for each known face
        for (int faceIdx=1; faceIdx< idDatabase.identities.size(); faceIdx++)
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
            // if better score
            if(cosineScore>maxScore) {
                maxScore = cosineScore;
                identifiedIdx = faceIdx;
            }
        }

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
        int cols = frame.cols();
        int rows = frame.rows();

        int left   = (int)(leftIn * cols);
        int top    = (int)(topIn * rows);
        int right  = (int)(rightIn * cols);
        int bottom = (int)(bottomIn* rows);

        //If face touches the margin, skip -> we need a fully visible face for recognition
        if(left<10 || top <10 || right> frame.cols()-10 || bottom > frame.rows()-10)
            // take next detected face
            return;


        /*** Recognition ***/


        //crop image around face
        faceROI= new Rect( Math.max(left, (int)(left- MARGIN_FACTOR *(right-left)) ),
                Math.max(top, (int)(top- MARGIN_FACTOR *(bottom-top)) ),
                (int)(right-left)+(int)(2* MARGIN_FACTOR *(right-left)),
                (int)(bottom-top) + +(int)(MARGIN_FACTOR *(bottom-top)));
        faceMat = frame.submat(faceROI);


        ////////////////////////////// face orientation
        //convert to bitmap
        Bitmap bitmapImage = Bitmap.createBitmap(faceMat.cols(), faceMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(faceMat, bitmapImage);
        // detect-classify face
        Face detectedFace = mlKitFaceDetector.detectSingleFaceFromBitmap(bitmapImage);

        Log.i(TAG, "elapsed time face MLKit : " + (System.currentTimeMillis()-elapsedTime)  );
        elapsedTime = System.currentTimeMillis();

        if(detectedFace==null){
            Imgproc.putText(frame, "MLKIT Failure", new Point(100, 100),1, 2,
                    new Scalar(0, 255, 0), 2);
            return ;
        }
        // image rotation
        center.x = (int)faceMat.cols()/2;
        center.y = (int) faceMat.rows()/2;
        angle = detectedFace.getHeadEulerAngleZ();

        mapMatrix = Imgproc.getRotationMatrix2D(center, -angle, 1.0);
        // rotate
        Imgproc.warpAffine(faceMat, faceMat, mapMatrix, new Size(faceMat.cols(), faceMat.rows()));

        Log.i(TAG, "elapsed time rotation : " + (System.currentTimeMillis()-elapsedTime)  );
        elapsedTime = System.currentTimeMillis();

        // Compute embeddings
        faceRecognizer.feature(faceMat, faceEmbedding);

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
            }
        });
        savingThread.start();


    }


}
