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
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;

import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;

public class FaceRecognizer {

    public interface IFaceRecogRsp {
        void onSuccess(String success);
        void onFailed(String error);
    }

    private final String TAG = "FaceRecognizerSfaceImpl";

    //
    private final float MARGIN_FACTOR = 0.05f;
    // default file for storing identities
    private final String STORAGE_FILE = "/sdcard/identities.bin";
    //where to find the models
    private final String DIR = "/sdcard/Android/data/com.bfr.opencvapp/files/nnmodels/";
//    private final String MODEL_NAME = "/face_recognition_sface_2021dec.onnx";
    private final String MODEL_NAME = "face_recognition_sface_2021dec-act_int8-wt_int8-quantized.onnx";
    private final String YNMODEL_NAME = "face_detection_yunet_2022mar.onnx";
//    private final String YNMODEL_NAME = "face_recognition_sface_2021dec-act_int8-wt_int8-quantized.onnx";
    // Face recognition
    private FaceRecognizerSF faceRecognizer;
    Mat faceEmbedding;
    Mat faceMat, rotatedFace, croppedFace;
    Rect faceROI, adjustedROI;

    // MLKit face detector
    Bitmap bitmapImage;
    private MLKitFaceDetector mlKitFaceDetector = new MLKitFaceDetector();

    // Face detector for robust crop
    FaceDetectorYN ynFaceDetector;
    Mat ynFaces;
    public Net detector;
    int leftInFaceMat, rightInFaceMat, topInFaceMat, bottomInFaceMat;
    int leftYN, rightYN, topYN, bottomYN;

    // image rotation
    int left, right, top, bottom;
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
    boolean saving = false;


    Mat toSave;
    Thread tbitmpa = new Thread(new Runnable() {
        @Override
        public void run() {
            try (FileOutputStream out = new FileOutputStream("/sdcard/faceReco_"+ currentDateandTime+"_bitmap"+".bmp")) {
                bitmapImage.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
                // PNG is a lossless format, the compression factor (100) is ignored
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    });

    Thread save1 = new Thread(new Runnable() {
        @Override
        public void run() {
            Imgcodecs.imwrite("/sdcard/faceReco_"+ currentDateandTime+"_1stfacemat"+".jpg", toSave);
        }
    });

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

        croppedFace = new Mat();

        // Face detection for robust crop
        ynFaceDetector = FaceDetectorYN.create(DIR+YNMODEL_NAME, "", new Size(640, 480), 0.7f);
        ynFaces = new Mat();

        //legacy face detector
        String modelFace = DIR + "/opencv_face_detector.pbtxt";
        String weightsFace = DIR + "/opencv_face_detector_uint8.pb";
        detector = Dnn.readNetFromTensorflow(weightsFace, modelFace);


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
        try{

            left   = (int)(leftIn * frame.cols())-10;
            top    = (int)(topIn * frame.rows()) -10;
            right  = (int)(rightIn * frame.cols())+10;
            bottom = (int)(bottomIn* frame.rows())+10;


            elapsedTime = System.currentTimeMillis();
            //crop image around face
            faceROI= new Rect(
                    // alternative to crop more: Math.max(left, (int)(left- MARGIN_FACTOR *(right-left)) ),
                    Math.max(0,left ) ,
                    Math.max(0,top),
                    //alternative to crop more: (int)(right-left)+(int)(MARGIN_FACTOR *(right-left)),
                    Math.min(frame.cols()-left,right-left),
                    Math.min(frame.rows()-top, bottom-top) );

            if(saving)
                Log.i(TAG, "1st crop : " + left + " " + right + " " + top + " "  + bottom);

            faceMat = frame.submat(faceROI);

            //TODEBUG only
            toSave = new Mat();
            Imgproc.cvtColor(faceMat, toSave, Imgproc.COLOR_RGB2BGR);
            if(saving)
                Imgcodecs.imwrite("/sdcard/faceReco_"+ currentDateandTime+"_01_1stfacemat"+".jpg", toSave);

            if(withPreprocess) {
                /*** face orientation*/
                //convert to bitmap
                bitmapImage = Bitmap.createBitmap(faceMat.cols(), faceMat.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(faceMat, bitmapImage);
//                if(saving)
//                    tbitmpa.start();

                // detect-classify face
                Face detectedFace = mlKitFaceDetector.detectSingleFaceFromBitmap(bitmapImage);

                if(saving)
                    Log.i(TAG, "elapsed time face MLKit : " + (System.currentTimeMillis() - elapsedTime));
                elapsedTime = System.currentTimeMillis();

                if (detectedFace == null) {
                    Log.i(TAG, "error processing image with MLKit");
                    return null;
                }

                if(saving)
                    Log.i(TAG, "MLKit crop : " + detectedFace.getBoundingBox().left + " "
                        + detectedFace.getBoundingBox().right + " "
                        + detectedFace.getBoundingBox().top + " "
                        + detectedFace.getBoundingBox().bottom);

                /****************************** Legacy face detection */

                if(true)
                {
                ynFaceDetector.setInputSize(new Size(faceMat.cols(), faceMat.rows()));
                ynFaceDetector.detect(faceMat, ynFaces);
//
                leftYN   = (int)( ynFaces.get(0, 0)[0] );
                topYN    = (int)( ynFaces.get(0, 1)[0] );
                rightYN  = leftYN+(int)( ynFaces.get(0, 2)[0] );
                bottomYN = topYN+(int)( ynFaces.get(0, 3)[0]);

                Log.i(TAG, "YN crop : " + leftYN + " " + rightYN + " " + topYN + " "  + bottomYN);
                Log.i(TAG, "YN crop : " + ynFaces.get(0, 0)[0] + " "
                        + ynFaces.get(0, 1)[0] + " "
                        + ynFaces.get(0, 2)[0] + " "
                        + ynFaces.get(0, 3)[0]);


//                    Size frameResize = new Size(300,300);
//                    Scalar frameMeanFace = new Scalar(104, 117, 123);
//
//                    Mat blob = Dnn.blobFromImage(faceMat, 1.0,
//                            frameResize,
//                            frameMeanFace, /*swapRB*/true, /*crop*/false);
//                    detector.setInput(blob);
//                    // Detection
//                    Log.i(TAG, "Legacy detection");
//                    Mat detections = detector.forward();
//                    int frameCols = frame.cols();
//                    int frameRows = frame.rows();
//                    detections = detections.reshape(1, (int) detections.total() / 7);
//
//                    //init
//                    double confidence=0;
//
//                    // for each detection
//                    for (int i = 0; i < detections.rows(); ++i) {
//                        confidence = detections.get(i, 2)[0];
//                        Log.i(TAG, "Found face " + confidence);
//                        // if confidence level > Threshold
//                        if (confidence > 0.8) {
//                            // save detection
//
//
//                            leftInFaceMat   =  (int)((detections.get(i, 3)[0]) * faceMat.cols());
//                            topInFaceMat    =  (int)( (detections.get(i, 4)[0]) * faceMat.rows());
//                            rightInFaceMat  = (int)( (detections.get(i, 5)[0]) * faceMat.cols());
//                            bottomInFaceMat =  (int)( (detections.get(i, 6)[0]) * faceMat.rows());
//
//                            Log.i(TAG, "Found face OK at "
//                                    + (detections.get(i, 3)[0])+ "x"+ leftInFaceMat + " "
//                                    + (detections.get(i, 5)[0])+ "x"+ rightInFaceMat + " "
//                                    + (detections.get(i, 4)[0])+ "x"+ topInFaceMat  + " "
//                                    + (detections.get(i, 6)[0])+ "x"+ bottomInFaceMat + " in "
//                                    + faceMat.cols() + " "
//                                    + faceMat.rows()+ " " );
//                            break;
//                            //classId = (int) detections.get(i, 1)[0];
//                        }
//                    }
//
//                    Imgproc.rectangle(faceMat, new Point(leftInFaceMat, topInFaceMat),
//                            new Point(rightInFaceMat, bottomInFaceMat),
//                            new Scalar(0, 255, 0), 2);
                }

//                Imgproc.rectangle(faceMat, new Point(leftYN, topYN),
//                        new Point(rightYN, bottomYN),
//                        new Scalar(0, 0, 255), 2);



                /******************************* end legacy detection */
                //TODEBUG only
                // Draw rectangle around detected face.
//                Imgproc.rectangle(faceMat, new Point(detectedFace.getBoundingBox().left, detectedFace.getBoundingBox().top),
//                        new Point(detectedFace.getBoundingBox().right, detectedFace.getBoundingBox().bottom),
//                        new Scalar(255, 0, 0), 2);

                toSave = new Mat();
                Imgproc.cvtColor(faceMat, toSave, Imgproc.COLOR_RGB2BGR);
                if(saving)
                    Imgcodecs.imwrite("/sdcard/faceReco_"+ currentDateandTime+"_02_facematwithrecog"+".jpg", toSave);


                //check abnormal behaviour from MLKit detector (should not detect out of the image)
//                if(detectedFace.getBoundingBox().right >right-left ||
//                        detectedFace.getBoundingBox().bottom >bottom-top)
//                    return null;
//
//                //crop on the actual detected face
//                right = left + detectedFace.getBoundingBox().right;
//                left = left + detectedFace.getBoundingBox().left;
//                //empiric added margin to compensate tendencies of the face detector
//                bottom = top + detectedFace.getBoundingBox().bottom + (int)(MARGIN_FACTOR *(bottom-top));
//                top = top + + detectedFace.getBoundingBox().top;//
//
//                right = left + rightInFaceMat;
                right = left + rightYN;
//                left = left + leftInFaceMat;
                left = left + leftYN;
                //empiric added margin to compensate tendencies of the face detector
//                bottom = top + bottomInFaceMat;
                bottom = top + bottomYN;
//                top = top + topInFaceMat;
                top = top + topYN;

                if (saving)
                    Log.i(TAG, "2nd crop : " + left + " " + right + " " + top + " "  + bottom);
                adjustedROI= new Rect(
                        Math.max(0,left ) ,
                        Math.max(0,top),
                        //alternative to crop more: (int)(right-left)+(int)(MARGIN_FACTOR *(right-left)),
                        Math.min(frame.cols()-left,right-left),
                        Math.min(frame.rows()-top, bottom-top) );

                rotatedFace = frame.submat(adjustedROI).clone();

                //TODEBUG only

                if(saving) {
                    Mat toSave2 = new Mat();
                    Imgproc.cvtColor(rotatedFace, toSave2, Imgproc.COLOR_RGB2BGR);
                    Imgcodecs.imwrite("/sdcard/faceReco_" + currentDateandTime + "_03_rotatedFace" + ".jpg", toSave2);
                }
                // image rotation
                center.x = rotatedFace.cols() / 2;
                center.y = rotatedFace.rows() / 2;
                angle = detectedFace.getHeadEulerAngleZ();
                mapMatrix = Imgproc.getRotationMatrix2D(center, -angle, 1.0);
                // rotate
                Imgproc.warpAffine(rotatedFace, rotatedFace, mapMatrix, new Size(rotatedFace.cols(), rotatedFace.rows()));
                if(saving)
                    Log.i(TAG, "elapsed time rotating : " + (System.currentTimeMillis() - elapsedTime));

                //TODEBUG only
                Mat toSave3 = new Mat();
                Imgproc.cvtColor(rotatedFace, toSave3, Imgproc.COLOR_RGB2BGR);
                if(saving)
                    Imgcodecs.imwrite("/sdcard/faceReco_"+ currentDateandTime+"_04__2ndrotatedFace"+".jpg", toSave3);
                else
                    Imgcodecs.imwrite("/sdcard/faceReco_"+ currentDateandTime+"_ongoing"+".jpg", toSave3);
            }
            return rotatedFace;

        } catch (Exception e)
        {
            Log.e(TAG, "error during crop: " + left + " " + right + " " + top + " " + bottom + " "
                    + "\n" + frame.cols() + " " + frame.rows() +"\n"
                    + Log.getStackTraceString(e));
            return null;
        }
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

        saving = false;
        elapsedTime = System.currentTimeMillis();

        croppedFace = cropAndAlign(frame, leftIn, rightIn, topIn, bottomIn);
        if (croppedFace!=null)
            faceRecognizer.feature(
                croppedFace,
                faceEmbedding);
        else
            return null;
        if(saving)
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
    public void saveFace(Mat frame, float leftIn, float rightIn, float topIn, float bottomIn, String name, IFaceRecogRsp response)
    {
        saveFace(frame, leftIn, rightIn, topIn, bottomIn, name, STORAGE_FILE, response);
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
    public void saveFace(Mat frame, float leftIn, float rightIn, float topIn, float bottomIn, String name, String storingFile, IFaceRecogRsp response)
    {

        saving = true;

        left   = (int)(leftIn * frame.cols());
        top    = (int)(topIn * frame.rows());
        right  = (int)(rightIn * frame.cols());
        bottom = (int)(bottomIn* frame.rows());

        //If face touches the margin, skip -> we need a fully visible face for recognition
        if(left<10 || top <10 || right> frame.cols()-10 || bottom > frame.rows()-10)
        {
            // take next detected face
            response.onFailed("Failed saving face: the face must not touch the borders for a correct recognition");
            return;
        }

        croppedFace = cropAndAlign(frame, leftIn, rightIn, topIn, bottomIn);
        if (croppedFace!=null)
            faceRecognizer.feature(
                    croppedFace,
                    faceEmbedding);
        else
        {
            response.onFailed("Failed saving face: Failing to crop");
            return;
        }

        if(saving)
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
                try {
                    idDatabase.saveToStorage(storingFile);
                    response.onSuccess("Identity successfully saved in file " + storingFile);
                }
                catch (IOException e)
                {
                    response.onFailed("Error saving file with identities");
                }
                catch (Exception e)
                {
                    response.onFailed("Error saving file with identities: " + e.toString());
                }

                //TODEBUG only
                Mat toSave = new Mat();
                Imgproc.cvtColor(rotatedFace, toSave, Imgproc.COLOR_RGB2BGR);
                Imgcodecs.imwrite("/sdcard/faceReco_"+ name+"_"+currentDateandTime+".jpg", toSave);
            }
        });
        savingThread.start();

    } //end save face

    /**
     * Load the saved identities
     * @return null
     */
    public void loadFaces(IFaceRecogRsp response)
    {
        loadFaces(STORAGE_FILE, response);
    }

    /**
     * Load a custom file which contains saved identities
     * @param fileName file
     * @return null
     */
    public void loadFaces(String fileName, IFaceRecogRsp response)
    {
        try{
            idDatabase.identities.clear();
            idDatabase.loadFromStorage(fileName);
            response.onSuccess("Identities loaded with success");
        }
        catch (Exception e)
        {
            response.onSuccess("Failed loading identities: " + e);
        }

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
    public void removeSavedIdentity(int idx, IFaceRecogRsp response)
    {
        removeSavedIdentity(idx, STORAGE_FILE, response);
    } // end remove identity


    /**
     * remove i-th save identity and save to a custom file an the device.
     * WARNING: this operation is not reversible
     * @param idx number of first k candidates
     * @param storageFile custom file to save on device
     * @return nothing
     */
    public void removeSavedIdentity(int idx, String storageFile, IFaceRecogRsp response)
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

                        try {
                            idDatabase.saveToStorage(storageFile);
                            response.onSuccess("Identity successfully removed in file");
                        }
                        catch (IOException e)
                        {
                            response.onFailed("Error saving file with identities");
                        }
                        catch (Exception e)
                        {
                            response.onFailed("Error saving file with identities: " + e.toString());
                        }
                    }
                });
                savingThread.start();
            }
            else
                Log.w(TAG, "Can't remove the identity : you should always leave at least one identity in the database");

        } catch (Exception e)
        {
            Log.i(TAG, "Error removing saved identity " + e);
            response.onFailed("Error removing identities: " + e.toString());
        }

    } // end remove identity

}
