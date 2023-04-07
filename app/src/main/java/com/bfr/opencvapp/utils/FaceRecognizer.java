package com.bfr.opencvapp.utils;

import android.graphics.Bitmap;
import android.util.Log;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.facemesh.FaceMesh;
import com.google.mlkit.vision.facemesh.FaceMeshDetection;
import com.google.mlkit.vision.facemesh.FaceMeshDetector;
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions;
import com.google.mlkit.vision.facemesh.FaceMeshPoint;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;

public class FaceRecognizer {

    public interface IFaceRecogRsp {
        void onSuccess(String success);
        void onFailed(String error);
    }

    private final String TAG = "FaceRecognizerSfaceImpl";

    //
    private final float MARGIN_FACTOR = 0.05f;
    private final int BORDER_MARGIN = 10;
    // default file for storing identities
    private final String STORAGE_FILE = "/sdcard/identities.bin";
    //where to find the models
    private final String DIR = "/sdcard/Android/data/com.bfr.opencvapp/files/nnmodels/";
    private final String MODEL_NAME = "/face_recognition_sface_2021dec.onnx";
//    private final String MODEL_NAME = "face_recognition_sface_2021dec-act_int8-wt_int8-quantized.onnx";

    // Face recognition
    private FaceRecognizerSF faceRecognizer;
    Mat faceEmbedding;
    Mat faceMat, rotatedFace, croppedFace;
    Rect faceROI, adjustedROI;

    // MLKit face detector
    Bitmap bitmapImage;
    private MLKitFaceDetector mlKitFaceDetector = new MLKitFaceDetector();

    // Index for face landmarks
     /*
     https://developers.google.com/static/ml-kit/vision/face-mesh-detection/images/uv_unwrap_full.png
                454: near the left ear
                234: near the right ear
                280: left cheek
                50: right cheek
                1: nose tip
                152 : chin
                10 : top forehead
                 */
    int LEFT_EAR = 454;
    int LEFT_CHEEK = 352;
    int RIGHT_EAR = 234;
    int RIGHT_CHEEK = 123;
    int NOSE_TIP = 1;
    int CHIN = 175;
    int UNDER_CHIN = 152;
    int FOREHEAD = 10;

    // Face detector for robust crop
    FaceDetectorYN ynFaceDetector;
    Mat ynFaces;
    private final String YNMODEL_NAME = "face_detection_yunet_2022mar.onnx";
    //    private final String YNMODEL_NAME = "face_recognition_sface_2021dec-act_int8-wt_int8-quantized.onnx";
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



    class Point2Dbfr
    {
        double x,y;

        public Point2Dbfr(double x, double y)
        {
            this.x = x;
            this.y = y;
        }

        // Method used to display X and Y coordinates
        // of a point

    }

    Point2Dbfr lineLineIntersection(Point2Dbfr A, Point2Dbfr B, Point2Dbfr C, Point2Dbfr D)
        {
            // Line AB represented as a1x + b1y = c1
            double a1 = B.y - A.y;
            double b1 = A.x - B.x;
            double c1 = a1*(A.x) + b1*(A.y);

            // Line CD represented as a2x + b2y = c2
            double a2 = D.y - C.y;
            double b2 = C.x - D.x;
            double c2 = a2*(C.x)+ b2*(C.y);

            double determinant = a1*b2 - a2*b1;

            if (determinant == 0)
            {
                // The lines are parallel. This is simplified
                // by returning a pair of FLT_MAX
                return new Point2Dbfr(Double.MAX_VALUE, Double.MAX_VALUE);
            }
            else
            {
                double x = (b2*c1 - b1*c2)/determinant;
                double y = (a1*c2 - a2*c1)/determinant;
                return new Point2Dbfr(x, y);
            }
        }

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
        int newLeft=0, newRight=0, newTop=0, newBottom =0;
        try{

            // taking some margins to make the face detection easier
            left   = (int)(leftIn * frame.cols())-BORDER_MARGIN+5;
            top    = (int)(topIn * frame.rows()) -BORDER_MARGIN+5;
            right  = (int)(rightIn * frame.cols())+BORDER_MARGIN-5;
            bottom = (int)(bottomIn* frame.rows())+BORDER_MARGIN-5;

            int leftFaceMat = (int)(leftIn * frame.cols())-BORDER_MARGIN+5;
            int topFaceMat = (int)(topIn * frame.rows()) -BORDER_MARGIN+5;
            int rightFaceMat = (int)(rightIn * frame.cols())+BORDER_MARGIN-5;
            int bottomFaceMat = (int)(bottomIn* frame.rows())+BORDER_MARGIN-5;

            elapsedTime = System.currentTimeMillis();
            //crop image around face
            faceROI= new Rect(
                    // alternative to crop more: Math.max(left, (int)(left- MARGIN_FACTOR *(right-left)) ),
                    Math.max(0,left ) ,
                    Math.max(0,top),
                    //alternative to crop more: (int)(right-left)+(int)(MARGIN_FACTOR *(right-left)),
                    Math.min(frame.cols()-left,right-left),
                    Math.min(frame.rows()-top, bottom-top) );

            faceMat = frame.submat(faceROI);

            if(withPreprocess) {
                /*** face orientation*/
                //convert to bitmap
                bitmapImage = Bitmap.createBitmap(faceMat.cols(), faceMat.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(faceMat, bitmapImage);

                //FaceMesh
                FaceMeshDetector detector;
                // detect-classify face
                FaceMeshDetectorOptions.Builder optionsBuilder = new FaceMeshDetectorOptions.Builder();

                optionsBuilder.setUseCase(FaceMeshDetectorOptions.FACE_MESH);

                detector = FaceMeshDetection.getClient(optionsBuilder.build());

                InputImage inputImage = InputImage.fromBitmap(bitmapImage, 0);

                Task<List<FaceMesh>> mFaceMesh = detector.process(inputImage);

                try {
                    Tasks.await(mFaceMesh);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                Log.i(TAG, "elapsed time face MLKit : " + (System.currentTimeMillis() - elapsedTime));
                elapsedTime = System.currentTimeMillis();

                // Gets all points
                List<FaceMeshPoint> faceMeshpoints = mFaceMesh.getResult().get(0).getAllPoints();

                if (faceMeshpoints.size()<=0 ) {
                    Log.w(TAG, "error preprocessing image with MLKit");
                    return null;
                }

                /*** rotate ***/
                // Compute face orientation
                angle = - Math.toDegrees(
                        Math.atan2(faceMeshpoints.get(LEFT_EAR).getPosition().getY() - faceMeshpoints.get(RIGHT_EAR).getPosition().getY(),
                                faceMeshpoints.get(LEFT_EAR).getPosition().getX() - faceMeshpoints.get(RIGHT_EAR).getPosition().getX())
                               );

                Point2Dbfr pointLeftEar = new Point2Dbfr(faceMeshpoints.get(LEFT_EAR).getPosition().getX(), faceMeshpoints.get(LEFT_EAR).getPosition().getY());
                Point2Dbfr pointRightEar = new Point2Dbfr(faceMeshpoints.get(RIGHT_EAR).getPosition().getX(), faceMeshpoints.get(RIGHT_EAR).getPosition().getY());
                Point2Dbfr pointForeHead = new Point2Dbfr(faceMeshpoints.get(FOREHEAD).getPosition().getX(), faceMeshpoints.get(FOREHEAD).getPosition().getY());
                Point2Dbfr pointForeChin = new Point2Dbfr(faceMeshpoints.get(UNDER_CHIN).getPosition().getX(), faceMeshpoints.get(UNDER_CHIN).getPosition().getY());

                int centerInFaceMatX = (int)lineLineIntersection(pointLeftEar, pointRightEar, pointForeHead, pointForeChin).x;
                int centerInFaceMatY = (int)lineLineIntersection(pointLeftEar, pointRightEar, pointForeHead, pointForeChin).y;
                center.x = leftFaceMat+ lineLineIntersection(pointLeftEar, pointRightEar, pointForeHead, pointForeChin).x;
                center.y = topFaceMat + lineLineIntersection(pointLeftEar, pointRightEar, pointForeHead, pointForeChin).y;

                //Imgproc.circle(frame, new Point(center.x, center.y), 3, new Scalar(255, 50, 0), 3 );

                mapMatrix = Imgproc.getRotationMatrix2D(center, -angle, 1.0);

                Mat frame_cpy = frame.clone();

                Imgproc.warpAffine(frame_cpy, frame_cpy, mapMatrix, new Size(frame.cols(), frame.rows()));

                /***  Crop**/

                //Left limit
                //Get the point the most on the left
                int minLeft = (int) (Math.min(Math.min(faceMeshpoints.get(RIGHT_CHEEK).getPosition().getX(),
                                faceMeshpoints.get(RIGHT_EAR).getPosition().getX()),
                        faceMeshpoints.get(NOSE_TIP).getPosition().getX()));

                // init
                int distX=0;
                int distY=0;
                int posEdgeLX = 0;
                int posEdgeLY = 0;
                if (minLeft == (int)faceMeshpoints.get(NOSE_TIP).getPosition().getX()){
                    posEdgeLX = (int)faceMeshpoints.get(NOSE_TIP).getPosition().getX();
                    posEdgeLY = (int)faceMeshpoints.get(NOSE_TIP).getPosition().getY();
                }
                else if (minLeft == (int) faceMeshpoints.get(RIGHT_CHEEK).getPosition().getX()){
                    posEdgeLX = (int)faceMeshpoints.get(RIGHT_CHEEK).getPosition().getX();
                    posEdgeLY = (int)faceMeshpoints.get(RIGHT_CHEEK).getPosition().getY();
                }
                else if (minLeft == (int) faceMeshpoints.get(RIGHT_EAR).getPosition().getX()){
                    posEdgeLX = (int)faceMeshpoints.get(RIGHT_EAR).getPosition().getX();
                    posEdgeLY = (int)faceMeshpoints.get(RIGHT_EAR).getPosition().getY();
                }
                distX = posEdgeLX-centerInFaceMatX;
                distY = posEdgeLY-centerInFaceMatY;
                //Eucl dist
                newLeft =  (int) center.x -(int)Math.sqrt(distX*distX +distY*distY);


                //Right limit
                //Get the point the most on the right
                int maxRight = (int) (Math.max(Math.max(faceMeshpoints.get(LEFT_CHEEK).getPosition().getX(),
                                faceMeshpoints.get(LEFT_EAR).getPosition().getX()),
                        faceMeshpoints.get(NOSE_TIP).getPosition().getX()));
                int posEdgeRX = 0;
                int posEdgeRY = 0;
                if (maxRight == (int)faceMeshpoints.get(NOSE_TIP).getPosition().getX()){
                    posEdgeRX = (int)faceMeshpoints.get(NOSE_TIP).getPosition().getX();
                    posEdgeRY = (int)faceMeshpoints.get(NOSE_TIP).getPosition().getY();
                }
                else if (maxRight == (int)faceMeshpoints.get(LEFT_CHEEK).getPosition().getX()){
                    posEdgeRX = (int)faceMeshpoints.get(LEFT_CHEEK).getPosition().getX();
                    posEdgeRY = (int)faceMeshpoints.get(LEFT_CHEEK).getPosition().getY();
                }
                else if (maxRight == (int)faceMeshpoints.get(LEFT_EAR).getPosition().getX()){
                    posEdgeRX = (int)faceMeshpoints.get(LEFT_EAR).getPosition().getX();
                    posEdgeRY = (int)faceMeshpoints.get(LEFT_EAR).getPosition().getY();
                }
                distX = posEdgeRX-centerInFaceMatX;
                distY = posEdgeRY-centerInFaceMatY;
                //Eucl dist
                newRight = (int) center.x +(int)Math.sqrt(distX*distX +distY*distY);


                //Top limit
                int posEdgeTX = (int)faceMeshpoints.get(FOREHEAD).getPosition().getX();
                int posEdgeTY = (int)faceMeshpoints.get(FOREHEAD).getPosition().getY();
                distX = posEdgeTX-centerInFaceMatX;
                distY = posEdgeTY-centerInFaceMatY;
                //Eucl dist
                newTop = (int) center.y -(int)Math.sqrt(distX*distX +distY*distY);

                // Bottom Limit
                int posEdgeBX = (int)faceMeshpoints.get(UNDER_CHIN).getPosition().getX();
                int posEdgeBY = (int)faceMeshpoints.get(UNDER_CHIN).getPosition().getY();
                distX = posEdgeBX-centerInFaceMatX;
                distY = posEdgeBY-centerInFaceMatY;
                //Eucl dist
                newBottom = (int) center.y +(int)Math.sqrt(distX*distX +distY*distY);


                //Imgproc.rectangle(frame_cpy, new Point(newLeft, newTop),  new Point(newRight, newBottom), new Scalar(0, 0, 255), 2);
                //Log.w("coucou", "cropping again : " + center.x + " " + distX + " " + distY +" "+newRight + " " + newLeft +" ");

                adjustedROI= new Rect(
                        Math.max(0,newLeft),
                        Math.max(0,newTop),
                        Math.min(frame_cpy.cols()-newLeft,newRight-newLeft),
                        Math.min(frame_cpy.rows()-newTop, newBottom-newTop) );

                //Crop around face
                rotatedFace = frame_cpy.submat(adjustedROI);

            }
            else // no preprocess
                rotatedFace = faceMat;

            return rotatedFace;

        } catch (Exception e)
        {
            Log.e(TAG, "error during crop: " + newLeft + " " + newRight + " " + newTop + " " + newBottom + " "
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

        // Check if face not touching the margins
        left   = (int)(leftIn * frame.cols());
        top    = (int)(topIn * frame.rows());
        right  = (int)(rightIn * frame.cols());
        bottom = (int)(bottomIn* frame.rows());

        //If face touches the margin, skip -> we need a fully visible face for recognition
        if(left<BORDER_MARGIN || top <BORDER_MARGIN || right> frame.cols()-BORDER_MARGIN || bottom > frame.rows()-BORDER_MARGIN)
        {
            // take next detected face
            Log.w(TAG, "Failed saving face: the face must not touch the borders for a correct recognition");
            return null;
        }
        elapsedTime = System.currentTimeMillis();

        croppedFace = cropAndAlign(frame, leftIn, rightIn, topIn, bottomIn);
        if (croppedFace!=null)
            faceRecognizer.feature(
                croppedFace,
                faceEmbedding);
        else
            return null;

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

        // Check if face not touching the margins
        left   = (int)(leftIn * frame.cols());
        top    = (int)(topIn * frame.rows());
        right  = (int)(rightIn * frame.cols());
        bottom = (int)(bottomIn* frame.rows());

        //If face touches the margin, skip -> we need a fully visible face for recognition
        if(left<BORDER_MARGIN || top <BORDER_MARGIN || right> frame.cols()-BORDER_MARGIN || bottom > frame.rows()-BORDER_MARGIN)
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
