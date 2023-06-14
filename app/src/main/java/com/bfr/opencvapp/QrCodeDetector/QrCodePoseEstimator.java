package com.bfr.opencvapp.QrCodeDetector;

import android.util.Log;


import com.bfr.opencvapp.utils.OpenCvUtils;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.objdetect.QRCodeDetector;

import java.util.ArrayList;
import java.util.List;

public class QrCodePoseEstimator {

    /*=================================== Constants ===============================================*/
    protected final String LOG_TAG = "QrCodePoseEstimator";
    /*================================== Attributes ===============================================*/

    protected QRCodeDetector qrCodeDetector;

    protected TermCriteria criteria;
    long startTime;

    protected Mat qrCodeTranslation, qrCodeRotation;
    protected QrCodeDescriptor qrCodeDescriptor = null;
    protected MatOfPoint2f qrCodeCorners;

    protected Mat cameraCalibrationMatrix;
    protected MatOfDouble cameraDistortionVector;

    protected int pnpSolver = Calib3d.SOLVEPNP_ITERATIVE;

    protected double angle;

    /** default constructor for QrCodeDetectorThread (PnP solver SOLVEPNP_ITERATIVE is used as default)
     *
     * @param calibrationMatrixCoeffs - coefficients of the calibration matrix as a 3*3 array matrix ([[fx, s, cx], [0, fy, cy], [0, 0, 1]])
     * @param distortionCoeffs - coefficient of the distortion vector [k1, k2, p1, p2, k3]
     */
    public QrCodePoseEstimator(double[][] calibrationMatrixCoeffs, double[] distortionCoeffs){
        qrCodeDetector = new QRCodeDetector();
        qrCodeTranslation = new Mat();
        qrCodeRotation = new Mat();
        qrCodeCorners = new MatOfPoint2f();

        setCameraProperties(calibrationMatrixCoeffs, distortionCoeffs);
        criteria = new TermCriteria(TermCriteria.COUNT+TermCriteria.EPS, 1, 2);
    }

    /** constructor of CodeDetectorThread wich specify the PnP solver to use
     *
     * @param calibrationMatrixCoeffs - coefficients of the calibration matrix as a 3*3 array matrix ([[fx, s, cx], [0, fy, cy], [0, 0, 1]])
     * @param distortionCoeffs - coefficient of the distortion vector [k1, k2, p1, p2, k3]
     * @param pnpSolver - flag for the PnP solver (check : http://www.rsqdz.net:907/%E8%AE%BA%E5%9D%9B/javadoc/org/opencv/calib3d/Calib3d.html#solvePnP-org.opencv.core.MatOfPoint3f-org.opencv.core.MatOfPoint2f-org.opencv.core.Mat-org.opencv.core.MatOfDouble-org.opencv.core.Mat-org.opencv.core.Mat-boolean-int-)
     */
    public QrCodePoseEstimator(double[][] calibrationMatrixCoeffs, double[] distortionCoeffs, int pnpSolver){
        this(calibrationMatrixCoeffs, distortionCoeffs);
        this.pnpSolver = pnpSolver;
    }

    /** setup description of qrCode
     *
     * @param sideLength - side length of the qrCode
     */
    public void setupQrCodeDescription(double sideLength){
        if(qrCodeDescriptor == null){
            qrCodeDescriptor = new QrCodeDescriptor(sideLength);
            Log.i(LOG_TAG, "QrCode description sets");
        }
        else{
            Log.i(LOG_TAG, "QrCode descriptor already set");
        }
    }

    /** detect QrCode in image and send detected code to main thread
     *
     * @param image
     * @param qrCodes - list of pre-detected QrCodes for which we want to find the pose. Its elements will be modified if pose is found
     */
    public void estimateCodesPose(Mat image, List<QrCode> qrCodes) {
        startTime = System.currentTimeMillis();

        if(qrCodes.size() > 0){
            for(QrCode code : qrCodes){
                boolean poseFound = getPose(isolateQrCode(image, code), code);
                code.poseKnown = poseFound;
            }
        }

    }

    protected boolean getPose(Mat image, QrCode code){
        boolean poseFound = false;
        Mat corners = new Mat();
        if(!image.empty() && qrCodeDetector.detect(image, corners)){
            if(qrCodeDescriptor == null){
                Log.e(LOG_TAG, "QrCode detected in image but its description is not set (should call QrCodeDetectorThread.setupQrCodeDescription()");
                return false;
            }
            OpenCvUtils.logMat(LOG_TAG, corners);
            code.setCorners(OpenCvUtils.matToMatOfPoints2f(corners));
            Calib3d.solvePnP(qrCodeDescriptor.worldModel,
                    code.corners, // don't forget to set corner before calling
                    cameraCalibrationMatrix,
                    cameraDistortionVector,
                    qrCodeRotation,
                    qrCodeTranslation,
                    false, // check if it should stay to false
                    pnpSolver);

            code.setPose(qrCodeRotation, qrCodeTranslation);
            poseFound = true;
        }
        return poseFound;
    }



    /** setup the camera properties (caibration matrix / distortion vector)
     *
     * @param calibrationMatrixCoeffs - coefficients of the calibration matrix as a 3*3 array matrix ([[fx, s, cx], [0, fy, cy], [0, 0, 1]])
     * @param distortionCoeffs - coefficient of the distortion vector [k1, k2, p1, p2, k3]
     */
    protected void setCameraProperties(double[][] calibrationMatrixCoeffs, double[] distortionCoeffs){
        cameraCalibrationMatrix = new Mat(3, 3, CvType.CV_64FC1);
        cameraDistortionVector = new MatOfDouble();
        // fill calibration matrix
        for(int row=0; row<3; ++row){
            for(int column=0; column<3; ++column){
                cameraCalibrationMatrix.put(row, column, calibrationMatrixCoeffs[row][column]);
            }
        }
        // fill camera distortion vector
        cameraDistortionVector.fromArray(distortionCoeffs);
    }


    private Mat isolateQrCode(Mat img, QrCode code){
        Mat isolated = new Mat(img.size(), CvType.CV_8U);
        isolated.setTo(new Scalar(255));
        img.copyTo(isolated, code.getMask());

        return isolated;
    }

    protected MatOfPoint2f getQrCodeFrame(){

        double frameSize = qrCodeDescriptor.side / 2 ;
        MatOfPoint3f wordFrame = new MatOfPoint3f();
        MatOfPoint2f imageFrame = new MatOfPoint2f();
        List<Point3> wordFrameList = new ArrayList<Point3>();
        wordFrameList.add(new Point3(0, 0, 0));
        wordFrameList.add(new Point3(frameSize, 0, 0));
        wordFrameList.add(new Point3(0, frameSize, 0));
        wordFrameList.add(new Point3(-0, 0, frameSize));
        wordFrame.fromList(wordFrameList);

        Calib3d.projectPoints(  wordFrame,
                qrCodeRotation,
                qrCodeTranslation,
                cameraCalibrationMatrix,
                cameraDistortionVector,
                imageFrame);

        wordFrame.release();
        return imageFrame;
    }

    protected List<MatOfPoint2f> getQrCodeCube(){
        double cubeSize = qrCodeDescriptor.side;
        List<MatOfPoint2f> cube = new ArrayList<MatOfPoint2f>();
        MatOfPoint2f imageFacePoints = new MatOfPoint2f();
        MatOfPoint3f worldFacePoints = new MatOfPoint3f();
        List<Point3> worldFacesPointsList = new ArrayList<Point3>();
        worldFacesPointsList.add(new Point3(0, 0, 0));
        worldFacesPointsList.add(new Point3(cubeSize, 0, 0));
        worldFacesPointsList.add(new Point3(cubeSize, cubeSize, 0));
        worldFacesPointsList.add(new Point3(0, cubeSize, 0));
        worldFacePoints.fromList(worldFacesPointsList);
        Calib3d.projectPoints(worldFacePoints,
                                qrCodeRotation,
                                qrCodeTranslation,
                                cameraCalibrationMatrix,
                                cameraDistortionVector,
                                imageFacePoints);
        cube.add(imageFacePoints);
        worldFacesPointsList.clear();
        worldFacesPointsList.add(new Point3(0, 0, cubeSize));
        worldFacesPointsList.add(new Point3(cubeSize, 0, cubeSize));
        worldFacesPointsList.add(new Point3(cubeSize, cubeSize, cubeSize));
        worldFacesPointsList.add(new Point3(0, cubeSize, cubeSize));
        worldFacePoints.fromList(worldFacesPointsList);
        Calib3d.projectPoints(worldFacePoints,
                qrCodeRotation,
                qrCodeTranslation,
                cameraCalibrationMatrix,
                cameraDistortionVector,
                imageFacePoints);
        cube.add(imageFacePoints);

        return cube;
    }


}
