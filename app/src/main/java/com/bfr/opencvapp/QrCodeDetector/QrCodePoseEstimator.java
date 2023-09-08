package com.bfr.opencvapp.QrCodeDetector;


import android.util.Log;

import com.bfr.opencvapp.utils.Utils;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;
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
    protected MatOfPoint2f qrCodeCorners;

    public MatOfPoint3f worldModel = null;

    protected Mat cameraCalibrationMatrix;
    protected MatOfDouble cameraDistortionVector;

    protected int pnpSolver = Calib3d.SOLVEPNP_IPPE;

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


    /** compute pose of the QRCode from its corner
     *
     * @param qrCode qrCode to estimate the pose
     * @param qrSize real size of the QRCode (in mm or m) the Pose wil be given in this same unit
     */
    protected QRCodePose EstimatePose( QrCode qrCode, double qrSize ){

        return getPose(qrCode.matOfCorners, qrSize);
    }


    /** compute pose of the QRCode from its corner
     *
     * @param qrSize real size of the QRCode (in mm or m) the Pose wil be given in this same unit
     * @param corners -  Mat of corners - should be a 2xN double matrix
     */
    public QRCodePose getPose( Mat corners, double qrSize ){
        boolean poseFound = false;

        //set world model
        setWorldModel(qrSize);

        // convert to Mat of points 2f
        qrCodeCorners = Utils.matToMatOfPoints2f(corners);

        // check calibration matrix

        // check distortion vector

        // Estimate pose
        poseFound = Calib3d.solvePnP(worldModel,
                qrCodeCorners,
                cameraCalibrationMatrix,
                cameraDistortionVector,
                qrCodeRotation,
                qrCodeTranslation,
                false, // check if it should stay to false
                pnpSolver);

        if (poseFound)
            return new QRCodePose(qrCodeRotation, qrCodeTranslation);
        else
            return new QRCodePose();
    }

    /** get the translation vector of the QRCode
     *
     * @param qrCode qrCode to estimate the pose
     * @param qrSize real size of the QRCode (in mm or m) the Pose wil be given in this same unit
     */
    public Double[] getTranslation(QrCode qrCode, double qrSize ){

        Double [] translation = new Double[]{0.0,0.0,0.0};
        QRCodePose mPose= getPose(qrCode.matOfCorners, qrSize);

        for (int i =0; i<mPose.qrCodeTranslation.size(0);i++)
        {
            translation[i] = qrCodeTranslation.get(i, 0)[0];
        }

        return translation;
    }

    /** get the rotation angles of the QRCode
     *
     * @param qrCode qrCode to estimate the pose
     * @param qrSize real size of the QRCode (in mm or m) the Pose wil be given in this same unit
     */
    public Double[] getRotation(QrCode qrCode, double qrSize ){

        Double [] rotation = new Double[]{0.0,0.0,0.0};

        QRCodePose mPose= getPose(qrCode.matOfCorners, qrSize);

        Mat rotationMatrix = new Mat();

        Log.d("QRCode", "mPoseRotation " + mPose.qrCodeRotation.get(0, 0)[0]
        + " " +  mPose.qrCodeRotation.get(1, 0)[0]
        + " " +  mPose.qrCodeRotation.get(2, 0)[0]
        );

        Calib3d.Rodrigues(mPose.qrCodeRotation, rotationMatrix);
        Log.d("QRCode", "rotationMatrix " + rotationMatrix.get(0, 0)[0]
                + " " +  rotationMatrix.get(1, 0)[0]
                + " " +  rotationMatrix.get(2, 0)[0]
        );
        for (int i =0; i<rotationMatrix.size(0);i++)
        {
            rotation[i] = (-180/3.14)*Math.asin(rotationMatrix.get(i, 0)[0]);
        }

        return rotation;
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

    /** setup real world model of the QRCode, with the origin in the center of the QRCode
     *
     * @param qrCodeSide real size of the QRCode (in mm or m) the Pose wil be given in this same unit
     */
    protected void setWorldModel(double qrCodeSide){
        worldModel = new MatOfPoint3f();
        List<Point3> cornerDescription = new ArrayList<Point3>();
        cornerDescription.add(new Point3(-qrCodeSide/2.0, qrCodeSide / 2, 0.0));
        cornerDescription.add(new Point3(qrCodeSide/2.0, qrCodeSide / 2, 0.0));
        cornerDescription.add(new Point3(qrCodeSide/2.0, -qrCodeSide / 2, 0.0));
        cornerDescription.add(new Point3(-qrCodeSide/2.0, -qrCodeSide / 2, 0.0));
        worldModel.fromList(cornerDescription);
    }

}
