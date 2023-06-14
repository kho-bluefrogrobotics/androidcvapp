package com.bfr.opencvapp.QrCodeDetector;

import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;

import java.util.ArrayList;
import java.util.List;

public class QrCodeDescriptor {
    // Attributes
    MatOfPoint3f worldModel;
    double side;

    // Methods
    public QrCodeDescriptor(double qrCodeSide){
        side = qrCodeSide;
        worldModel = new MatOfPoint3f();
        List<Point3> cornerDescription = new ArrayList<Point3>();
        cornerDescription.add(new Point3(-side/2.0, side / 2, 0.0));
        cornerDescription.add(new Point3(side/2.0, side / 2, 0.0));
        cornerDescription.add(new Point3(side/2.0, -side / 2, 0.0));
        cornerDescription.add(new Point3(-side/2.0, -side / 2, 0.0));
        worldModel.fromList(cornerDescription);
    }
}
