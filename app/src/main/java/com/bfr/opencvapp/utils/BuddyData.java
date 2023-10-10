package com.bfr.opencvapp.utils;

import android.os.RemoteException;


import com.bfr.buddy.usb.shared.BodySensorData;
import com.bfr.buddy.usb.shared.HeadSensorData;
import com.bfr.buddy.usb.shared.MotorHeadData;
import com.bfr.buddy.usb.shared.MotorMotionData;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

    // Class for callbac
public class BuddyData  {

    // Enable / disable
    public String yesStatus, noStatus, leftWheelStatus, rightWheelStatus;
    // position
    public int yesPos, noPos;

//
//        @Override
//        // called when the datas from the motor are received
//        public void ReceiveMotorMotionData(MotorMotionData msg) throws RemoteException {
//            leftWheelStatus = msg.leftWheelMode;
//            rightWheelStatus = msg.rightWheelMode;
//        }
//
//        @Override
//        // called when the datas from the head motor are received
//        public void ReceiveMotorHeadData(MotorHeadData msg) throws RemoteException {
//            yesStatus = msg.yesMode;
//            noStatus = msg.noMode;
//            yesPos = msg.yesPosition;
//            noPos = msg.noPosition;
//        } // end receiveMotorHead Data
//
//        @Override
//        // called when the datas from the head sensor motor are received
//        public void ReceiveHeadSensorData(HeadSensorData msg) throws RemoteException {
//        }
//
//        @Override
//        // called when the datas from the body sensor motor are received
//        public void ReceiveBodySensorData(BodySensorData data) throws RemoteException {
//        } // receiveBodySensorData


} // end BuddyData class

