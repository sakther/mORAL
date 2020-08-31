import pandas as pd
import matplotlib.pylab as plt
import math
import statistics
import numpy as np
import os

def apply_complementary_filter(accel_gyro):
    AG = np.array(accel_gyro)
    thetaX, thetaY, thetaZ = complementary_filter(AG[:,0], AG[:,1], AG[:,2], AG[:,3], AG[:,4], AG[:,5], AG[:,6], fq=16)

    AGO = list(np.column_stack((AG, thetaX, thetaY, thetaZ)))

    return AGO

def complementary_filter(ts, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, fq=16):

    # print(len(gyr_x))
    dt = 1.0/fq  #1/16.0; 
    M_PI =  math.pi;
    hpf = 0.90;
    lpf = 0.10;

    thetaX_acc = [0]*len(acc_x) # math.atan2(-acc_z,acc_y)*180/M_PI;
    thetaY_acc = [0]*len(acc_x) # math.atan2(acc_x,acc_z)*180/M_PI;
    thetaZ_acc = [0]*len(acc_x) # math.atan2(acc_y,acc_x)*180/M_PI;

    thetaX = [0]*len(gyr_x)
    thetaY = [0]*len(gyr_y)
    thetaZ = [0]*len(gyr_z)

    # print(len(gyr_x))
    for a in range(len(gyr_x)):
        thetaX_acc[a] = math.atan2(-acc_z[a],acc_y[a])*180/M_PI
        thetaY_acc[a] = math.atan2(acc_x[a],acc_z[a])*180/M_PI
        thetaZ_acc[a] = math.atan2(acc_y[a],acc_x[a])*180/M_PI
        
        if a == 0:
            thetaX[a] = hpf*thetaX[a]*dt + lpf*thetaX_acc[a]
            thetaY[a] = hpf*thetaY[a]*dt + lpf*thetaY_acc[a]
            thetaZ[a] = hpf*thetaZ[a]*dt + lpf*thetaZ_acc[a]
        else:
            thetaX[a] = hpf*(thetaX[a-1] + gyr_x[a]*dt) + lpf*thetaX_acc[a]
            thetaY[a] = hpf*(thetaY[a-1] + gyr_y[a]*dt) + lpf*thetaY_acc[a]
            thetaZ[a] = hpf*(thetaZ[a-1] + gyr_z[a]*dt) + lpf*thetaZ_acc[a]
            
    return thetaX, thetaY, thetaZ


def calculateRoll(Ax, Ay, Az) :
    roll = []
    for i in range(len(Ax)):
        ax = Ax[i]
        ay = Ay[i]
        az = Az[i]
        rll = 180 * math.atan2(ax, math.sqrt(ay * ay + az * az)) / math.pi
        roll.append(rll)
    return roll

def calculatePitch(Ax, Ay, Az) :
    pitch = []
    for i in range(len(Ax)):
        ax = Ax[i]
        ay = Ay[i]
        az = Az[i]
        ptch = 180 * math.atan2(-ay, -az) / math.pi
        pitch.append(ptch)
    return pitch

def calculateYaw(Ax, Ay, Az) :
    yaw = []
    for i in range(len(Ax)):
        ax = Ax[i]
        ay = Ay[i]
        az = Az[i]
        yw = 180 * math.atan2(ay, ax) / math.pi
        yaw.append(yw)
    return yaw

   