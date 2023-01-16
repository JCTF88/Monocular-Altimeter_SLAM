import numpy as np
import math

def Euler2MatRot(fii, tee, psii):
    
    R = np.array([[math.cos(psii)*math.cos(tee), -math.sin(psii)*math.cos(fii)+math.cos(psii)*math.sin(tee)*math.sin(fii), math.sin(psii)*math.sin(fii)+math.cos(psii)*math.sin(tee)*math.cos(fii)],
                  [math.sin(psii)*math.cos(tee), math.cos(psii)*math.cos(fii)+math.sin(psii)*math.sin(tee)*math.sin(fii), -math.cos(psii)*math.sin(fii)+math.sin(psii)*math.sin(tee)*math.cos(fii)],
                  [-math.sin(tee), math.cos(tee)*math.sin(fii), math.cos(tee)*math.cos(fii)]])
    
    return R



def Quaternion2MatRot(qx, qy, qz, qw):

    R = np.array([ [qw**2+qx**2-qy**2-qz**2, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                   [2*(qx*qy+qw*qz), qw**2-qx**2+qy**2-qz**2, 2*(qy*qz-qw*qx)],
                   [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), qw**2-qx**2-qy**2+qz**2]])
    
    return R



def Euler2Quaternion(roll, pitch, yaw):
    
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
    return np.array([[qx], [qy], [qz], [qw]])



def GPS2WorldFrame(lat, lng, alt):
    
    AntGPSoffset = np.array([[0], [0], [0]])
                                   
    Ry = np.array([[-1, 0, 0],      
                   [0,  1, 0],
                   [0,  0, -1]])
   
    Rz = np.array([[0, 1, 0],      
                   [-1, 0, 0],
                   [0, 0, 1]])   
    
    scl = 100000 

    lat = lat * np.pi / 180
    lng = lng * np.pi / 180
    
    lat = round(scl*lat)/scl
    lng = round(scl*lng)/scl
    
    aa = 6378137.0
    bb = 6356752.314             
    ff = (aa-bb) / aa;                                           
    ee = math.sqrt((2-ff)*ff)
    
    Mat = np.array([ [-math.sin(lat)*math.cos(lng), -math.sin(lat)*math.sin(lng), math.cos(lat)],
                     [        -math.sin(lng),              math.cos(lng),             0     ],
                     [-math.cos(lat)*math.cos(lng),   -math.cos(lat)*math.sin(lng), -math.sin(lat)]])
    
    
    #R_M = (aa*(1-ee*ee)) / ((1-ee*ee*math.sin(lat)*math.sin(lat))**(3/2))
    R_N = aa / math.sqrt(1-ee*ee*math.sin(lat)*math.sin(lat)) 
    
    Gx = (R_N + alt)*math.cos(lat)*math.cos(lng)
    Gy = (R_N + alt)*math.cos(lat)*math.sin(lng)
    Gz = (R_N*(1-ee*ee)+alt)*math.sin(lat)
    
    Gx = round(Gx*1000) / 1000
    Gy = round(Gy*1000) / 1000
    Gz = round(Gz*1000) / 1000
    
    G = np.array([[Gx], [Gy], [Gz]])
    
    X_GPS = np.dot(Mat,G-AntGPSoffset)
    
    if X_GPS[1] > 20:
       X_GPS[1] = X_GPS[1] - 60
       
    X_GPS = np.dot(np.dot(Ry,Rz),X_GPS)   
    
    return X_GPS
    
    