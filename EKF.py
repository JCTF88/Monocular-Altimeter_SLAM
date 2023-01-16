import numpy as np

from Jacobians import JacCamRespectRobot, JacCamRespectLandmark

          
def Prediction(Q_robot, q_landmarks, P, num_landmarks, A_robot):   
    
    A1 = np.hstack([A_robot,np.zeros((12,3*num_landmarks))])
    A2 = np.hstack([np.zeros((3*num_landmarks,12)),np.identity(3*num_landmarks)])
    A = np.vstack([A1,A2])
        
    Q1 = np.hstack([Q_robot,np.zeros((12,3*num_landmarks))])
    Q2 = np.hstack([np.zeros((3*num_landmarks,12)),np.identity(3*num_landmarks)*q_landmarks])
    Q = np.vstack([Q1,Q2]) 
        
    P = np.dot(np.dot(A,P),A.T) + Q
        
    return P

def EKFcorrectionCamera(x_robot, x_landmark, R0, R, cam_parameters, XE, Z, P, i, size_map, M):
    
    Jac_Img_Robot  = JacCamRespectRobot(x_robot, x_landmark, R0, cam_parameters) 
    Jac_Img_Lan  = JacCamRespectLandmark(x_robot, x_landmark, R0, cam_parameters) 
    
    C = np.hstack([Jac_Img_Robot[0:2,0:3],np.zeros((2,9)),np.zeros((2,3*(i-1))),Jac_Img_Lan,np.zeros((2,3*(size_map-i)))]) 

    K = np.dot(np.dot(P,C.T),np.linalg.inv(np.dot(np.dot(C,P),C.T) + R))

    XE = XE + np.dot(K,(Z-M))
    
    P = np.dot(np.identity(12+(size_map*3)) - np.dot(K,C),P)

    return XE, P
    
def EKFcorrectionAltimeter(XE,Z,P,size_map,R):
 
    M = XE[2]

    Z = np.array([[Z]])

    C = np.hstack([np.array([[0,0,1,0,0,0,0,0,0,0,0,0]]),np.zeros((1,3*size_map))])
    
    K = np.dot(np.dot(P,C.T),np.linalg.inv(np.dot(np.dot(C,P),C.T) + R))
    
    XE = XE + np.dot(K,Z-M)
    
    P = np.dot(np.identity(12+(size_map*3)) - np.dot(K,C),P)

    return XE, P
        
        
    
    
          
        