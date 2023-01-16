import numpy as np

from Transformations import Euler2MatRot

def JacCamRespectRobot(x_robot, x_landmark, R0, cam_parameters):
    
     R = Euler2MatRot(x_robot[3], x_robot[4], x_robot[5])

     As = np.dot(np.dot(R0.T,R.T),(x_landmark - x_robot[0:3]))
   
     Anti_x = np.array([[0,0,0],[0,0,-1],[0,1,0]])
     Anti_y = np.array([[0,0,1],[0,0,0],[-1,0,0]])
     Anti_z = np.array([[0,-1,0],[1,0,0],[0,0,0]])
    
     V1 = np.dot(np.dot(np.dot(R0.T,R.T),Anti_z.T),x_landmark - x_robot[0:3])
     V2 = np.dot(np.dot(np.dot(R0.T,R.T),Anti_y.T),x_landmark - x_robot[0:3])
     V3 = np.dot(np.dot(np.dot(R0.T,R.T),Anti_x.T),x_landmark - x_robot[0:3])
    
     VT = np.hstack([V1,V2,V3])
    
     HC = 1 / pow(As[2,0],2) * np.array([[cam_parameters[0][0]*As[2,0],0,-cam_parameters[0][0]*As[0,0]], \
                                         [0,cam_parameters[0][1]*As[2,0],-cam_parameters[0][1]*As[1,0]]])
    
     Jac_Img_Robot = np.dot(HC,np.hstack([-np.dot(R0.T,R.T),VT]))
        
     return Jac_Img_Robot
 

def JacCamRespectLandmark(x_robot, x_landmark, R0, cam_parameters):
    
    R = Euler2MatRot(x_robot[3], x_robot[4], x_robot[5])
    
    As = np.dot(np.dot(R0.T,R.T),(x_landmark - x_robot[0:3]))
    
    HC = 1 / pow(As[2,0],2) * np.array([[cam_parameters[0][0]*As[2,0],0,-cam_parameters[0][0]*As[0,0]], \
                                         [0,cam_parameters[0][1]*As[2,0],-cam_parameters[0][1]*As[1,0]]])
    
    Jac_Img_Lan = np.dot(HC,np.dot(R0.T,R.T))
    
    return Jac_Img_Lan 
    