import numpy as np

from Transformations import Euler2MatRot, Quaternion2MatRot 
from TreatmentPoint import DistortPoint 

def ModelCamera(x_robot, x_landmark, cam_parameters, R0, type_ori):
    
    if type_ori == "euler":
       R = Euler2MatRot(x_robot[3,0], x_robot[4,0], x_robot[5,0])
    if type_ori == "quaternion":
       R = Quaternion2MatRot(x_robot[3,0], x_robot[4,0], x_robot[5,0], x_robot[6,0]) 
       
    As = np.dot(np.dot(R0.T,R.T),(x_landmark - x_robot[0:3]))
          
    u =  As[0] / As[2] 
    v =  As[1] / As[2]
    
    point = np.hstack([u,v])
    
    pun_est = DistortPoint(point, cam_parameters)

    return pun_est
