import numpy as np
import math

from Transformations import Euler2MatRot, Quaternion2MatRot 

def Initdepth(x_robot, pun_undis, R0, type_ori, depth_est): 
    
    if type_ori == "euler":
       R = Euler2MatRot(x_robot[3,0], x_robot[4,0], x_robot[5,0])
    if type_ori == "quaternion":
       R = Quaternion2MatRot(x_robot[3,0], x_robot[4,0], x_robot[5,0], x_robot[6,0]) 
    
    dii = depth_est
    
    AN =  np.dot(np.dot(R,R0),np.array([[pun_undis[0]],[pun_undis[1]],[1]]))
    
    ti = math.atan2(AN[1],AN[0])
    fi = math.atan2(AN[2],math.sqrt(pow(AN[0], 2) + pow(AN[1], 2)))           
    VM = np.array([[math.cos(fi)*math.cos(ti)],[math.cos(fi)*math.sin(ti)],[math.sin(fi)]])  
    
    land = np.transpose(x_robot[0:3] + dii * VM)
    
    return fi, ti, VM, land