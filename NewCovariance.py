import numpy as np
import math

def NewCovarianceLandmarkInitdepth(pun_undis, fi, ti, dii, VM, R_cam, P, inc_ini_lan):
    
    u, v = pun_undis[0], pun_undis[1]
    
    Der_VM_ti = np.vstack([-math.cos(fi)*math.sin(ti)*dii,math.cos(fi)*math.cos(ti)*dii,0])
    Der_VM_fi = np.vstack([-math.cos(ti)*math.sin(fi)*dii,-math.sin(ti)*math.sin(fi)*dii,math.cos(fi)*dii])
    
    der_ti_u = 1 / (v * (pow(u, 2)/pow(v, 2) + 1))
    der_ti_v = -u / (pow(v, 2) * (pow(u, 2)/pow(v, 2) + 1)) 
    
    der_fi_u = u / (pow(pow(u, 2)+pow(v, 2),3/2) * (1 / pow(u, 2)+pow(v, 2) + 1))
    der_fi_v = v / (pow(pow(u, 2)+pow(v, 2),3/2) * (1 / pow(u, 2)+pow(v, 2) + 1))
    
    R_new = np.array([[R_cam[0,0],0,0],[0,R_cam[0,0],0],[0,0,inc_ini_lan]])
    
    Jac_ini = np.hstack([np.identity(3),np.zeros((3,3)),np.zeros((3,np.shape(P)[0]-6)),Der_VM_ti*der_ti_u+Der_VM_fi*der_fi_u,Der_VM_ti*der_ti_v+Der_VM_fi*der_fi_v,VM])
    
    P_o_1 = np.hstack([P,np.zeros((np.shape(P)[0],3))])
    P_o_2 = np.hstack([np.zeros((3,np.shape(P)[0])),R_new])
    P_o = np.vstack([P_o_1,P_o_2])
    P_new = np.dot(np.dot(Jac_ini,P_o),Jac_ini.T)
    
    return P_new