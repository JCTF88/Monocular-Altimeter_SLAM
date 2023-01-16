import math
import numpy as np

def UndistortPoint(point, cam_parameters):
    
    u = (point[0] - cam_parameters[1][0]) / cam_parameters[0][0];
    v = (point[1] - cam_parameters[1][1]) / cam_parameters[0][1];
    
    rd = math.sqrt(pow(u, 2) + pow(v, 2))
    
    radial_u = 1 + cam_parameters[2][0] * pow(rd, 2) + cam_parameters[2][1] * pow(rd, 4)
    radial_v = 1 + cam_parameters[2][0] * pow(rd, 2) + cam_parameters[2][1] * pow(rd, 4)
    
    tangencial_u = 2 * cam_parameters[2][2] * u * v + cam_parameters[2][3] * (pow(rd, 2) + 2*pow(u, 2))
    tangencial_v = cam_parameters[2][2] * (pow(rd, 2) + 2*pow(v, 2)) + 2 * cam_parameters[2][3] * u * v 
    
    u = (point[0] - cam_parameters[1][0] - tangencial_u) / (cam_parameters[0][0] * radial_u)
    v = (point[1] - cam_parameters[1][1] - tangencial_v) / (cam_parameters[0][1] * radial_v)
    
    pun_undis = np.hstack([u,v])
    
    return pun_undis


def DistortPoint(point, cam_parameters):
    
    up, vp = point[0], point[1]
    
    rd = math.sqrt(pow(up, 2) + pow(vp, 2))
    
    radial_u = 1 + cam_parameters[2][0] * pow(rd, 2) + cam_parameters[2][1] * pow(rd, 4)
    radial_v = 1 + cam_parameters[2][0] * pow(rd, 2) + cam_parameters[2][1] * pow(rd, 4)
    
    tangencial_u = 2 * cam_parameters[2][2] * up * vp + cam_parameters[2][3] * (pow(rd, 2) + 2*pow(up, 2))
    tangencial_v = cam_parameters[2][2] * (pow(rd, 2) + 2*pow(vp, 2)) + 2 * cam_parameters[2][3] * up * vp
    
    uup = up * radial_u + tangencial_u; 
    vvp = vp * radial_v + tangencial_v; 
    
    u = cam_parameters[0][0] * uup + cam_parameters[1][0]
    v = cam_parameters[0][1] * vvp + cam_parameters[1][1]
    
    pun_dis = np.hstack([u,v])
    
    return pun_dis