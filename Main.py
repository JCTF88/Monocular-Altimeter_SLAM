import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

import Parameters
from ModelRobot import Dynamic
from SLAM import EKFmonocular


slam = EKFmonocular(x_robot = Parameters.x0,
                    Q_robot = Parameters.Q,
                    P0_robot = Parameters.P0,
                    camera_size_img = Parameters.cam_size_img,
                    camera_intrinsic_par = Parameters.cam_intrinsic_par,
                    camera_R0 = Parameters.R0_cam,
                    camera_R = Parameters.R_cam,
                    slam_par = Parameters.slam_parameters,
                    q_landmark = Parameters.q_landmark,
                    type_ori = Parameters.type_ori,
                    altimeter_R = Parameters.R_alt,)

path_img = 'Imagenes/'
df_frame = pd.read_excel('Archivos/frame_list.xlsx', sheet_name='frame_list', header=None)
df_frame_time = pd.read_excel('Archivos/Frame.xlsx', sheet_name='Hoja1', header=None)
df_gps_time = pd.read_excel('Archivos/GPS.xlsx', sheet_name='Hoja1', header=None)
df_altimeter_time = pd.read_excel('Archivos/Altimetro.xlsx', sheet_name='Hoja1', header=None)

# Principal cycle
XE = np.empty((0, 12))
XR = np.empty((0, 3))
tim = 0
cont_frame = 0
cont_gps = 0
cont_altimeter = 0
while tim <= Parameters.time_simulation:

    # Times and flags
    time_frame = df_frame_time.iloc[cont_frame, 0] 
    if time_frame >= tim and time_frame < tim + Parameters.dt:
      bandera_frame = 1
    else:
      bandera_frame = 0

    time_gps = df_gps_time.iloc[cont_gps, 0] 
    if time_gps >= tim and time_gps < tim + Parameters.dt:
       bandera_gps = 1
    else:
       bandera_gps = 0  

    time_altimeter = df_altimeter_time.iloc[cont_altimeter, 0] 
    if time_altimeter >= tim and time_altimeter < tim + Parameters.dt:
       bandera_altimeter = 1
    else:
       bandera_altimeter = 0      

    #Read GPS
    if bandera_gps == 1:
       x_gps = np.array([[df_gps_time.iloc[cont_gps, 1],df_gps_time.iloc[cont_gps, 2],df_gps_time.iloc[cont_gps, 3]]])
       XR = np.vstack([XR, x_gps])   

    #Estimation a priori
    x_robot, Map = slam.Prediction(Dynamic, args = (Parameters.dt))  

    #Estimation a posteriori (camera)               
    if bandera_frame == 1:
       path_img_read = path_img + df_frame.iloc[cont_frame, 0] 
       img = cv2.imread(path_img_read, cv2.IMREAD_GRAYSCALE)
       x_robot, Map = slam.CameraMeasurement(img) 

    #Estimation a posteriori (altimeter)               
    if bandera_altimeter == 1:
       x_robot, Map = slam.AltimeterMeasurement(df_altimeter_time.iloc[cont_altimeter, 1])     

    # Save Estimated State   
    XE = np.vstack([XE, x_robot.T])  
    
    # Counters
    if bandera_frame == 1:
       cont_frame = cont_frame + 1 
    if bandera_gps == 1:
       cont_gps = cont_gps + 1 
    if bandera_altimeter == 1:
       cont_altimeter = cont_altimeter + 1       
    
    # Time
    tim = tim + Parameters.dt

# Graph
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(XR.T[0], XR.T[1], XR.T[2],linestyle='-')
ax1.scatter(XE.T[0], XE.T[1], XE.T[2],linestyle='-')
ax1.legend(['GPS','Estimated Trajectory'])
plt.show()    
