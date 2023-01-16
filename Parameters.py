import numpy as np
import math

##############################################################################
#######################  Parameters of the Robot #############
##############################################################################

# Initial conditions
x0 = np.vstack([np.array([[0], [0], [0], [25*math.pi/180],[0*math.pi/180],[-10*math.pi/180]]), np.zeros((6,1))])

# Matrix Q
Q = np.array([[.00001,0,0,0,0,0,0,0,0,0,0,0],
              [0,.0001,0,0,0,0,0,0,0,0,0,0],
              [0,0,.00001,0,0,0,0,0,0,0,0,0],
              [0,0,0,.00000001,0,0,0,0,0,0,0,0],
              [0,0,0,0,.00000001,0,0,0,0,0,0,0],
              [0,0,0,0,0,.00000001,0,0,0,0,0,0],
              [0,0,0,0,0,0,.00001,0,0,0,0,0],
              [0,0,0,0,0,0,0,.00001,0,0,0,0],
              [0,0,0,0,0,0,0,0,.00001,0,0,0],
              [0,0,0,0,0,0,0,0,0,.00000001,0,0],
              [0,0,0,0,0,0,0,0,0,0,.00000001,0],
              [0,0,0,0,0,0,0,0,0,0,0,.00000001]])

# Initial conditions of P of the robot
P0 = np.identity(12) * .0001

# Type orientation
type_ori = "euler"

##############################################################################
#######################  Parameters of the camera ############################
##############################################################################

# Size of the image
cam_size_img = (240,320)  

# Intrinsic parameters
cam_dis = [-0.38999,0.13667,0,0.00057,0]
cam_cen = [171.60729,137.90109]
cam_fc = [209.08860,206.46388]
cam_alpha_c = 0    
cam_intrinsic_par = (cam_fc,cam_cen,cam_dis,cam_alpha_c)

# Initial rotation
R0_cam = np.array([[0,1,0],[1,0,0],[0,0,-1]])

# Matrix R
R_cam = np.identity(2) * 3

##############################################################################
######################  Parameters of the altimeter ##########################
##############################################################################

# Matrix R
R_alt = .1


##############################################################################
############################  Parameters of the SLAM #########################
##############################################################################

# Matrix Q for landmarks
q_landmark = .00001

# size of search into image
size_search = (240/2,320/2,100,100)

# size of the patch
size_patch = (11,11)     

# size of the patch for matching
size_matching = (24,24)

# Size of landmarks on the map
size_map_min = 15
size_map_max = 20

# Initial uncertainty of the depth
inc_ini_landmark = .5

# Correlation value
val_corr = .8

# Number of times without measurements to delete landmark
max_no_visible = 10

# Hypothesis of initial depth for Landmarks
depth_est = 8

# Type of Initialization
type_ini = "Initdepth"

slam_parameters = (size_search,size_patch,size_matching,size_map_min,size_map_max,inc_ini_landmark,val_corr,max_no_visible,depth_est,type_ini)


##############################################################################
########################### Simulation parameters ############################
##############################################################################

# Time of simulation in sec
time_simulation = 90

# Diferential of time in sec
dt = .01
