import numpy as np

from Get_Features import Highlights
import Initializations 
from Cut_Patches import ObtPatches
from EKF import Prediction, EKFcorrectionCamera, EKFcorrectionAltimeter 
from ProjectionCamera import ModelCamera
from Correlation import normxcorr2
from TreatmentPoint import UndistortPoint
from NewCovariance import NewCovarianceLandmarkInitdepth


class EKFmonocular:
    
    def __init__(self, x_robot, Q_robot, P0_robot, camera_size_img, camera_intrinsic_par, camera_R0, camera_R,
                 slam_par, q_landmark, type_ori, altimeter_R):
       
       self.x_robot = x_robot
       self.Q_robot = Q_robot
       self.P0_robot = P0_robot
       self.type_ori = type_ori
       self.x_landmarks = np.empty((0,3))
       self.P = self.P0_robot
       self.camera_size_img = camera_size_img 
       self.camera_intrinsic_par = camera_intrinsic_par 
       self.camera_R0 = camera_R0
       self.camera_R = camera_R
       self.altimeter_R = altimeter_R
       self.parameters = slam_par
       self.q_landmark = q_landmark
       self.Cont_num_measurements = np.empty((0,2))
       self.Pro_landmarks = np.empty((0,2))
       self.Patchs = np.empty((0,self.parameters[1][0],self.parameters[1][1])) 
       self.FinalMap = np.empty((0,3))
       
    def Prediction(self, fun, args):
        
        # Estimation a priori       
        self.x_robot, A_robot = fun(self.x_robot, args)
        self.P = Prediction(self.Q_robot,
                            self.q_landmark,
                            self.P,
                            self.x_landmarks.shape[0],
                            A_robot)         
        return self.x_robot, self.FinalMap

    def CameraMeasurement(self, img):
        
        # Join State vector
        if self.x_landmarks.shape[0] >= 1:
            XE = self.x_robot
            X_lan = self.x_landmarks.ravel()[:,np.newaxis] 
            XE = np.vstack([XE,X_lan])                   
        else:
            XE = self.x_robot    

        # Point estimation, Matching, Estimation a posteriori (Camera) and Delete 
        if self.x_landmarks.shape[0] >= 1:
            delete = np.empty((0,1))
            for i in range(0,self.x_landmarks.shape[0],1): 
                pun_est = ModelCamera(self.x_robot,
                                      np.transpose(np.array([self.x_landmarks[i][0:3]])),
                                      self.camera_intrinsic_par,
                                      self.camera_R0,
                                      self.type_ori)
                # Visibility condition
                if pun_est[0] - self.parameters[2][1] / 2 > 0 and \
                   pun_est[0] + self.parameters[2][1] / 2 <= self.camera_size_img[1] and \
                   pun_est[1] - self.parameters[2][0] / 2 > 0 and \
                   pun_est[1] + self.parameters[2][0] / 2 <= self.camera_size_img[0]:
                       Patch_med = ObtPatches(img, pun_est, self.parameters[2])
                       # Correlation condition
                       corr = normxcorr2(self.Patchs[i][:][:], Patch_med)
                       val = np.amax(corr) 
                       if val >= self.parameters[6]:
                           self.Cont_num_measurements[i][0] += 1
                           self.Cont_num_measurements[i][1] = 0
                           place = np.where(corr == np.amax(corr))
                           self.Pro_landmarks[i][0] = place[1][0] + pun_est[0] - self.parameters[2][1]/2 - self.parameters[1][1]/2
                           self.Pro_landmarks[i][1] = place[0][0] + pun_est[1] - self.parameters[2][0]/2 - self.parameters[1][0]/2
                           XE, self.P = EKFcorrectionCamera(self.x_robot,
                                                            np.transpose(np.array([self.x_landmarks[i][0:3]])),
                                                            self.camera_R0,
                                                            self.camera_R,
                                                            self.camera_intrinsic_par,
                                                            XE,
                                                            np.transpose(np.array([self.Pro_landmarks[i][0:2]])),
                                                            self.P,
                                                            i+1,
                                                            self.x_landmarks.shape[0],
                                                            pun_est[:,np.newaxis])
                       else:
                           self.Cont_num_measurements[i][1] += 1              
                else:
                    self.Cont_num_measurements[i][1] += 1 
                if self.Cont_num_measurements[i][1] >= self.parameters[7]:
                    delete = np.vstack([delete,i]) 
                    self.FinalMap = np.vstack([self.FinalMap,np.array([self.x_landmarks[i][0:3]])])
            if np.shape(delete)[0] >= 1:
               self.x_landmarks = np.delete(self.x_landmarks,delete.astype(int)[0:],axis=0)
               self.Cont_num_measurements = np.delete(self.Cont_num_measurements,delete.astype(int)[0:],axis=0)
               self.Pro_landmarks = np.delete(self.Pro_landmarks,delete.astype(int)[0:],axis=0)
               self.Patchs = np.delete(self.Patchs,delete.astype(int)[0:],axis=0)
            for i in range(0,np.shape(delete)[0],1): 
                re = int(delete[i] - i) 
                a = list(range(12+re*3,12+re*3+3))
                self.P = np.delete(self.P,a,axis=0)
                self.P = np.delete(self.P,a,axis=1)   
                
        # Separation of the State vector
        if self.x_landmarks.shape[0] >= 1:
            for i in range(0,self.x_landmarks.shape[0],1):
                self.x_landmarks[i][0:3] = XE[12+(i*3):12+(i*3)+3].T
        self.x_robot = XE[0:12] 
            
        # Inicialization of Landmarks
        if np.shape(self.x_landmarks)[0] < self.parameters[3]:
            New_points = Highlights(img, self.camera_size_img, self.parameters[0])
            if np.shape(New_points)[0] > 0:
                if self.parameters[9] == "Initdepth":        
                      i = 0
                      while np.shape(self.x_landmarks)[0] < self.parameters[4] and i < np.shape(New_points)[0]: 
                            pun_undis = UndistortPoint(New_points[i][:], self.camera_intrinsic_par)
                            fi, ti, VM, land = Initializations.Initdepth(self.x_robot,
                                                                               pun_undis,
                                                                               self.camera_R0,
                                                                               self.type_ori,
                                                                               self.parameters[8])
                            P_new = NewCovarianceLandmarkInitdepth(pun_undis,
                                                                   fi,
                                                                   ti,
                                                                   self.parameters[8],
                                                                   VM,
                                                                   self.camera_R,
                                                                   self.P,
                                                                   self.parameters[5])
                            self.x_landmarks = np.vstack([self.x_landmarks,land])
                            self.Cont_num_measurements = np.vstack([self.Cont_num_measurements,np.array([[0,0]])])
                            self.Pro_landmarks = np.vstack([self.Pro_landmarks,New_points[i][0:2]])
                            New_Patch = ObtPatches(img, New_points[i][0:2], self.parameters[1])
                            self.Patchs = np.vstack([self.Patchs,[New_Patch]])
                            P_1 = np.hstack([self.P,np.zeros((np.shape(self.P)[0],3))])
                            P_2 = np.hstack([np.zeros((3,np.shape(self.P)[0])),P_new])
                            self.P = np.vstack([P_1,P_2])
                            i += 1    
        
        # Final Map
        self.FinalMap = np.vstack([self.FinalMap, self.x_landmarks])
                        
        return self.x_robot, self.FinalMap


    def AltimeterMeasurement(self, z_altimeter):

        # Join State vector
        if self.x_landmarks.shape[0] >= 1:
            XE = self.x_robot
            X_lan = self.x_landmarks.ravel()[:,np.newaxis] 
            XE = np.vstack([XE,X_lan])                   
        else:
            XE = self.x_robot 
        
        # Estimation a posteriori (altimeter)
        XE, self.P = EKFcorrectionAltimeter(XE,
                                            z_altimeter,
                                            self.P,
                                            self.x_landmarks.shape[0],
                                            self.altimeter_R)
        
        # Separation of the State vector
        if self.x_landmarks.shape[0] >= 1:
            for i in range(0,self.x_landmarks.shape[0],1):
                self.x_landmarks[i][0:3] = XE[12+(i*3):12+(i*3)+3].T
        self.x_robot = XE[0:12] 

        return self.x_robot, self.FinalMap

