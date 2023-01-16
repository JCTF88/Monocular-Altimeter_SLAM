import cv2
import numpy as np

orb = cv2.ORB_create(nfeatures=1500, fastThreshold = 5)

def Highlights(img, size_img, size_path_search):
    
     cut_u_1 = int(size_path_search[1] - size_path_search[3]/2) 
     cut_u_2 = int(size_path_search[1] + size_path_search[3]/2) 
     cut_v_1 = int(size_path_search[0] - size_path_search[2]/2) 
     cut_v_2 = int(size_path_search[0] + size_path_search[2]/2)
     
     if cut_u_1 < 0:
         cut_u_1 = 0
         
     if cut_u_2 > size_img[1]:
         cut_u_2 = size_img[1]
         
     if cut_v_1 < 0:
         cut_v_1 = 0
         
     if cut_v_2 > size_img[0]:
         cut_v_2 = size_img[0]    
    
     img2 = img[cut_v_1:cut_v_2,cut_u_1:cut_u_2]
     keypoints_orb, descriptors = orb.detectAndCompute(img2, None)
     
     Points = np.empty((0,2))     
     for i in range(0,np.shape(keypoints_orb)[0],1):
         Points = np.vstack([Points,np.array([[round(keypoints_orb[i].pt[0])+cut_u_1,round(keypoints_orb[i].pt[1])+cut_v_1]])])
         
     return Points

