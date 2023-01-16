import numpy as np 
         
def Dynamic(x_robot, dt):
         
    A = np.identity(12)
    A[0:6,6:12] = np.identity(6) * dt 
    
    x_robot[0:6] += dt * x_robot[6:12] 
    
    return x_robot, A