from math import atan
from cv2 import cv2
import numpy as np
from numpy import linalg as la
import math
from numpy.core.fromnumeric import size
from detector2d import detectNN
import statistics as stat

def trackRegion(frame, bbox, x_pred, P_pred):
    #Constants for the uncertainity 
    c2 = 5.991
    s = [5,5]
    
    #ellipses
    (v, d) = la.eig(P_pred)
    if d[0][0] > d[1][1]:
        a = math.sqrt(c2*d[0][0])
        b = math.sqrt(c2*d[1][1])
        alpha = atan(v[0][1]/v[0][0])
    else:
        a = math.sqrt(c2*d[1][1])
        b = math.sqrt(c2*d[0][0])
        alpha = atan(v[0][1]/v[0][0])

    w2 = np.ceil(math.sin(alpha)*b + math.cos(alpha)*a)
    h2 = np.ceil(math.cos(alpha)*b + math.sin(alpha)*a)
    #voldré fer servir x_pred per trobar la longitud? fcuk.
    win = np.array([[np.array([[x_pred[0], [x_pred[1]]]]) - np.array([w2, h2])], [np.array([[x_pred[0], [x_pred[1]]]]) + np.array([w2, h2])]])
    #Crec que bbox per mi és size
    w2 = np.ceil(bbox[0]/2)
    h2 = np.ceil(bbox[1]/2)

    #predicted detection window
    win[0] = win[0] - np.array([w2, h2]) - s
    win[1] = win[1] + np.array([w2, h2]) - s

    #Change the frame in oder to just select the window.  
    centers, size = detectNN(frame)

    if len(centers) == 0:
        #STOP TRACKING
        bbox = -1
        z = np.array([[0], [0]])
        return z

    
    z = np.array([[centers[0]], [centers[1]]])
    return z, size
    
