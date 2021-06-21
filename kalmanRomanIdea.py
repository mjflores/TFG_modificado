from cv2 import cv2
import numpy as np
import os
from detector2d import detectNN
from kalman2d import KalmanFilter
from os.path import isfile, join

noise_factor = 2

#state matrix
A = np.matrix([[1, 0, 1, 0],
               [0, 1, 0, 1],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

#Define measurement matrix
H = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0]])

#Begin
v = []

#Initialize noise cov. matrix Q&R
sampleR = np.matrix([[400, 80]]
                    [401, 80]
                    [399, 80],
                    [401, 81],
                    [400, 80])
R = np.cov(sampleR)
sampleQ = np.matrix([[400, 80, 2, 1],
                    [401, 80, 2.1, 1],
                    [399, 80, 1.9, 1],
                    [401, 81, 1.9, 0.9],
                    [400, 80, 2.2, 1.1]])
Q = noise_factor*np.cov(sampleQ)
#Initial Covariance Matrix
P = np.eye(A.shape[1])
#Inicialització imatge bàsicament.
pathin = os.chdir("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\DATASETS\\U2")
frame_array = []
files = [f for f in os.listdir(pathin) if isfile(join(pathin, f))]
filename = pathin+files[0]
frame = cv2.imread(filename)

#Detect and create bounding boxes
centers, size = detectNN(frame)

#For d'imatges
for i in len(files):
    P = np.dot(np.dot(A, P), A.T) + Q
    #Ajuda escriure S = H*P*H' + R
    S = np.dot(H, np.dot(P, H.T)) + R
    #Kalman Gain K = P*H'/S
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    





