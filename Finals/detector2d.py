import numpy as np 
from cv2 import cv2

def detect(frame, debugMode):

    #Convert to gray bc of data
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if(debugMode):
        cv2.imshow('gray', gray)
    
    img_edges = cv2.Canny(gray, 50, 190, 3)
    if(debugMode):
        cv2.imshow('img_edges', img_edges)

    #B&W
    ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)
    if (debugMode):
        cv2.imshow('img_thresh', img_thresh)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Set the centroid
    min_radius_thresh = 3
    max_radius_thresh = 30

    centers = []
    for c in contours:
        (x,y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        #Time for catching just what we want
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
    
    cv2.imshow('contours', img_thresh)
    return centers
