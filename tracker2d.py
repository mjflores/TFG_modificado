from math import dist, sqrt
from cv2 import cv2
import numpy as np
import os
from detector2d import detectNN
from kalman2d import KalmanFilter
from os.path import isfile, join
import statistics as stat
from trackregion2d import trackRegion
from pylab import *

def main():

    HiSpeed = 100
    ControlSpeed = 30 
    debugMode = None

    #Reading with videos
    frame_array = [] 
    tracker_array = []
    distance_detectx = []
    distance_detecty = []
    distance_detect = []
    VideoCap = cv2.VideoCapture("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\Python\\tennis.mp4")
    #Read the video time to split into the frames
    while(VideoCap.isOpened()):
        ret, frame = VideoCap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4) #need this for the detector.
        frame_array.append(frame) 
    print(str(len(frame_array)))
    #Reading but with datasets

    k = 0 #Frames control
    #Detect and create bounding boxes
    centers, size = detectNN(frame_array[0])
    #make sure there's a detection
    if len(centers) == 0:
        print("NOTHING FOUND FIRST FRAME")
        while len(centers) == 0:
            k = k + 1 
            centers, size = detectNN(frame_array[k])
    
    #Bbox
    bboxW = centers[0]+size[0]
    bboxH = centers[1]+size[1]
    cv2.rectangle(frame_array[k], (centers[0],centers[1]), (bboxW,bboxH), (0,139,139), 2)
    cv2.imshow('image_detectNN', frame_array[k]) #print the first frame to see that it is working and detecting.
    cv2.waitKey(1999) 

    #Values for KF object
    xstdmeas = stat.stdev([centers[0],centers[0]+size[0]/10,centers[0]+size[0]/9,centers[0]+size[0]/8,centers[0]+size[0]/5,centers[0]+size[0]/4,centers[0]+size[0]/7])
    ystdmeas = stat.stdev([centers[1],centers[1]+size[1]/10,centers[1]+size[1]/9,centers[1]+size[1]/8,centers[1]+size[1]/5,centers[1]+size[1]/4,centers[1]+size[1]/7])
    #noise
    stdacc = 2
    #Time for creating Kalman Filter
    KF = KalmanFilter(0.1, centers[0], centers[1], 1, 1, stdacc, xstdmeas, ystdmeas)
    print("KF Object created")
    #Arrays for the dots
    pred_dot = []
    updt_dot = []
    #Now kalman shows off
    frames_used = k
    for i in range(len(frame_array)-frames_used):
        #Read frame
        '''filename = pathin+files[i]
        frame = cv2.imread(filename)'''
        print("STARTING MAIN FOR, ITERATION: " + str(i))
        if k > len(frame_array):
            print("diguem-ne final")
            break
        frame = frame_array[k] #put the next frame
        k = k + 1 
        print(str(k))
        #Centroids pls work
        if (len(centers) > 0):
            #Predict
            x_pred, p_pred = KF.predict()
            if size != [1,1]:
                pred_point = (int(x_pred[0]+size[0]/2), int(x_pred[1]+size[1]/2))
            
            pred_dot.append(pred_point)
            #Draw the positions
            cv2.rectangle(frame, (int(x_pred[0]), int(x_pred[1])), (int(x_pred[0]) + size[0], int(x_pred[1]) + size[1]), (0, 0, 255), 2)
            for m in range(len(pred_dot)):
                if size[0] != 1:
                    cv2.circle(frame, pred_dot[m],3,(0, 0, 255), -1)
            #Measurement function
            #z, size = trackRegion(frame, size, x_pred, p_pred)
            centers, size = detectNN(frame)
            if len(centers) != 0:
                centers_prior = centers
                size_prior = size
            #Make sure there's a detection
            if len(centers) == 0:
                print("No detection")
                quecoi = 0
                while len(centers) == 0:
                    k = k + 1 
                    if k >= len(frame_array):
                        print("The end")
                        centers = centers_prior
                        size = size_prior
                        break
                    frames_used = frames_used + 1
                    centers, size = detectNN(frame_array[k])
                    quecoi = quecoi + 1
                    if quecoi == 10:
                        print("For sure, not detection")
                        centers = centers_prior
                        size = size_prior
                        break
                k = k - quecoi
 
            #Use the measurment function
            z = np.array([[centers[0]], [centers[1]]])
            cv2.rectangle(frame, (centers[0],centers[1]), (centers[0]+size[0],centers[1]+size[1]), (0,191,255), 2) #Showing to know if it is working
            
            #Update
            (x1, y1, vx1, vy1) = KF.update(z)
            #Print the update position
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + size[0], int(y1) + size[1]), (255,0,0), 2)
            if size != [1,1]:
                updt_point = (int(x1+size[0]/2), int(y1+size[1]/2))
            
            updt_dot.append(updt_point)
            for m in range(len(updt_dot)):
                cv2.circle(frame, updt_dot[m],3,(255, 0, 0), -1)

            distance_detectx.append(int(x1-centers[0]))  
            distance_detecty.append(int(y1-centers[1]))
            distance_detect.append(int(sqrt(int(x1-centers[0])**2+int(y1-centers[1])**2)))


            cv2.putText(frame, "Updated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Predicted Position", (int(x_pred[0] + 15), int(x_pred[1])), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Measured Position", ((centers[0]+size[0],centers[1]+size[1])), 0, 0.5, (0,191,255), 2)
        cv2.imshow('image', frame)
        tracker_array.append(frame)
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed - ControlSpeed + 1)
    #Create the video with the tracked frames
    
    os.chdir("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\V2Final")
    plot(distance_detect)
    xlabel("distance between detection and estimated position")
    title("Euclidian Distance")
    draw()
    savefig("distanceseuclidianTENNIS",dpi=300)
    close()
 
    out = cv2.VideoWriter('TRACKING_point3.avi', cv2.VideoWriter_fourcc(*'DIVX'),15,(round(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.4),round(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.4)))
    for k in range(0, len(tracker_array)):
        out.write(tracker_array[k])

if __name__ == "__main__":
    #execute main
    main()
