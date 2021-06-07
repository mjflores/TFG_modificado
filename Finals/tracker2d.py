from cv2 import cv2
import numpy as np
import os
from detector2d import detectNN
from kalman2d import KalmanFilter
from os.path import isfile, join
import statistics as stat
from trackregion2d import trackRegion

#EM FALTA L'ÀREA DE CERCA QUE SIGUI EXACTAMENT EL PUTO YOLO
#NOSE LLEGIR FRAMES

def main():

    HiSpeed = 100
    ControlSpeed = 30 
    debugMode = None

    #Reading with videos
    frame_array = [] 
    tracker_array = []
    VideoCap = cv2.VideoCapture("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\Python\\videoOpcio5.mp4")
    #Read the video time to split into the frames
    while(VideoCap.isOpened()):
        ret, frame = VideoCap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4) #need this for the detector.
        frame_array.append(frame) 
    print(str(len(frame_array)))
    #Reading but with datasets
    '''pathin = os.chdir("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\DATASETS\\U2")
    frame_array = []
    tracker_array = []
    files = [f for f in os.listdir(pathin) if isfile(join(pathin, f))]
    filename = pathin+files[0]
    frame = cv2.imread(filename)'''
    '''
    #show that I'm reading images
    cv2.imshow('frame4', frame_array[95])
    cv2.imshow('frame1', frame_array[0])
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    '''
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
    KF = KalmanFilter(0.1, 1, 1, stdacc, xstdmeas, ystdmeas)
    print("KF Object created")
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
    
            #Draw the detection
            cv2.rectangle(frame, (centers[0],centers[1]), (centers[0]+size[0],centers[1]+size[1]), (0,255,0), 2)
            #Predict
            x_pred, p_pred = KF.predict()
            #Draw the rectangle
            cv2.rectangle(frame, (int(x_pred[0]), int(x_pred[1])), (int(x_pred[0]) + size[0], int(x_pred[1]) + size[1]), (0, 0, 255), 2)
            
            #Measurement function
            #z, size = trackRegion(frame, size, x_pred, p_pred)
            centers, size = detectNN(frame)
            if len(centers) == 0:
                print("MEEEERDA")
                quecoi = 0
                while len(centers) == 0:
                    k = k + 1 
                    frames_used = frames_used + 1
                    centers, size = detectNN(frame_array[k])
                    quecoi = quecoi + 1
                    if quecoi == 10:
                        print("Se fue a la puta y no sabés porqué")
                        break
                k = k - quecoi
            '''            bboxW = centers[0]+size[0]
            bboxH = centers[1]+size[1]'''
        
            #Use the measurment function
            z = np.array([[centers[0]], [centers[1]]])
            cv2.rectangle(frame, (centers[0],centers[1]), (centers[0]+size[0],centers[1]+size[1]), (0,139,139), 2) #Showing to know if it is working
            cv2.imshow('image_detectNN', frame)
            #cv2.waitKey(2000)
            
            #Update
            (x1, y1, vx1, vy1) = KF.update(z)
            #Print the update position
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + size[0], int(y1) + size[1]), (255,0,0), 2)

            cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (int(x_pred[0] + 15), int(x_pred[1])), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", ((centers[0]+size[0],centers[1]+size[1])), 0, 0.5, (0,191,255), 2)
        cv2.imshow('image', frame)
        tracker_array.append(frame)
        '''os.chdir("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\V2Final")
        video = cv2.VideoWriter('video.mp4', 0, 1, (VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH )*0.4,VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT )*0.4))
        video.write(frame)'''
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed - ControlSpeed + 1)
    #Create the video with the tracked frames
    os.chdir("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\V2Final")
    out = cv2.VideoWriter('TRACKING1.avi', cv2.VideoWriter_fourcc(*'DIVX'),15,(round(848*0.4),round(480*0.4)))
    
    for k in range(0, len(tracker_array)):
        out.write(tracker_array[k])

if __name__ == "__main__":
    #execute main
    main()
