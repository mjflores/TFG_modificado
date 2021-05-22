from cv2 import cv2
import os
from detector2d import detect
from detector2d import detectNN
from kalman2d import KalmanFilter

def main():

    HiSpeed = 100
    ControlSpeed = 30 
    debugMode = 1

    #Time for creating Kalman Filter
    KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

    #Open the stream
    #VideoCap = cv2.VideoCapture('C:\\Program Files\\video_randomball.avi')
    VideoCap = cv2.VideoCapture("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\Python\\videoOpcio4.avi")
    while(True):
        #Read frame
        ret, frame = VideoCap.read()
        #detect object
        centers, size = detectNN(frame, debugMode)
        #Centroids pls work
        if (len(centers) > 0):
    
            #Draw the detection
            cv2.rectangle(frame, (centers[0],centers[1]), (centers[0]+size[0],centers[1]+size[1]), (0,139,139), 2)
            #cv2.putText(frame, label, (centers[0], centers[1] + 30), (0,139,139), 2)
            #cv2.rectangle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)
            #Predict
            (x, y) = KF.predict()

            #Draw the rectangle
            cv2.rectangle(frame, (int(x - size[0]), int(y - size[1])), (int(x + size[0]), int(y + size[1])), (139, 0, 139), 2)

            #Update
            (x1, y1) = KF.update(centers[0])
            
            cv2.rectangle(frame, (int(x1 - size[0]), int(y1 - size[1])), (int(x1 + size[0]), int(y1 + size[1])), (0, 0, 255), 2)
            
            cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", ((centers[0]+size[0],centers[1]+size[1])), 0, 0.5, (0,191,255), 2)
        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed - ControlSpeed + 1)
        


if __name__ == "__main__":
    #execute main
    main()
