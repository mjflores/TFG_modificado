from cv2 import cv2
from detector2d import detect
from kalman2d import KalmanFilter

def main():

    HiSpeed = 100
    ControlSpeed = 30 
    debugMode = 1

    #Time for creating Kalman Filter
    KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

    #Open the stream
    #VideoCap = cv2.VideoCapture('C:\\Program Files\\video_randomball.avi')
    VideoCap = cv2.VideoCapture("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\Python\\rectes.avi")
    while(True):
        #Read frame
        ret, frame = VideoCap.read()

        #detect object
        centers = detect(frame, debugMode)
        #Centroids pls work
        if (len(centers) > 0):
    
            #Draw the detection
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)
            #Predict
            (x, y) = KF.predict()

            #Draw the rectangle
            cv2.rectangle(frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (0, 139, 139), 2)

            #Update
            (x1, y1) = KF.update(centers[0])
            
            cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)
            
            cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (int(centers[0][0] + 15), int(centers[0][1] - 15)), 0, 0.5, (0,191,255), 2)
        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed - ControlSpeed + 1)


if __name__ == "__main__":
    #execute main
    main()
