from types import FrameType
import numpy as np 
import os
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

        
def detectNN(frame):
    print("detecting :)")
    # Load Yolo
    os.chdir("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\Python")
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = ["person"]
    #If we just want one class change this classes array and put the classes that we want
    #with open("coco.names", "r") as f:
    #    classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    #frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape
    print(str(height))
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    centers = []
    size = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centers = [x,y]
                size = [w,h]
                #label = str(classes[class_ids])
    """indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, color, 3)
    cv2.imshow("Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    print("DETECTET")
    print(str(centers))
    return centers, size