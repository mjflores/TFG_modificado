from cv2 import cv2
import os
from os.path import isfile, join
from detector2d import detectNN

#def readAndDetect(pathin):
def main(pathin):
    HiSpeed = 100
    ControlSpeed = 30 
    debugMode = 1
    imgpath = pathin
    frame_array = []
    files = [f for f in os.listdir(pathin) if isfile(join(pathin, f))]
    
    for i in range(len(files)):
        filename = pathin+files[i]
        print(filename)
        img=cv2.imread(filename)
        
        #height, width, layers = img.shape
        size = (640, 480)

        for k in range(5):
            frame_array.append(img)
        
        centers, size = detectNN(img, debugMode)
        os.chdir("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\DATASETS")
        fichero = open('Boxes1.txt','wb')
        str = str(centers[0]) + " " + str(centers[1]) + " " + str(size[0]) + " " + str(size[1]) + "\n"
        fichero.write(str)
        fichero.close()

if __name__ == "__main__":
    #execute main
    main("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\DATASETS\\U2")