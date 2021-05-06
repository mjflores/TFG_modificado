import numpy as np 
from cv2 import cv2
from PIL import Image
import os as os 

dotColor = (255, 255, 255)
puntInici = (100,50)
puntFinal = (300,50)
#Càlcul m i n
m = (puntFinal[1]-puntInici[1])/(puntFinal[0]-puntInici[0])
n = puntInici[1]*(1-m)
#Càlcul la cosa polinòmica
r = np.sqrt((puntInici[0]**2+puntInici[1]**2))

position = (0,0)
img_array = []
frames = 300
i = 0
j = 1
#Primera imatge
img = np.zeros((300, 400, 3), np.uint8)
cv2.circle(img, puntInici, 10, dotColor,-1)
img_array.append(img)
#Creo imatge
while i < frames:
    #Update position
    #posy = (puntInici[0]+j)*m + n
    posy = r**2 - (puntInici[0]+j)**2
    position = (puntInici[0]+j, round(posy))
    
    #Afegeixo imatge
    img = np.zeros((300, 400, 3), np.uint8)
    cv2.circle(img, position, 10, dotColor,-1)
    img_array.append(img)
    i += 1
    j += 1

os.chdir("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\Python")
out = cv2.VideoWriter('cercle1.avi', cv2.VideoWriter_fourcc(*'DIVX'),15,(400,300))

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()

#img.show()
#Creo array amb el que fer el vídeo


