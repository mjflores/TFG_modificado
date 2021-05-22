import cv2
import os

image_folder = 'C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\Python\\datasetMAN'
video_name = 'videoMAN.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
#height, width, layers = frame.shape
height = 180
width = 320

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()