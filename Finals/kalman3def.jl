using Kalman, GaussianDistributions, LinearAlgebra
using GaussianDistributions: ⊕ # independent sum of Gaussian r.v.
using Statistics
using VideoIO
using Images
using ImageView
using PyCall
using Conda

function detector(image)
   py"""
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
    """
end

function predict()
    x = A*x + B*u
    P = A*P*A' + Q
    x_ret = (x[0], x[1])
    return x_ret
end

function update(z)
    S = H*P*H' + R
    K = (P*H')/S
    x = x + K*(z - K*x)
    x = round(x)
    P = (eye(H) - K*H)*P
    x_ret = (x[0], x[1])
    return x_ret
end

# INICI PARÀMETRES
dt = 0.1 #Times used to sample
stdacc = 1 #covariança acceleració
xstdmeas = 0.1
ystdmeas = 0.1
u = [1 ; 1] #Acceleration parameter
x = zeros(4,4)
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]
B = [(dt^2)/2 0; 0 (dt^2)/2; dt 0; 0 dt]
H = [1 0 0 0; 0 1 0 0]
P = eye(4)
Q = [(dt^4)/4 0 (self.dt^3)/2 0; 0 (dt^4)/4 0 (dt^3)/2; (dt^3)/2 0 (dt^2) 0; 0 (dt^3)/2 0 (dt^2)] * stdacc^2
R = [xstdmeas^2 0; 0 ystdmeas^2]
# dynamics
Φ = [0.8 0.2; -0.1 0.8]

#OBRIR VIDEO
VideoCap = VideoIO.testvideo("C:\\Users\\Èric Quintana\\Desktop\\TELECOM\\TFG\\Scripts\\Python\\rectes.avi")
f = VideoIO.openvideo(VideoCap)
img = read(f)

while true
    frame = read!(f,img)
    centers = detector(frame)

    #Comprovar que tenim coses
    if length(centers) > 0
        #encerclar les coses
        draw!(frame, Ellipse(CirclePointRadius(50, centers[0], centers[1]; thickness = 15; fill = false)), RGB{N0f8}(1))
        #Predir
        (x, y) = predict()
        #dibuixar predicció
        draw!(frame, Ellipse(CirclePointRadius(50, x, y; thickness = 15; fill = false)), RGB{N0f32}(1))
        #Update i dibuixar Update
        (x1, y1) = update(centers[0])
        draw!(frame, Ellipse(CirclePointRadius(50, x1, y1; thickness = 15; fill = false)), RGB{N10f22}(1))
    end
    imshow(frame)
 

end
