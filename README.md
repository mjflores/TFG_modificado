# TFG_modificado

This program is part of the thesis. 

It presents an implementation for tracking by detection using YOLO and Kalman filter.
This system allows detections with YOLO for a concrete object in a video sequence, then the Kalman filter models the motion and tracks the positions of the objects along the sequence.

The YOLO detection would be used to initialize the tracking and as the measurement function of the Kalman Filter. With the Kalman Filter, the motion of the object between frames is estimated, tracking its trajectory.

Finally, the system is tested with different datasets with different objects to track. Every dataset has just one target and each one regards a possible problem of visual object tracking. 
