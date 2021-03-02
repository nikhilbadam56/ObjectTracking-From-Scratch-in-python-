# ObjectTracking-From-Scratch-in-python

This project aim is to design a Object Tracking Pipeline from scratch in python.

The main components or the pipeline for object tracking is as indicated below : 
  
  1) Load the Video frame by frame.
  
  2) Based on the object of interest(OOI) that we are aiming to track, seting the thresholds for localizing the object in the frame.
  
  3) Locating the OOI centroid for tracking. This measurement tend to be noisy.
  
  4) We use an IMM(Interactive Multiple Model) filter for dynamically selecting the type of physical model as per the dynamics of the object at that instant.
 
 ## PROGRESS
  
  1) Computer Vision Part of the Project has been successfully implemented and the output is as shown in the video below.
  
  [![](https://img.youtube.com/vi/nVDXqHCInUM/0.jpg)](https://www.youtube.com/watch?v=nVDXqHCInUM)
  
  2) For testing and also understanding the working of kalman filter, factored the code following the OOP's principle and then included the kalman filter algorithm in the working pipeline.
     Output is as shown in the video below.
     
  [![](https://img.youtube.com/vi/TarbhNDhHgs/0.jpg)](https://www.youtube.com/watch?v=TarbhNDhHgs)
     
   
