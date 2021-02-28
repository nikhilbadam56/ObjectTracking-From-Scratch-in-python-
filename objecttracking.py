# -*- coding: utf-8 -*-


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
In this code we are going to continuously find the centroid location of the an object in the given video feed,
and then use this centroid location and also use a IMM tracking algorithm to find the optimal centroid 
the imm uses a JPDA joint probability data association to associate the correct probability to the tyep of model this object follows.
There by using a correct physical model for prediction and also enchancing the overall tracking capability.
"""
#saturation channel does give me the best values
#l channel does give me the best values.
video = cv.VideoCapture('./car-overhead-1.avi')
flag = True
while video.isOpened():
  ret,frame = video.read()
  if ret == True:
    frame = frame[30:,140:,:]
    r_channel = frame[:,:,0]

    #binarization of the given image
    #threshold values..
    lower = 30
    higher = 80
    r_channel = np.asarray(r_channel)
    r_channel = cv.GaussianBlur(r_channel,(3,3),0)
    binary_image = ((r_channel>lower) & (r_channel<higher))*255
    binary_image = cv.Canny(np.uint8(binary_image),100,200)
    contours,_ = cv.findContours(np.uint8(binary_image), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(frame, contours, -1, (255,0,), 2)
    hull = []
    for i in range(len(contours)):
      # creating convex hull object for each contour
      hull.append(cv.convexHull(contours[i], False))

    for c in range(len(contours)):
      # compute the center of the contour
      M = cv.moments(contours[c])
      if M["m00"] == 0:
        continue
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      # draw the contour and center of the shape on the image
      cv.drawContours(frame, hull[c], -1, (0, 255, 0), 2)
      # cv.circle(frame, (cX, cY), 2, (255, 255, 255), -1)
      # cv.putText(frame, "center", (cX - 20, cY - 20),
      # cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      # show the image
      cv.imshow("Image", frame)
      cv.waitKey(3)
    #region of interest.
  #   cv.imshow('Frame',np.uint8(frame))
  # if cv.waitKey(10) & 0xFF == ord('q'):
  #       break
video.release()
cv.destroyAllWindows()