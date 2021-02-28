import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


"""
In this code we are going to continuously find the centroid location of the an object in the given video feed,
and then use this centroid location and also use a IMM tracking algorithm to find the optimal centroid 
the imm uses a JPDA joint probability data association to associate the correct probability to the tyep of model this object follows.
There by using a correct physical model for prediction and also enchancing the overall tracking capability.
"""
  

def opencv_ground_truth():

  video = cv.VideoCapture('./car-overhead-1.avi') #reading in the data

  out = cv.VideoWriter('outpy.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 20, (180,210)) #object reference to store the output video

  overall_points = [] #ground truth tracked centroid positions.

  while video.isOpened():
    ret,frame = video.read() #reading in frame by frame
    if ret == True:
      frame = frame[30:,140:,:] #localizing the main part of the overall frame

      frame[50:56,85:98,:] = [177,195,195] #occluding a part of the box that is noisy and making centroid shift towards it,

      r_channel = frame[:,:,0] #extracting the r channel

      lower = 40 #lower threshold
      higher = 75 #upper threshold
      r_channel = np.asarray(r_channel) #numpy array of r_channel

      #binarization of the given image
      binary_image = ((r_channel>lower) & (r_channel<higher))*255

      #removing noise
      binary_image = cv.GaussianBlur(np.uint8(binary_image),(7,7),0)
      
      #detecting edges
      binary_image = cv.Canny(np.uint8(binary_image),190,220)

      #detecting the contours
      contours,_ = cv.findContours(binary_image,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

      #this algorithm, by above steps, extracts only relevant contours and takes the average to get the centroid position

      x_avg = 0
      y_avg = 0
      count = 0
      for c in range(len(contours)):
        # compute the center of the contour
        M = cv.moments(contours[c])
        if M["m00"] == 0:
          continue

        count+=1
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        x_avg+=cX
        y_avg+=cY

      if not count==0:
        x_avg = x_avg/count
        y_avg = y_avg/count
        overall_points.append([int(x_avg),int(y_avg)])
        
        cv.drawContours(frame,contours,-1,(0,0,255),2)
        cv.circle(frame, (int(x_avg), int(y_avg)), 3, (255, 255, 255), -1)
        cv.polylines(frame,[np.asarray(overall_points)],False,(0,255,0),1)
        cv.imshow('Frame',np.uint8(frame))
      else:
        cv.imshow('Frame',np.uint8(frame))

      out.write(frame)
    
    #reading in the frame a particular deulay.
    if cv.waitKey(6) & 0xFF == ord('q'):
          break
  video.release()
  cv.destroyAllWindows()
opencv_ground_truth()