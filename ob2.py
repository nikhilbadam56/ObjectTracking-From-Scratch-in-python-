import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math

"""
In this code we are going to continuously find the centroid location of the an object in the given video feed,
and then use this centroid location and also use a IMM tracking algorithm to find the optimal centroid 
the imm uses a JPDA joint probability data association to associate the correct probability to the tyep of model this object follows.
There by using a correct physical model for prediction and also enchancing the overall tracking capability.
"""

class Object_Tracking():
  def __init__(self):
    
    self.first_measurement = True
    self.prev_optimal_state_estimate = None
    self.ax = None #noise along the x axis
    self.ay = None #noise along the y axis
    self.prev_time = None
    self.Q = None #process covariance matrix
    self.F=  None #state transition function
    self.R = None
    self.H = None
    self.frame_row = None
    self.frame_col = None

  def get_state_transition_matrix(self,time_elapsed):

    return np.asarray([[1,0,time_elapsed,0],[0,1,0,time_elapsed],[0,0,1,0],[0,0,0,1]])

  def get_process_covariance(self,time_elapsed):

    dt = time_elapsed
    dt_2 = dt*dt
    dt_3 = dt_2*dt
    dt_4 = dt_3*dt
    
    noise_ax = self.ax
    noise_ay = self.ay
    Q = np.asarray([[(dt_4/4)*noise_ax,0,(dt_3/2)*noise_ax,0],
                [0,(dt_4/4)*noise_ay,0,(dt_3/2)*noise_ay],
                [(dt_3/2)*noise_ax,0,(dt_2)*noise_ax,0],
                [0,(dt_3/2)*noise_ay,0,(dt_2)*noise_ay]])
    return Q

  def kalman_filter_2d_simple(self,measurement):

    """
    This algorithm uses the measurement received and then predicts and then corrects the position estimate and return
    the result for plotting

    For testing here we will use a linear kalman filter with the assumption that every measurement is a gaussian, and noise is also a gaussian

    Physical model used is Constant velocity model which is not the case, so there will be a noise in terms of the accelerations.
    """
    if self.first_measurement:
      #then just initialize the matrices
      #what matrices?
      #State transition matrix A
      #Prediction covariance matrix
      #initialize the previous_optimal_state_estimate
      #out state estimate a four dimensional state with x and y positions and x and y velocity component magnitude 

      self.prev_time = time.time()
      
      #acceleration noise parameters
      self.ax = 2 #1m/s^2
      self.ay = 2 #1m/s^2


      self.P = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #process error 

      self.prev_optimal_state_estimate = np.asarray([self.frame_col/2,self.frame_row,0,0])

      self.R = np.asarray([[0.22,0,0,0],[0,0.22,0,0],[0,0,0.22,0],[0,0,0,0.22]])

      self.H = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

      self.first_measurement = False

      return self.prev_optimal_state_estimate

    #elapesed time
    a = time.time()
    elapesed_time = a - self.prev_time
    self.prev_time = a

    self.F = self.get_state_transition_matrix(elapesed_time)

    #prediction
    x_estimate = np.matmul(self.F,self.prev_optimal_state_estimate)
    self.P = np.matmul(np.matmul(self.F,self.P),np.transpose(self.F)) + self.get_process_covariance(elapesed_time)

    if not measurement is None:
      #correction
      s = np.matmul(np.matmul(self.H,self.P),np.transpose(self.H))+self.R
      kalman_gain = np.matmul(np.matmul(self.P,np.transpose(self.H)),np.linalg.inv(s))

      self.prev_optimal_state_estimate = x_estimate = x_estimate + np.matmul(kalman_gain ,( measurement - np.matmul(self.H,x_estimate) ))

      self.P = np.matmul((np.identity(4) - np.matmul(kalman_gain,self.H)),self.P)

    return x_estimate

  def opencv_ground_truth(self):

    video = cv.VideoCapture('./car-overhead-1.avi') #reading in the data

    out = cv.VideoWriter('output_constant_vel_2_2_meas_0_1.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 20, (180,210)) #object reference to store the output video

    prev_x_avg = 0
    prev_y_avg = 0
    prev_vel_x = 0
    prev_vel_y = 0
    prev_time = time.time()

    overall_points = [] #ground truth tracked centroid positions.
    overall_estimate = []

    while video.isOpened():
      ret,frame = video.read() #reading in frame by frame
      if ret == True:
        frame = frame[30:,140:,:] #localizing the main part of the overall frame

        self.frame_row = frame.shape[0]
        self.frame_col = frame.shape[1]

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
          # cv.polylines(frame,[np.asarray(overall_points)],False,(0,255,0),1)

          #optimal state estimation
          dist = math.sqrt(pow(x_avg - prev_x_avg , 2) + pow(y_avg - prev_y_avg,2))
          elapsed_time = time.time() - prev_time

          theta = math.atan((y_avg - prev_y_avg) / (x_avg - prev_x_avg))
          prev_vel_x = velocity_x = dist/elapsed_time * math.cos(theta)
          prev_vel_y = velocity_y = dist/elapsed_time * math.sin(theta)


          estimate = self.kalman_filter_2d_simple(np.asarray([x_avg,y_avg,prev_vel_x,prev_vel_y]))
          overall_estimate.append([int(estimate[0]),int(estimate[1])])
          
        else:
          #that is when we did not have any measurment now in this period
          #then also we apply the same kalman filter for prediction but without any measurment
          estimate = self.kalman_filter_2d_simple(None)
          print(estimate)
          overall_estimate.append([int(estimate[0]),int(estimate[1])])
          # cv.polylines(frame,[np.asarray(overall_estimate)],False,(0,255,255),1)
        cv.polylines(frame,[np.asarray(overall_estimate)],False,(0,255,255),1)
        cv.imshow('Frame',np.uint8(frame))

        out.write(frame)
      
      #reading in the frame a particular deulay.
      if cv.waitKey(6) & 0xFF == ord('q'):
            break
    # pd.DataFrame(overall_points)
    video.release()
    cv.destroyAllWindows()

obj = Object_Tracking()
obj.opencv_ground_truth()