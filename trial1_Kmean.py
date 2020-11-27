# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author: atul
"""
#import lib.
import numpy as np
import cv2

#import the image
img = cv2.imread('input33.jpg')

#reshape the image
img2 = img.reshape((-1,3))

#convert img2 format uint8 to float32
img2 = np.float32(img2)

#define the criteria or the conditions for epsilon and maximum iteration value
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#now choose clusters
k = 4

#define no. of time to run
attempts = 10

#K-mean algo which returns: compactness,label and center
#we can take 2 type of flags for centers i.e. cv2.KMEANS_PP_CENTERS(for perticular centers) and cv2.KMEANS_RANDOM_CENTERS(for random)
ret,label,center = cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

#now chng the format of center to unsigned integer 8 for plotting
center = np.uint8(center)

#convert label or reshape the image, uint8 to original 
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imwrite('segmented_K_4.jpg', res2)


