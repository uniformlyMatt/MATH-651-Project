#! /usr/bin/env/python python3

''' This function takes in a binary image, finds all masked objects in the image, and saves
    each individual object in its own subimage file.
'''

import numpy as np
from scipy import ndimage
import cv2
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Read in the image.
filepath = 'D:/MATH651/MATH-651-Project/'
filename = 'validation_objects'
img = ndimage.imread(filepath + filename + '.png')

if img.ndim > 2:
    img = rgb2gray(img)

# Label all objects in the image.
[Labels, Num_features] = ndimage.measurements.label(img)
[m, n] = np.shape(img)

# This function takes in an image and returns the first respective rows and columns where
# a non-zero element is encountered.
def FindObjectLimits(im):
    [m, n] = np.shape(im)
    im = im*1
    
    # Scan through rows to find first row that contains a 1.
    imR = im.flatten('C')
    firstrow = np.where(imR == 1)
    topside = firstrow[0][0]
    
    # Scan through columns to find first column that contains a 1.
    imC = im.flatten('F')
    firstcol = np.where(imC == 1)
    leftside = firstcol[0][0]
    
    return leftside%n, topside%m

# This function fills holes in a binary image. The code is from 
# https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
def FillHoles(binary_image):
    # Copy the thresholded image.
    im_floodfill = binary_image.copy()
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = binary_image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    binary_image = binary_image | im_floodfill_inv
    
    return binary_image
    
# Now create a 128x128 subimage for each object in the original image.
subsize = 128
for i in range(1, Num_features-1):
    thisBlob = (Labels == i)*1
    [topside, leftside] = FindObjectLimits(thisBlob)
    
    # Set the limits for where we slice the object out of the image.
    if (topside - subsize/2 < 0):
        x_ind = 0
    elif (topside + subsize/2 > (m-1)):
        x_ind = (m-1) - subsize
    else:
        x_ind = topside - subsize/2
    
    if (leftside - subsize/2 < 0):
        y_ind = 0
    elif (leftside + subsize/2 > (n-1)):
        y_ind = (n-1) - subsize
    else:
        y_ind = leftside - subsize/2
    
    x_ind = np.int(x_ind)
    y_ind = np.int(y_ind)
    
    # Slice the object out of the image.
    subimage = thisBlob[x_ind:x_ind + (subsize-1),y_ind:y_ind + (subsize-1)]
    
    # Fill holes in the binary subimage.
    subimage = FillHoles(subimage)
 
    # Save subimage to png.
    temp_image = Image.fromarray(subimage.astype('uint8'))
    temp_image.save(filepath + 'TestingSet/NewObjects/testobject_' + str(i) + '.png')