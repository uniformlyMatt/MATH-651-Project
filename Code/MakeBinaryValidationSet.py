#! /usr/bin/env python3
'''
Created on Mar 22, 2017

@author: Matt
'''
##########################################################
# This code takes in grayscale edge-detected images and  #
# returns binary masked objects for use in validating    #
# our neural network.                                    #
##########################################################

import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

def MakeValidationImage(filepath, filename):
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    threshold = 24
    
    # Read in the image data.
    img = ndimage.imread(filepath + filename + '.png')
    
    if img.ndim > 2:
        img = rgb2gray(img)
    
    print(img.shape)
    
    # Create a mask for all indices where the grayscale values 
    # are above the chosen threshold.
    mask = img > threshold
    
    # Apply the mask to the image.
    img[~mask] = 0
    img[mask] = 255
    
    # Create filter array for removal of spurious isolated 8-connected pixels.
    A = np.array(81*[0])
    A[0:81:10] = 1
    A[4:81:9] = 1       # Set center pixels to one.
    
    # Split the array A into 9 filter vectors of the same length.
    filters = np.array(np.split(A,9))
    
    # Reshape the individual filters to apply them to the binary image. This loop removes
    # all spurious isolated 8-connected pixels.
    for i in range(9):
        Ifilter = filters[i].reshape((3,3))
        hitormiss = ndimage.morphology.binary_hit_or_miss(img, Ifilter).astype('bool')
        img[hitormiss] = 0
    
    # Perform an opening with 8-connected pixels.
    for i in range(9):
        Dfilter = filters[i].reshape((3,3))
        img = ndimage.morphology.binary_dilation(img, Dfilter, iterations=2).astype(img.dtype)
        img = ndimage.morphology.binary_erosion(img, Dfilter,iterations=2).astype(img.dtype)
        
    # Again remove spurious isolated 8-connected pixels.
    for i in range(9):
        Ifilter = filters[i].reshape((3,3))
        hitormiss = ndimage.morphology.binary_hit_or_miss(img, Ifilter).astype('bool')
        img[hitormiss] = 0
      
    # Close holes smaller than 3x3.
    img = ndimage.morphology.binary_closing(img, np.ones((3,3)), iterations = 1).astype(img.dtype)
    
    # Save the output as an image.
    outfile = Image.fromarray(255*img)
    outfile = outfile.convert('RGB')
    outfile.save(filepath + 'TestingSet/NewObjects/' + filename + '_binary.png')

    plt.imshow(img, cmap = 'gray')
    plt.show()

filepath = 'D:/MATH651/MATH-651-Project/'
filename = 'validation_objects'
MakeValidationImage(filepath, filename)