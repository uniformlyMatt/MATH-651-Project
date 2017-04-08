

''' This script takes in a grayscale image, extracts all objects containing 
more than a specified number of pixels, and saves each object as an 
individual .png file. '''

import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
