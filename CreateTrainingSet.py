'''
Created on Mar 10, 2017

@author: Matt
'''

from collections import defaultdict
import sys

from PIL import Image
import cv2
import pandas as pd
import numpy as np
import tifffile as tiff
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import matplotlib.pyplot

filepath = 'D:/MATH651/MATH-651-Project/'

df = pd.read_csv(filepath + 'train_wkt_v4.csv', header = 0)

bldg = df[df['ClassType']==1]
bldg = bldg[bldg['MultipolygonWKT']!='MULTIPOLYGON EMPTY']
wkt = bldg[bldg.ImageId == '6120_2_2'].iloc[0,2]
sMultiPolygon = shapely.wkt.loads(wkt)

grid_sizes = pd.read_csv(filepath + 'grid_sizes.csv', header = 0)
im1 = grid_sizes[grid_sizes['Unnamed: 0'] == '6120_2_2']
x_max = float(im1.Xmax)
y_min = float(im1.Ymin)

imName = filepath + 'three_band/6120_2_2.tif'
im_rgb = tiff.imread(imName).transpose([1,2,0])

im_size = im_rgb.shape[:2]

def get_scalars():
    h,w =im_size
    w_=w*(w/(w+1))/x_max
    h_=h*(h/(h+1))/y_min
    return w_,h_

xscaler, yscaler = get_scalars()

poly_scaled=shapely.affinity.scale(sMultiPolygon,xfact=xscaler,yfact=yscaler,origin=(0,0,0))

def mask_for_polygons(polygons):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

train_mask=mask_for_polygons(poly_scaled)

def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 2, axis=0)
    maxs = np.percentile(matrix, 98, axis=0) - mins
    matrix = (matrix - mins[None, :]) / (maxs[None, :])
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

# [2900:3200,2000:2300]

tiff.imshow(255 * scale_percentile(im_rgb))
matplotlib.pyplot.show()

def show_mask(m):
    # hack for nice display
    tiff.imshow(255 * np.stack([m, m, m]));
show_mask(train_mask)
matplotlib.pyplot.show()

img = Image.fromarray(train_mask)
img.save(filepath + 'training_image.png')