# This code block reads in a geojson file as a Python dictionary, then prints the data type (polygon) 
# and coordinates of polygon vertices.
# pygeoj is developed and maintained by Karim Bahgat.

import pygeoj

filename = '...'
data = pygeoj.load(filename)
for feature in data:
  print feature.geometry.type
  print feature.geometry.coordinates
