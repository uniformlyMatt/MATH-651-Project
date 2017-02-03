import numpy as np

def Sgn(y):
  if y > 0:
    return 1
  else:
    return -1

def Learn(TrainingSet, Class, Epochs, LearningRate):
  n, m = TrainingSet.shape()
  
  # Defining random weights
  w = 0.5*np.random.rand(1,m)
  a = LearningRate
  
  for i in range(0, Epochs):
    for j in range(0, n):
      x = TrainingSet[j,:]
      wsum = np.dot(w,x)
      
      y = Sgn(wsum)
      Error = Class(j) - y
      
      w += Error*a*x
  
  return w
