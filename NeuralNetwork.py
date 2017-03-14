import numpy as np
import matplotlib.pyplot as plt

def Sgn(y):
  if y > 0:
    return 1
  else:
    return -1

def Learn(TrainingSet, Class, Epochs, LearningRate):
    n = len(TrainingSet[:,0])
    m = len(TrainingSet[0,:])

    # Defining random weights
    w = 0.5*np.random.rand(1,m)
    a = LearningRate

    for i in range(0, Epochs):
        for j in range(0, n):
          x = TrainingSet[j,:]
          wsum = np.dot(w,x)

          y = Sgn(wsum)
          Error = Class[j] - y

          w += Error*a*x

    return w
  
X = np.random.rand(25,2)
Y = np.random.rand(25,2) + 1

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(X[:,0], X[:,1], s=10, c='b', marker="s")
ax1.scatter(Y[:,0],Y[:,1], s=10, c='r', marker="o")

# Creating training set.
Z = np.concatenate((X,Y))

# Classifying training set.
Classes = np.vstack((np.ones((25,1)), np.zeros((25,1))))

# Applying perceptron learning stage with 25 epochs.
Epochs = 25
LearningRate = 0.003
w = Learn(Z, Classes, Epochs, LearningRate)

print(w)
