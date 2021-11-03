import cv2
import numpy as np
from math import exp

import matplotlib.pyplot as plt

# Probability as a function of distance from the center derived
# from a gaussian distribution with mean = 0 and stdv = 1
scaledGaussian = lambda x : exp(-(1/2)*(x**2))

imgSize = 512
isotropicGrayscaleImage = np.zeros((imgSize,imgSize),np.uint8)

for i in range(imgSize):
  for j in range(imgSize):

    # find euclidian distance from center of image (imgSize/2,imgSize/2) 
    # and scale it to range of 0 to 2.5 as scaled Gaussian
    # returns highest probability for x=0 and approximately
    # zero probability for x > 2.5
    distanceFromCenter = np.linalg.norm(np.array([i-imgSize/2,j-imgSize/2]))
    distanceFromCenter = 2.5*distanceFromCenter/(imgSize/2)
    scaledGaussianProb = scaledGaussian(distanceFromCenter)
    isotropicGrayscaleImage[i,j] = np.clip(scaledGaussianProb*255,0,255)

# Convert Grayscale to HeatMap Using Opencv
plt.imshow(isotropicGrayscaleImage, cmap='hot', interpolation='nearest')
plt.show()

# isotropicGaussianHeatmapImage = cv2.applyColorMap(isotropicGrayscaleImage, 
#                                                   cv2.COLORMAP_JET)

# plt.imshow(isotropicGaussianHeatmapImage)
# plt.show()
# print(isotropicGaussianHeatmapImage.shape)