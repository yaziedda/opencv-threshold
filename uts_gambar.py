import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('gambarxx.png')


grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Global Thresholding
ret1,global_threshold = cv.threshold(grey,127,255,cv.THRESH_BINARY)

# Otsu Tresholding
# ret2,th2 = cv.threshold(grey,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
# ret3,otsu_trreshold = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
_, otsu_trreshold = cv.threshold(grey, 127, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# Prewitt Tresholding
blur = cv.GaussianBlur(grey,(5,5),0)
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv.filter2D(blur, -1, kernelx)
img_prewitty = cv.filter2D(blur, -1, kernely)

images = [grey, 0, global_threshold,
          grey, 0, otsu_trreshold,
          grey, 0, img_prewitty]

titles = ['Greyscale Image','Histogram','Global Thresholding (v=127)',
          'Greyscale Image','Histogram',"Otsu Thresholding",
          'Greyscale Image','Histogram',"Prewitt-y Thresholding"]

for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()